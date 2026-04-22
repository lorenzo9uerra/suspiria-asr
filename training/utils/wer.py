from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import torch

from training.data.collator import SpecialTokenIds


def remove_symbols(text: str, *, remove_diacritics: bool) -> str:
    normalized = unicodedata.normalize("NFKD" if remove_diacritics else "NFKC", text)
    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if remove_diacritics and category == "Mn":
            continue
        if category[0] in "MSP":
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


@dataclass(frozen=True)
class WERNormalizer:
    remove_diacritics: bool = False

    def __call__(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        text = remove_symbols(text, remove_diacritics=self.remove_diacritics).lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def _edit_distance(ref_units: list[str], hyp_units: list[str]) -> int:
    previous = list(range(len(hyp_units) + 1))
    for i, ref_unit in enumerate(ref_units, start=1):
        current = [i]
        for j, hyp_unit in enumerate(hyp_units, start=1):
            substitution = previous[j - 1] + (0 if ref_unit == hyp_unit else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def wer_stats(reference: str, hypothesis: str, normalizer: WERNormalizer) -> tuple[int, int]:
    ref_words = normalizer(reference).split()
    hyp_words = normalizer(hypothesis).split()
    return _edit_distance(ref_words, hyp_words), len(ref_words)


def cer_stats(
    reference: str,
    hypothesis: str,
    normalizer: WERNormalizer,
    *,
    ignore_spaces: bool,
) -> tuple[int, int]:
    normalized_ref = normalizer(reference)
    normalized_hyp = normalizer(hypothesis)
    if ignore_spaces:
        normalized_ref = normalized_ref.replace(" ", "")
        normalized_hyp = normalized_hyp.replace(" ", "")
    return _edit_distance(list(normalized_ref), list(normalized_hyp)), len(normalized_ref)


def compute_wer(total_errors: int, total_ref_words: int) -> float:
    if total_ref_words == 0:
        return float("nan")
    return float(total_errors) / float(total_ref_words)


def decode_generated_tokens(
    token_ids: list[int],
    *,
    tokenizer,
    special_tokens: SpecialTokenIds,
) -> str:
    chunks: list[str] = []
    current: list[int] = []

    def flush_current() -> None:
        if current:
            chunks.append(tokenizer.decode(current, skip_special_tokens=True))
            current.clear()

    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == special_tokens.eos:
            break
        if token_id in (special_tokens.bos, special_tokens.pad_wait):
            continue
        if token_id == special_tokens.word_start:
            flush_current()
            continue
        current.append(token_id)
    flush_current()
    return re.sub(r"\s+", " ", " ".join(piece.strip() for piece in chunks if piece.strip())).strip()


def _sample_audio_at_step(
    *,
    sample: dict[str, Any],
    step: int,
    left_pad_steps: int,
    allowed_steps: int,
    feature_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    latents = sample["projected"]
    latent_idx = step - left_pad_steps
    if 0 <= latent_idx < int(latents.shape[0]) and step < allowed_steps:
        return latents[latent_idx].to(dtype=dtype)
    return torch.zeros(feature_dim, dtype=dtype)


@torch.no_grad()
def generate_batch_greedy(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    *,
    tokenizer,
    special_tokens: SpecialTokenIds,
    device: torch.device,
    data_dtype: torch.dtype,
    left_pad_steps: int,
    delay_steps: int,
    flush_steps: int,
    max_decode_steps: int | None,
) -> list[str]:
    if not samples:
        return []

    model_was_training = model.training
    model.eval()

    batch_size = len(samples)
    feature_dim = int(samples[0]["projected"].shape[-1])
    allowed_steps = torch.tensor(
        [
            int(left_pad_steps) + int(sample["projected"].shape[0]) + int(flush_steps)
            for sample in samples
        ],
        dtype=torch.long,
        device=device,
    )
    max_steps = int(allowed_steps.max().item())
    if max_steps <= 0:
        return ["" for _ in samples]

    prefix_len = 1 + int(left_pad_steps) + int(delay_steps)
    max_steps = max(max_steps, prefix_len)
    if max_decode_steps is not None:
        max_steps = min(max_steps, prefix_len + int(max_decode_steps))
    prefill_len = prefix_len
    delay_tensor = torch.full((batch_size,), int(delay_steps), dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in samples]
    kv_cache = None
    prev_tokens = torch.full((batch_size,), int(special_tokens.bos), dtype=torch.long, device=device)

    prefix_input_ids = torch.full(
        (batch_size, prefill_len),
        int(special_tokens.pad_wait),
        dtype=torch.long,
        device=device,
    )
    prefix_input_ids[:, 0] = int(special_tokens.bos)
    prefix_audio = torch.stack(
        [
            torch.stack(
                [
                    _sample_audio_at_step(
                        sample=sample,
                        step=step,
                        left_pad_steps=left_pad_steps,
                        allowed_steps=int(allowed_steps[idx].item()),
                        feature_dim=feature_dim,
                        dtype=data_dtype,
                    )
                    for step in range(prefill_len)
                ],
                dim=0,
            )
            for idx, sample in enumerate(samples)
        ],
        dim=0,
    ).to(device)
    prefix_position_ids = torch.arange(prefill_len, dtype=torch.long, device=device).unsqueeze(0).expand(
        batch_size,
        -1,
    )

    logits, kv_cache = model.forward_generate_prefill(
        input_ids=prefix_input_ids,
        audio_features=prefix_audio,
        position_ids=prefix_position_ids,
        delay_steps=delay_tensor,
    )
    next_tokens = logits[:, -1, :].argmax(dim=-1)
    first_prediction_step = prefill_len - 1
    first_active = first_prediction_step < allowed_steps
    if prefill_len == prefix_len:
        for idx in range(batch_size):
            if not bool(first_active[idx].item()):
                continue
            token_id = int(next_tokens[idx].item())
            generated[idx].append(token_id)
            prev_tokens[idx] = token_id
            if token_id == special_tokens.eos:
                done[idx] = True

    for step in range(prefix_len, max_steps):
        within_budget = torch.arange(batch_size, device=device) >= 0
        within_budget &= step < allowed_steps
        active = within_budget & (~done)

        audio_step = torch.stack(
            [
                _sample_audio_at_step(
                    sample=sample,
                    step=step,
                    left_pad_steps=left_pad_steps,
                    allowed_steps=int(allowed_steps[idx].item()),
                    feature_dim=feature_dim,
                    dtype=data_dtype,
                )
                for idx, sample in enumerate(samples)
            ],
            dim=0,
        ).to(device)
        position_ids = torch.full((batch_size,), int(step), dtype=torch.long, device=device)
        logits, kv_cache = model.forward_generate_step(
            input_ids=prev_tokens,
            audio_features=audio_step,
            position_ids=position_ids,
            delay_steps=delay_tensor,
            kv_cache=kv_cache,
        )
        next_tokens = logits.argmax(dim=-1)

        for idx in range(batch_size):
            if not bool(active[idx].item()):
                continue
            token_id = int(next_tokens[idx].item())
            generated[idx].append(token_id)
            prev_tokens[idx] = token_id
            if token_id == special_tokens.eos:
                done[idx] = True

        if bool((done | (~within_budget)).all().item()):
            break

    if model_was_training:
        model.train()

    return [
        decode_generated_tokens(tokens, tokenizer=tokenizer, special_tokens=special_tokens)
        for tokens in generated
    ]
