from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch

from training.data.types import AlignedSample


def _timestamp_to_step(seconds: float, *, step_ms: int, delay_steps: int, left_pad_steps: int) -> int:
    ms = max(0.0, float(seconds) * 1000.0)
    step = max(0, math.ceil(ms / step_ms) - 1)
    return left_pad_steps + step + delay_steps


def _normalize_timestamp_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _build_groups_from_timestamps(
    *,
    timestamps: list[dict[str, Any]] | None,
    transcript: str,
    tokenizer,
    step_ms: int,
    delay_steps: int,
    left_pad_steps: int,
    fallback_real_steps: int,
) -> list[tuple[int, list[int]]]:
    if not timestamps:
        emission_step = left_pad_steps + max(0, fallback_real_steps - 1) + delay_steps
        token_ids = tokenizer.encode(transcript, add_special_tokens=False)
        return [(emission_step, token_ids)] if token_ids else []

    grouped_words: "OrderedDict[int, list[str]]" = OrderedDict()
    for item in timestamps:
        text = _normalize_timestamp_text(item.get("text", ""))
        if not text:
            continue
        emission_step = _timestamp_to_step(
            float(item.get("end", 0.0)),
            step_ms=step_ms,
            delay_steps=delay_steps,
            left_pad_steps=left_pad_steps,
        )
        grouped_words.setdefault(emission_step, []).append(text)

    groups: list[tuple[int, list[int]]] = []
    for emission_step, words in grouped_words.items():
        joined = " ".join(words)
        token_ids = tokenizer.encode(joined, add_special_tokens=False)
        if token_ids:
            groups.append((emission_step, token_ids))
    return groups


def build_delayed_target_stream(
    *,
    key: str,
    latents: torch.Tensor,
    transcript: str,
    timestamps: list[dict[str, Any]] | None,
    tokenizer,
    bos_token_id: int,
    eos_token_id: int,
    pad_wait_token_id: int,
    word_start_token_id: int,
    delay_steps: int,
    left_pad_steps: int,
    step_ms: int = 80,
) -> AlignedSample:
    real_steps = int(latents.shape[0])
    groups = _build_groups_from_timestamps(
        timestamps=timestamps,
        transcript=transcript,
        tokenizer=tokenizer,
        step_ms=step_ms,
        delay_steps=delay_steps,
        left_pad_steps=left_pad_steps,
        fallback_real_steps=real_steps,
    )

    targets: list[int] = []
    audio_steps: list[torch.Tensor] = []
    pending_tokens: list[int] = []
    group_idx = 0
    time_step = 0
    eos_emitted = False
    max_steps = left_pad_steps + real_steps + delay_steps + sum(len(ids) + 2 for _, ids in groups) + 8

    def latent_for_step(step_idx: int) -> torch.Tensor:
        latent_idx = step_idx - left_pad_steps
        if 0 <= latent_idx < real_steps:
            return latents[latent_idx]
        return torch.zeros_like(latents[0])

    while not eos_emitted:
        if time_step > max_steps:
            raise RuntimeError(f"Alignment overflow for sample {key}: failed to emit EOS within {max_steps} steps.")

        audio_steps.append(latent_for_step(time_step))

        if time_step == 0:
            targets.append(int(bos_token_id))
        elif pending_tokens:
            targets.append(int(pending_tokens.pop(0)))
        elif group_idx < len(groups) and groups[group_idx][0] <= time_step:
            token_ids = groups[group_idx][1]
            group_idx += 1
            pending_tokens = [int(tok) for tok in token_ids]
            targets.append(int(word_start_token_id))
        elif group_idx >= len(groups):
            targets.append(int(eos_token_id))
            eos_emitted = True
        else:
            targets.append(int(pad_wait_token_id))
        time_step += 1

    target_tensor = torch.tensor(targets, dtype=torch.long)
    input_ids = torch.empty_like(target_tensor)
    input_ids[0] = int(bos_token_id)
    if target_tensor.numel() > 1:
        input_ids[1:] = target_tensor[:-1]

    return AlignedSample(
        key=key,
        input_ids=input_ids,
        labels=target_tensor,
        audio_features=torch.stack(audio_steps, dim=0),
        delay_steps=int(delay_steps),
    )

