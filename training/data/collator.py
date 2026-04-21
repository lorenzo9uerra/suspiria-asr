from __future__ import annotations

from dataclasses import dataclass

import torch

from training.data.alignment import build_delayed_target_stream


@dataclass(frozen=True)
class SpecialTokenIds:
    bos: int
    eos: int
    pad_wait: int
    word_start: int


class PackedLatentCollator:
    def __init__(
        self,
        *,
        tokenizer,
        special_tokens: SpecialTokenIds,
        left_pad_steps: int,
        delay_min_ms: int,
        delay_max_ms: int,
        step_ms: int = 80,
        feature_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.left_pad_steps = int(left_pad_steps)
        self.delay_min_ms = int(delay_min_ms)
        self.delay_max_ms = int(delay_max_ms)
        self.step_ms = int(step_ms)
        self.feature_dtype = feature_dtype

        if self.delay_min_ms % self.step_ms != 0 or self.delay_max_ms % self.step_ms != 0:
            raise ValueError("Delay range must be a multiple of the 80 ms latent step.")

    def _sample_delay_steps(self) -> int:
        low = self.delay_min_ms // self.step_ms
        high = self.delay_max_ms // self.step_ms
        return int(torch.randint(low, high + 1, (1,)).item())

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        aligned = []
        for sample in samples:
            delay_steps = self._sample_delay_steps()
            aligned.append(
                build_delayed_target_stream(
                    key=str(sample["key"]),
                    latents=sample["projected"],
                    transcript=str(sample["transcription"]),
                    timestamps=sample.get("timestamps"),
                    tokenizer=self.tokenizer,
                    bos_token_id=self.special_tokens.bos,
                    eos_token_id=self.special_tokens.eos,
                    pad_wait_token_id=self.special_tokens.pad_wait,
                    word_start_token_id=self.special_tokens.word_start,
                    delay_steps=delay_steps,
                    left_pad_steps=self.left_pad_steps,
                    step_ms=self.step_ms,
                )
            )

        seq_lens = torch.tensor([item.token_ids.numel() - 1 for item in aligned], dtype=torch.long)
        delay_steps = torch.tensor([item.delay_steps for item in aligned], dtype=torch.long)

        packed_input_ids = []
        packed_labels = []
        packed_audio_features = []
        packed_position_ids = []
        cu_seqlens = [0]

        for item in aligned:
            length = int(item.token_ids.numel() - 1)
            if length <= 0:
                raise ValueError(f"Aligned sample {item.key!r} is too short to shift into input/target pairs.")

            input_ids = item.token_ids[:-1]
            labels = item.token_ids[1:]
            audio_features = item.audio_features[:-1]
            if input_ids.numel() != labels.numel() or labels.numel() != audio_features.shape[0]:
                raise RuntimeError(f"Shifted alignment invariant failed for sample {item.key!r}.")

            packed_input_ids.append(input_ids)
            packed_labels.append(labels)
            packed_audio_features.append(audio_features.to(dtype=self.feature_dtype))
            # RoPE positions restart at each packed sequence boundary.
            packed_position_ids.append(torch.arange(length, dtype=torch.long))
            cu_seqlens.append(cu_seqlens[-1] + length)

        return {
            "seq_lens": seq_lens,
            "delay_steps": delay_steps,
            "packed_input_ids": torch.cat(packed_input_ids, dim=0),
            "packed_labels": torch.cat(packed_labels, dim=0),
            "packed_audio_features": torch.cat(packed_audio_features, dim=0),
            "packed_position_ids": torch.cat(packed_position_ids, dim=0),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        }
