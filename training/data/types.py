from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PairedManifestRow:
    key: str
    country: str
    split: str
    transcription: str
    latent_shard_path: str
    latent_row_idx: int
    num_frames: int | None
    speaker_prefix_frames: int | None
    timestamps: list[dict[str, Any]] | None

@dataclass
class AlignedSample:
    key: str
    token_ids: torch.Tensor
    audio_features: torch.Tensor
    delay_steps: int
