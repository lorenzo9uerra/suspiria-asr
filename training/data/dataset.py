from __future__ import annotations

import torch
from torch.utils.data import Dataset

from training.data.materialize_latents import load_split_manifest_rows
from training.data.types import PairedManifestRow


class MaterializedLatentDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_root,
        materialized_root,
        split: str,
        country: str,
    ):
        self.materialized_root = materialized_root
        self.samples: list[PairedManifestRow] = load_split_manifest_rows(
            manifest_root=manifest_root,
            country=country,
            split=split,
        )
        if not self.samples:
            raise RuntimeError(
                f"No manifest rows found for country={country!r} split={split!r} under {manifest_root}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        sample_path = self.materialized_root / sample.country / sample.split / f"{sample.key}.pt"
        payload = torch.load(sample_path, map_location="cpu")
        projected = payload["projected"].float()
        if sample.num_frames is None:
            raise ValueError(f"Manifest row for {sample.key} is missing num_frames.")
        projected = projected[: sample.num_frames].contiguous()
        return {
            "key": sample.key,
            "transcription": sample.transcription,
            "timestamps": sample.timestamps,
            "projected": projected,
        }
