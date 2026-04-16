from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.data.collator import PackedLatentCollator, SpecialTokenIds
from training.data.dataset import MaterializedLatentDataset
from training.data.materialize_latents import (
    _is_empty_path,
    materialize_latent_dataset,
    resolve_manifest_path,
    resolve_manifest_root,
)
from training.utils.config import resolve_torch_dtype


def ensure_materialized_dataset(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg["dataset"]
    latents_path = dataset_cfg.get("latents_path")
    materialized_root = Path(
        dataset_cfg.get("materialized_latents_dir", "out/materialized_latents")
    ).expanduser().resolve()
    if _is_empty_path(latents_path):
        materialize_latent_dataset(
            dataset_cfg=dataset_cfg,
            materialized_root=materialized_root,
            force_rematerialize=bool(dataset_cfg.get("force_rematerialize", False)),
            cleanup_parquet_after_materialize=bool(dataset_cfg.get("cleanup_parquet_after_materialize", False)),
        )
        return materialized_root
    return Path(str(latents_path)).expanduser().resolve()


def build_dataloader(
    *,
    cfg: dict[str, Any],
    tokenizer,
    special_tokens: SpecialTokenIds,
    materialized_root: Path,
    split: str,
    manifest_root: Path | None = None,
) -> DataLoader:
    split_country = str(cfg["dataset"]["country"])
    if manifest_root is None:
        manifest_root = resolve_manifest_root(cfg["dataset"])
    dataset = MaterializedLatentDataset(
        manifest_root=manifest_root,
        materialized_root=materialized_root,
        split=split,
        country=split_country,
    )
    collator = PackedLatentCollator(
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        left_pad_steps=int(cfg["dataset"].get("left_pad_steps", 0)),
        delay_min_ms=int(cfg["dataset"].get("delay_min_ms", 80)),
        delay_max_ms=int(cfg["dataset"].get("delay_max_ms", 2400)),
        step_ms=int(cfg["dataset"].get("step_ms", 80)),
        feature_dtype=resolve_torch_dtype(
            cfg["runtime"].get("data_dtype", "bf16"),
            default=None,
        )
        or torch.bfloat16,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["optimization"].get("batch_size", 4)),
        shuffle=split == "train",
        num_workers=int(cfg["runtime"].get("num_workers", 0)),
        pin_memory=bool(cfg["runtime"].get("pin_memory", False)),
        collate_fn=collator,
    )


def build_raw_dataloader(
    *,
    cfg: dict[str, Any],
    materialized_root: Path,
    split: str,
    manifest_root: Path | None = None,
) -> DataLoader:
    split_country = str(cfg["dataset"]["country"])
    if manifest_root is None:
        manifest_root = resolve_manifest_root(cfg["dataset"])
    dataset = MaterializedLatentDataset(
        manifest_root=manifest_root,
        materialized_root=materialized_root,
        split=split,
        country=split_country,
    )
    wer_cfg = cfg.get("wer", {})
    return DataLoader(
        dataset,
        batch_size=int(wer_cfg.get("batch_size", cfg["optimization"].get("batch_size", 4))),
        shuffle=False,
        num_workers=int(cfg["runtime"].get("num_workers", 0)),
        pin_memory=bool(cfg["runtime"].get("pin_memory", False)),
        collate_fn=lambda samples: samples,
    )


def resolve_manifest_split(
    *,
    manifest_root: Path,
    country: str,
    split: str,
) -> str:
    manifest_path = resolve_manifest_path(
        manifest_root=manifest_root,
        country=country,
        split=split,
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for country={country!r} split={split!r}: {manifest_path}"
        )
    if manifest_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Manifest is empty for country={country!r} split={split!r}: {manifest_path}"
        )
    return split


def discover_materialized_splits(
    *,
    manifest_root: Path,
    country: str,
) -> set[str]:
    split_names: set[str] = set()
    for split in ("train", "validation", "test"):
        try:
            resolve_manifest_split(
                manifest_root=manifest_root,
                country=country,
                split=split,
            )
            split_names.add(split)
        except FileNotFoundError:
            continue
    return split_names
