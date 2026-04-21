from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download

from training.data.types import PairedManifestRow

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _is_empty_path(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "null", "None"}:
        return True
    return False


def _has_usable_timestamps_payload(payload: dict[str, Any]) -> bool:
    timestamps = payload.get("timestamps")
    if not timestamps:
        return False
    for item in timestamps:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if text and item.get("end") is not None:
            return True
    return False


def load_manifest_rows(manifest_root: Path, manifest_glob: str) -> list[PairedManifestRow]:
    rows: list[PairedManifestRow] = []
    for path in sorted(manifest_root.glob(manifest_glob)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not _has_usable_timestamps_payload(payload):
                    continue
                rows.append(
                    PairedManifestRow(
                        key=str(payload["key"]),
                        country=str(payload.get("country", "unknown_country")),
                        split=str(payload.get("split", "unknown_split")),
                        transcription=str(payload.get("transcription", "")),
                        latent_shard_path=str(payload["latent_shard_path"]),
                        latent_row_idx=int(payload.get("latent_row_idx", 0)),
                        num_frames=None if payload.get("num_frames") is None else int(payload["num_frames"]),
                        speaker_prefix_frames=None
                        if payload.get("speaker_prefix_frames") is None
                        else int(payload["speaker_prefix_frames"]),
                        timestamps=payload.get("timestamps"),
                    )
                )
    if not rows:
        raise RuntimeError(
            f"No paired manifest rows found under {manifest_root} with glob {manifest_glob!r}."
        )
    return rows


def resolve_manifest_path(
    *,
    manifest_root: Path,
    country: str,
    split: str,
) -> Path:
    return manifest_root / "manifests" / country / split / "paired_manifest.jsonl"


def load_split_manifest_rows(
    *,
    manifest_root: Path,
    country: str,
    split: str,
) -> list[PairedManifestRow]:
    manifest_path = resolve_manifest_path(
        manifest_root=manifest_root,
        country=country,
        split=split,
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for country={country!r} split={split!r}: {manifest_path}"
        )

    rows: list[PairedManifestRow] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not _has_usable_timestamps_payload(payload):
                continue
            rows.append(
                PairedManifestRow(
                    key=str(payload["key"]),
                    country=str(payload.get("country", country)),
                    split=str(payload.get("split", split)),
                    transcription=str(payload.get("transcription", "")),
                    latent_shard_path=str(payload["latent_shard_path"]),
                    latent_row_idx=int(payload.get("latent_row_idx", 0)),
                    num_frames=None if payload.get("num_frames") is None else int(payload["num_frames"]),
                    speaker_prefix_frames=None
                    if payload.get("speaker_prefix_frames") is None
                    else int(payload["speaker_prefix_frames"]),
                    timestamps=payload.get("timestamps"),
                )
            )
    return rows


def resolve_manifest_root(dataset_cfg: dict[str, Any]) -> Path:
    local_root = dataset_cfg.get("local_dataset_root")
    if not _is_empty_path(local_root):
        path = Path(str(local_root)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    repo_id = dataset_cfg.get("repo_id")
    if _is_empty_path(repo_id):
        raise ValueError("dataset.repo_id must be set when local_dataset_root is not provided.")

    manifest_glob = str(dataset_cfg.get("manifest_glob", "manifests/*/*/paired_manifest.jsonl"))
    local_dir = snapshot_download(
        repo_id=str(repo_id),
        repo_type="dataset",
        revision=dataset_cfg.get("revision"),
        allow_patterns=[manifest_glob, "summary.json"],
    )
    return Path(local_dir).resolve()


def _load_tensor_from_bytes(blob: bytes) -> torch.Tensor:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu")


def _resolve_materialized_dtype(value: Any, *, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported materialized latent dtype: {value}")
    return mapping[normalized]


def _snapshot_country_split_latents(
    *,
    country: str,
    splits: list[str],
    dataset_cfg: dict[str, Any],
    parquet_cache_dir: Path,
    dataset_root: Path | None,
) -> dict[str, Path]:
    patterns = [f"latents/{country}/{split}/*.parquet" for split in splits]
    if dataset_root is not None:
        local_shards = {
            path.relative_to(dataset_root).as_posix(): path.resolve()
            for split in splits
            for path in sorted((dataset_root / "latents" / country / split).glob("*.parquet"))
        }
        if local_shards:
            return local_shards

    repo_id = dataset_cfg.get("repo_id")
    if _is_empty_path(repo_id):
        raise FileNotFoundError(
            f"No local parquet shards found for country={country!r} splits={splits!r} and dataset.repo_id is unset."
        )
    parquet_cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(
        snapshot_download(
            repo_id=str(repo_id),
            repo_type="dataset",
            revision=dataset_cfg.get("revision"),
            allow_patterns=patterns,
            local_dir=str(parquet_cache_dir),
            local_dir_use_symlinks=False,
        )
    ).resolve()
    downloaded_shards = {
        path.relative_to(snapshot_root).as_posix(): path.resolve()
        for split in splits
        for path in sorted((snapshot_root / "latents" / country / split).glob("*.parquet"))
    }
    if not downloaded_shards:
        raise FileNotFoundError(
            f"Downloaded snapshot contains no parquet shards for country={country!r} splits={splits!r}."
        )
    return downloaded_shards


def _materialized_sample_path(
    materialized_root: Path,
    *,
    country: str,
    split: str,
    key: str,
) -> Path:
    return materialized_root / country / split / f"{key}.pt"


def _materialize_shard_rows(
    *,
    shard_path: Path,
    latent_shard_path: str,
    materialized_root: Path,
    force_rematerialize: bool,
    materialize_speaker_prefix: bool,
    tensor_dtype: torch.dtype,
) -> tuple[int, int]:
    table = pq.read_table(shard_path)
    written = 0
    skipped = 0

    for row_idx, record in enumerate(table.to_pylist()):
        key = str(record["key"])
        country = str(record.get("country", "unknown_country"))
        split = str(record.get("split", "unknown_split"))
        sample_path = _materialized_sample_path(
            materialized_root,
            country=country,
            split=split,
            key=key,
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        if sample_path.exists() and not force_rematerialize:
            skipped += 1
            continue

        payload = {
            "key": key,
            "country": country,
            "split": split,
            "projected": _load_tensor_from_bytes(record["projected_bytes"]).to(dtype=tensor_dtype),
            "num_frames": int(record["num_frames"]),
            "latent_shard_path": latent_shard_path,
            "latent_row_idx": int(row_idx),
        }
        if materialize_speaker_prefix:
            payload["speaker_prefix_frames"] = int(record["speaker_prefix_frames"])
            payload["speaker_prefix_prequant"] = _load_tensor_from_bytes(
                record["speaker_prefix_prequant_bytes"]
            ).to(dtype=tensor_dtype)
        torch.save(payload, sample_path)
        written += 1

    return written, skipped


def _progress(iterable, *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="file")


def _materialized_country_has_samples(materialized_root: Path, *, country: str) -> bool:
    country_root = materialized_root / country
    return country_root.exists() and any(country_root.rglob("*.pt"))


def materialize_latent_dataset(
    *,
    dataset_cfg: dict[str, Any],
    materialized_root: Path,
    force_rematerialize: bool = False,
    cleanup_parquet_after_materialize: bool = False,
    tensor_dtype: str | torch.dtype | None = None,
) -> None:
    country = str(dataset_cfg["country"])
    if not force_rematerialize and _materialized_country_has_samples(materialized_root, country=country):
        print(
            f"[MATERIALIZE] using existing materialized latents under "
            f"{materialized_root / country}; set dataset.force_rematerialize=true to rebuild."
        )
        return None

    dataset_root = None
    local_root = dataset_cfg.get("local_dataset_root")
    if not _is_empty_path(local_root):
        dataset_root = Path(str(local_root)).expanduser().resolve()

    manifest_root = resolve_manifest_root(dataset_cfg)
    materialize_speaker_prefix = bool(dataset_cfg.get("materialize_speaker_prefix", True))
    resolved_tensor_dtype = _resolve_materialized_dtype(
        dataset_cfg.get("materialized_dtype", tensor_dtype),
        default=torch.bfloat16,
    )
    materialization_num_workers = max(1, int(dataset_cfg.get("materialization_num_workers", 1)))
    available_splits = []
    for split in ("train", "validation", "test"):
        if resolve_manifest_path(manifest_root=manifest_root, country=country, split=split).exists():
            available_splits.append(split)
    parquet_cache_dir = materialized_root / "_parquet_cache"
    print("[MATERIALIZE] downloading country/split parquet snapshot with huggingface_hub")
    shard_map = _snapshot_country_split_latents(
        country=country,
        splits=available_splits,
        dataset_cfg=dataset_cfg,
        parquet_cache_dir=parquet_cache_dir,
        dataset_root=dataset_root,
    )

    shard_items = [(shard_rel_path, shard_path) for shard_rel_path, shard_path in sorted(shard_map.items())]
    total_shards = len(shard_map)
    print(
        f"[MATERIALIZE] country={country} parquet_files={total_shards} "
        f"workers={materialization_num_workers} tensor_dtype={resolved_tensor_dtype}"
    )

    written = 0
    skipped = 0
    worker_jobs = [
        {
            "shard_path": shard_path,
            "latent_shard_path": shard_rel_path,
            "materialized_root": materialized_root,
            "force_rematerialize": force_rematerialize,
            "materialize_speaker_prefix": materialize_speaker_prefix,
            "tensor_dtype": resolved_tensor_dtype,
        }
        for shard_rel_path, shard_path in shard_items
    ]
    if materialization_num_workers > 1:
        max_workers = min(materialization_num_workers, len(worker_jobs), max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_materialize_shard_rows, **job) for job in worker_jobs]
            for future in _progress(futures, total=len(futures), desc="Materializing latent shards"):
                shard_written, shard_skipped = future.result()
                written += shard_written
                skipped += shard_skipped
    else:
        for job in _progress(worker_jobs, total=len(worker_jobs), desc="Materializing latent shards"):
            shard_written, shard_skipped = _materialize_shard_rows(**job)
            written += shard_written
            skipped += shard_skipped

    if cleanup_parquet_after_materialize and parquet_cache_dir.exists():
        for path in sorted(parquet_cache_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    print(
        f"[MATERIALIZE] country={country} shards={len(worker_jobs)} "
        f"written={written} skipped={skipped} workers={materialization_num_workers}"
    )
    return None
