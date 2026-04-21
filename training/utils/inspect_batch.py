from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.collator import PackedLatentCollator, SpecialTokenIds
from training.data.materialize_latents import (
    _is_empty_path,
    _materialize_shard_rows,
    _resolve_materialized_dtype,
    load_split_manifest_rows,
    resolve_manifest_root,
)
from training.data.types import PairedManifestRow
from training.tokenizer import load_tokenizer
from training.utils.config import resolve_torch_dtype
from training.utils.data import ensure_materialized_dataset


class InspectLatentDataset:
    def __init__(
        self,
        *,
        samples: list[PairedManifestRow],
        materialized_root: Path,
    ) -> None:
        self.samples = samples
        self.materialized_root = materialized_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        sample_path = self.materialized_root / sample.country / sample.split / f"{sample.key}.pt"
        payload = torch.load(sample_path, map_location="cpu")
        projected = payload["projected"]
        if sample.num_frames is None:
            raise ValueError(f"Manifest row for {sample.key} is missing num_frames.")
        return {
            "key": sample.key,
            "transcription": sample.transcription,
            "timestamps": sample.timestamps,
            "projected": projected[: sample.num_frames].contiguous(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect packed batch construction for latent/text training.")
    parser.add_argument("--config-path", default="configs/training.yaml")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--country", default=None, help="Override dataset.country from config.")
    parser.add_argument("--index", type=int, default=0, help="Start sample index within the selected split.")
    parser.add_argument("--key", default=None, help="Start sample key. Overrides --index when set.")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of samples to collate.")
    parser.add_argument("--delay-ms", type=int, default=None, help="Fixed delay in ms. Defaults to dataset.delay_max_ms.")
    parser.add_argument("--max-steps", type=int, default=160, help="Max per-sequence steps to print.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--one-shard",
        dest="one_shard",
        action="store_true",
        default=True,
        help="Use only the parquet shard containing the selected row. This is the default.",
    )
    mode_group.add_argument(
        "--full-dataset",
        dest="one_shard",
        action="store_false",
        help="Use the normal full-dataset materialization path before inspecting.",
    )
    parser.add_argument("--output-path", default=None, help="Optional path to write the Markdown report.")
    return parser.parse_args()


def load_cfg(config_path: str) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Expected training config to resolve to a mapping.")
    return plain


def sample_path(materialized_root: Path, row: PairedManifestRow) -> Path:
    return materialized_root / row.country / row.split / f"{row.key}.pt"


def select_row(rows: list[PairedManifestRow], *, index: int, key: str | None, split: str) -> int:
    if key:
        for idx, row in enumerate(rows):
            if row.key == key:
                return idx
        raise KeyError(f"Sample key {key!r} not found in split={split!r}.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Sample index {index} out of range for split={split!r} with {len(rows)} rows.")
    return int(index)


def download_one_parquet_shard(
    *,
    dataset_cfg: dict[str, Any],
    dataset_root: Path | None,
    latent_shard_path: str,
    cache_root: Path,
) -> Path:
    if dataset_root is not None:
        local_path = dataset_root / latent_shard_path
        if not local_path.exists():
            raise FileNotFoundError(f"Local latent shard not found: {local_path}")
        return local_path.resolve()

    repo_id = dataset_cfg.get("repo_id")
    if _is_empty_path(repo_id):
        raise ValueError("dataset.repo_id must be set when --one-shard is used without local_dataset_root.")

    cache_root.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=str(repo_id),
            repo_type="dataset",
            revision=dataset_cfg.get("revision"),
            filename=latent_shard_path,
            local_dir=str(cache_root),
            local_dir_use_symlinks=False,
        )
    ).resolve()


def build_dataset(
    *,
    cfg: dict[str, Any],
    split: str,
    index: int,
    key: str | None,
    batch_size: int,
    one_shard: bool,
) -> tuple[InspectLatentDataset, int, str | None]:
    dataset_cfg = cfg["dataset"]
    country = str(dataset_cfg["country"])
    manifest_root = resolve_manifest_root(dataset_cfg)
    rows = load_split_manifest_rows(manifest_root=manifest_root, country=country, split=split)
    if not rows:
        raise RuntimeError(f"No manifest rows found for country={country!r} split={split!r}.")

    selected_index = select_row(rows, index=index, key=key, split=split)
    selected_row = rows[selected_index]
    materialized_root = Path(dataset_cfg.get("materialized_latents_dir", "out/materialized_latents")).expanduser().resolve()

    if not one_shard:
        materialized_root = ensure_materialized_dataset(cfg)
        return InspectLatentDataset(samples=rows, materialized_root=materialized_root), selected_index, None

    shard_rows = [row for row in rows if row.latent_shard_path == selected_row.latent_shard_path]
    remapped_index = select_row(shard_rows, index=0, key=selected_row.key, split=split)
    wanted_rows = shard_rows[remapped_index : remapped_index + batch_size]
    if len(wanted_rows) < batch_size:
        wanted_rows = shard_rows[max(0, len(shard_rows) - batch_size) :]
        remapped_index = select_row(wanted_rows, index=0, key=selected_row.key, split=split)

    force = bool(dataset_cfg.get("force_rematerialize", False))
    missing = [row for row in wanted_rows if not sample_path(materialized_root, row).exists()]
    if missing or force:
        local_root = dataset_cfg.get("local_dataset_root")
        dataset_root = None if _is_empty_path(local_root) else Path(str(local_root)).expanduser().resolve()
        shard_path = download_one_parquet_shard(
            dataset_cfg=dataset_cfg,
            dataset_root=dataset_root,
            latent_shard_path=selected_row.latent_shard_path,
            cache_root=materialized_root / "_inspect_one_shard_cache",
        )
        written, skipped = _materialize_shard_rows(
            shard_path=shard_path,
            latent_shard_path=selected_row.latent_shard_path,
            materialized_root=materialized_root,
            force_rematerialize=force,
            materialize_speaker_prefix=bool(dataset_cfg.get("materialize_speaker_prefix", True)),
            tensor_dtype=_resolve_materialized_dtype(
                dataset_cfg.get("materialized_dtype", cfg["runtime"].get("data_dtype", "bf16"))
            ),
            materialization_batch_size=int(dataset_cfg.get("materialization_batch_size", 128)),
        )
        print(
            f"[INSPECT_BATCH] one-shard materialized {selected_row.latent_shard_path}: "
            f"written={written} skipped={skipped}"
        )
    else:
        print(
            f"[INSPECT_BATCH] using existing materialized samples for {selected_row.latent_shard_path}; "
            "skipping parquet download."
        )

    return InspectLatentDataset(samples=wanted_rows, materialized_root=materialized_root), 0, selected_row.latent_shard_path


def token_kind(token_id: int, special_tokens: SpecialTokenIds) -> str:
    if token_id == special_tokens.bos:
        return "BOS"
    if token_id == special_tokens.eos:
        return "EOS"
    if token_id == special_tokens.pad_wait:
        return "P"
    if token_id == special_tokens.word_start:
        return "W"
    return "TEXT"


def render_token(token_id: int, tokenizer, special_tokens: SpecialTokenIds) -> str:
    kind = token_kind(token_id, special_tokens)
    if kind != "TEXT":
        return f"[{kind}]"
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    decoded = decoded.replace("\n", "\\n").replace("|", "\\|")
    return f"{token} / {decoded!r}"


def audio_region(step: int, *, left_pad_steps: int, real_steps: int) -> tuple[str, str]:
    latent_idx = step - left_pad_steps
    if step < left_pad_steps:
        return "left_pad", "-"
    if 0 <= latent_idx < real_steps:
        return "real", str(latent_idx)
    return "right_pad", "-"


def render_batch_report(
    *,
    cfg: dict[str, Any],
    split: str,
    shard_path: str | None,
    rows: list[PairedManifestRow],
    samples: list[dict[str, object]],
    batch: dict[str, torch.Tensor],
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
    max_steps: int,
) -> str:
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    left_pad_steps = int(cfg["dataset"].get("left_pad_steps", 0))
    seq_lens = batch["seq_lens"].tolist()
    cu_seqlens = batch["cu_seqlens"].tolist()
    max_seq_len = int(batch["max_seq_len"].item())
    total_tokens = int(batch["packed_input_ids"].numel())

    lines = [
        "# Packed Batch Inspection",
        "",
        f"- split: `{split}`",
        f"- country: `{cfg['dataset']['country']}`",
        f"- shard: `{shard_path or 'full-dataset'}`",
        f"- batch_size: `{len(samples)}`",
        f"- delay_ms: `{delay_ms}`",
        f"- step_ms: `{step_ms}`",
        f"- left_pad_steps: `{left_pad_steps}`",
        f"- total_packed_tokens: `{total_tokens}`",
        f"- max_seq_len: `{max_seq_len}`",
        f"- seq_lens: `{seq_lens}`",
        f"- cu_seqlens: `{cu_seqlens}`",
        "",
        "## Batch Samples",
        "",
        "| batch_idx | key | manifest_split | real_steps | packed_start | packed_end | seq_len |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]

    for batch_idx, sample in enumerate(samples):
        start = int(cu_seqlens[batch_idx])
        end = int(cu_seqlens[batch_idx + 1])
        key = str(sample["key"])
        manifest_row = rows[batch_idx]
        real_steps = int(sample["projected"].shape[0])
        lines.append(
            f"| {batch_idx} | `{key}` | `{manifest_row.split}` | {real_steps} | {start} | {end} | {seq_lens[batch_idx]} |"
        )

    input_ids = batch["packed_input_ids"].tolist()
    labels = batch["packed_labels"].tolist()
    position_ids = batch["packed_position_ids"].tolist()

    for batch_idx, sample in enumerate(samples):
        start = int(cu_seqlens[batch_idx])
        end = int(cu_seqlens[batch_idx + 1])
        real_steps = int(sample["projected"].shape[0])
        shown = min(max_steps, end - start)
        lines.extend(
            [
                "",
                f"## Sequence {batch_idx}: `{sample['key']}`",
                "",
                f"- packed_range: `[{start}, {end})`",
                f"- seq_len: `{end - start}`",
                f"- shown_steps: `{shown}`",
                "",
                "| local_step | global_idx | cu_start | cu_end | pos_id | audio | latent_idx | input_token | label_token |",
                "|---:|---:|---:|---:|---:|---|---:|---|---|",
            ]
        )
        for local_step in range(shown):
            global_idx = start + local_step
            input_id = int(input_ids[global_idx])
            label_id = int(labels[global_idx])
            region, latent_idx = audio_region(local_step, left_pad_steps=left_pad_steps, real_steps=real_steps)
            lines.append(
                "| "
                f"{local_step} | "
                f"{global_idx} | "
                f"{start} | "
                f"{end} | "
                f"{int(position_ids[global_idx])} | "
                f"{region} | "
                f"{latent_idx} | "
                f"{render_token(input_id, tokenizer, special_tokens)} | "
                f"{render_token(label_id, tokenizer, special_tokens)} |"
            )
        if shown < end - start:
            lines.append("")
            lines.append(f"_Truncated: {end - start - shown} additional steps not shown._")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_path)
    if args.country is not None:
        cfg["dataset"]["country"] = args.country

    batch_size = int(args.batch_size if args.batch_size is not None else cfg["optimization"].get("batch_size", 4))
    delay_ms = int(args.delay_ms if args.delay_ms is not None else cfg["dataset"].get("delay_max_ms", 2400))
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    if delay_ms % step_ms != 0:
        raise ValueError(f"--delay-ms must be a multiple of step_ms={step_ms}.")

    resolved_tokenizer = load_tokenizer(cfg["tokenizer"])
    tokenizer = resolved_tokenizer.tokenizer
    special_tokens = SpecialTokenIds(
        bos=resolved_tokenizer.bos_token_id,
        eos=resolved_tokenizer.eos_token_id,
        pad_wait=resolved_tokenizer.pad_wait_token_id,
        word_start=resolved_tokenizer.word_start_token_id,
    )

    dataset, start_index, shard_path = build_dataset(
        cfg=cfg,
        split=args.split,
        index=int(args.index),
        key=args.key,
        batch_size=batch_size,
        one_shard=bool(args.one_shard),
    )
    sample_indices = list(range(start_index, min(len(dataset), start_index + batch_size)))
    if len(sample_indices) < batch_size and len(dataset) >= batch_size:
        sample_indices = list(range(len(dataset) - batch_size, len(dataset)))
    samples = [dataset[idx] for idx in sample_indices]
    rows = [dataset.samples[idx] for idx in sample_indices]
    if not samples:
        raise RuntimeError("No samples available for batch inspection.")

    feature_dtype = resolve_torch_dtype(cfg["runtime"].get("data_dtype", "bf16"), default=None) or torch.bfloat16
    collator = PackedLatentCollator(
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        left_pad_steps=int(cfg["dataset"].get("left_pad_steps", 0)),
        delay_min_ms=delay_ms,
        delay_max_ms=delay_ms,
        step_ms=step_ms,
        feature_dtype=feature_dtype,
    )
    batch = collator(samples)

    report = render_batch_report(
        cfg=cfg,
        split=args.split,
        shard_path=shard_path,
        rows=rows,
        samples=samples,
        batch=batch,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
        max_steps=int(args.max_steps),
    )

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[INSPECT_BATCH] wrote {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
