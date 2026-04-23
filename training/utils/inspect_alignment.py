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

from training.data.alignment import build_delayed_target_stream
from training.data.collator import SpecialTokenIds
from training.data.materialize_latents import (
    _is_empty_path,
    _materialize_shard_rows,
    _resolve_materialized_dtype,
    load_split_manifest_rows,
    resolve_manifest_root,
)
from training.data.types import PairedManifestRow
from training.tokenizer import load_tokenizer
from training.utils.data import ensure_materialized_dataset
from training.utils.logging import silence_external_info_logs

silence_external_info_logs()


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
    parser = argparse.ArgumentParser(
        description="Inspect frame-synchronous token/latent alignment for one materialized sample."
    )
    parser.add_argument("--config-path", default="configs/training.yaml")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--country", default=None, help="Override dataset.country from config.")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the selected split.")
    parser.add_argument("--key", default=None, help="Sample key to inspect. Overrides --index when set.")
    parser.add_argument("--delay-ms", type=int, default=None, help="Fixed delay in ms. Defaults to dataset.delay_max_ms.")
    parser.add_argument("--max-steps", type=int, default=None, help="Only print the first N aligned steps.")
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=100,
        help="Number of samples from the selected split to aggregate target percentages over.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--one-shard",
        dest="one_shard",
        action="store_true",
        default=True,
        help="Download/materialize only the parquet shard containing the selected manifest row. This is the default.",
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


def _sample_path(materialized_root: Path, row: PairedManifestRow) -> Path:
    return materialized_root / row.country / row.split / f"{row.key}.pt"


def _download_one_parquet_shard(
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


def build_inspection_dataset(
    *,
    cfg: dict[str, Any],
    split: str,
    sample_index: int,
    sample_key: str | None,
    one_shard: bool,
) -> tuple[InspectLatentDataset, int]:
    dataset_cfg = cfg["dataset"]
    country = str(dataset_cfg["country"])
    manifest_root = resolve_manifest_root(dataset_cfg)
    rows = load_split_manifest_rows(
        manifest_root=manifest_root,
        country=country,
        split=split,
    )
    if not rows:
        raise RuntimeError(f"No manifest rows found for country={country!r} split={split!r}.")

    selected_index = next((idx for idx, row in enumerate(rows) if row.key == sample_key), None) if sample_key else sample_index
    if selected_index is None:
        raise KeyError(f"Sample key {sample_key!r} not found in split={split!r}.")
    if selected_index < 0 or selected_index >= len(rows):
        raise IndexError(f"Sample index {selected_index} out of range for split={split!r} with {len(rows)} rows.")

    selected_row = rows[selected_index]
    materialized_root = Path(
        dataset_cfg.get("materialized_latents_dir", "out/materialized_latents")
    ).expanduser().resolve()

    if one_shard:
        local_root = dataset_cfg.get("local_dataset_root")
        dataset_root = None if _is_empty_path(local_root) else Path(str(local_root)).expanduser().resolve()
        parquet_cache = materialized_root / "_inspect_one_shard_cache"
        shard_path = _download_one_parquet_shard(
            dataset_cfg=dataset_cfg,
            dataset_root=dataset_root,
            latent_shard_path=selected_row.latent_shard_path,
            cache_root=parquet_cache,
        )
        shard_rows = [row for row in rows if row.latent_shard_path == selected_row.latent_shard_path]
        needs_materialization = any(not _sample_path(materialized_root, row).exists() for row in shard_rows)
        if needs_materialization or bool(dataset_cfg.get("force_rematerialize", False)):
            written, skipped = _materialize_shard_rows(
                shard_path=shard_path,
                latent_shard_path=selected_row.latent_shard_path,
                materialized_root=materialized_root,
                force_rematerialize=bool(dataset_cfg.get("force_rematerialize", False)),
                materialize_speaker_prefix=bool(dataset_cfg.get("materialize_speaker_prefix", True)),
                tensor_dtype=_resolve_materialized_dtype(
                    dataset_cfg.get("materialized_dtype", cfg["runtime"].get("data_dtype", "bf16"))
                ),
                materialization_batch_size=int(dataset_cfg.get("materialization_batch_size", 128)),
            )
            print(
                f"[INSPECT] one-shard materialized {selected_row.latent_shard_path}: "
                f"written={written} skipped={skipped}"
            )
        else:
            print(f"[INSPECT] using existing materialized samples for {selected_row.latent_shard_path}")

        dataset = InspectLatentDataset(samples=shard_rows, materialized_root=materialized_root)
        remapped_index = next(idx for idx, row in enumerate(shard_rows) if row.key == selected_row.key)
        return dataset, remapped_index

    materialized_root = ensure_materialized_dataset(cfg)
    dataset = InspectLatentDataset(samples=rows, materialized_root=materialized_root)
    return dataset, selected_index


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
        return f"[{kind}]" if kind not in {"P", "W"} else f"[{kind}]"
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    decoded = decoded.replace("\n", "\\n").replace("|", "\\|")
    return f"{token} / {decoded!r}"


def _escape_md(value: str) -> str:
    return value.replace("\n", "\\n").replace("|", "\\|")


def _render_tokenizer_sequence(text: str, tokenizer) -> tuple[str, str, str, str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in token_ids]
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    return repr(text), " ".join(str(int(token_id)) for token_id in token_ids), _escape_md(" ".join(tokens)), repr(decoded)


def render_tokenizer_spacing_diagnostics(
    *,
    sample: dict[str, Any],
    token_ids: list[int],
    tokenizer,
    special_tokens: SpecialTokenIds,
    max_words: int = 8,
) -> str:
    words = []
    for item in sample.get("timestamps") or []:
        text = " ".join(str(item.get("text", "")).strip().split())
        if text:
            words.append(text)
        if len(words) >= max_words:
            break
    if not words:
        words = str(sample["transcription"]).split()[:max_words]

    lines = [
        "## Tokenizer Spacing Diagnostics",
        "",
        "This shows whether the tokenizer distinguishes a word from the same word with a leading space.",
        "",
        "| text | token_ids | tokens | decoded |",
        "|---|---|---|---|",
    ]
    for word in words:
        for text in (word, f" {word}"):
            rendered_text, ids, tokens, decoded = _render_tokenizer_sequence(text, tokenizer)
            lines.append(f"| `{_escape_md(rendered_text)}` | `{ids}` | `{tokens}` | `{_escape_md(decoded)}` |")

    text_token_ids = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == special_tokens.eos:
            break
        if token_id in {
            int(special_tokens.bos),
            int(special_tokens.pad_wait),
            int(special_tokens.word_start),
        }:
            continue
        text_token_ids.append(token_id)

    lines.extend(
        [
            "",
            "Reference-style decode of inspected target stream after skipping `[BOS]`, `[P]`, `[W]`, and stopping at `[EOS]`:",
            "",
            "```text",
            tokenizer.decode(text_token_ids, skip_special_tokens=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def audio_region(step: int, *, left_pad_steps: int, real_steps: int) -> tuple[str, str]:
    latent_idx = step - left_pad_steps
    if step < left_pad_steps:
        return "left_pad", "-"
    if 0 <= latent_idx < real_steps:
        return "real", str(latent_idx)
    return "right_pad", "-"


def summarize_targets_after_left_padding(
    labels: list[int],
    *,
    left_pad_steps: int,
    special_tokens: SpecialTokenIds,
) -> dict[str, float]:
    counted = labels[left_pad_steps:]
    total = len(counted)
    if total == 0:
        return {
            "counted_steps": 0.0,
            "text_or_w_count": 0.0,
            "text_or_w_pct": float("nan"),
            "pad_count": 0.0,
            "pad_pct": float("nan"),
        }

    text_or_w = 0
    pad = 0
    for token_id in counted:
        kind = token_kind(int(token_id), special_tokens)
        if kind in {"TEXT", "W"}:
            text_or_w += 1
        elif kind == "P":
            pad += 1

    return {
        "counted_steps": float(total),
        "text_or_w_count": float(text_or_w),
        "text_or_w_pct": 100.0 * float(text_or_w) / float(total),
        "pad_count": float(pad),
        "pad_pct": 100.0 * float(pad) / float(total),
    }


def align_sample(
    *,
    cfg: dict[str, Any],
    sample: dict[str, Any],
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
):
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    if delay_ms % step_ms != 0:
        raise ValueError(f"--delay-ms must be a multiple of step_ms={step_ms}.")
    return build_delayed_target_stream(
        key=str(sample["key"]),
        latents=sample["projected"],
        transcript=str(sample["transcription"]),
        timestamps=sample.get("timestamps"),
        tokenizer=tokenizer,
        bos_token_id=special_tokens.bos,
        eos_token_id=special_tokens.eos,
        pad_wait_token_id=special_tokens.pad_wait,
        word_start_token_id=special_tokens.word_start,
        delay_steps=delay_ms // step_ms,
        left_pad_steps=int(cfg["dataset"].get("left_pad_steps", 0)),
        step_ms=step_ms,
    )


def aggregate_target_summary(
    *,
    cfg: dict[str, Any],
    dataset: InspectLatentDataset,
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
    num_samples: int,
) -> dict[str, float]:
    left_pad_steps = int(cfg["dataset"].get("left_pad_steps", 0))
    limit = min(max(0, int(num_samples)), len(dataset))
    counted_steps = 0
    text_or_w_count = 0
    pad_count = 0

    for sample_idx in range(limit):
        aligned = align_sample(
            cfg=cfg,
            sample=dataset[sample_idx],
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            delay_ms=delay_ms,
        )
        for token_id in aligned.token_ids.tolist()[left_pad_steps:]:
            counted_steps += 1
            kind = token_kind(int(token_id), special_tokens)
            if kind in {"TEXT", "W"}:
                text_or_w_count += 1
            elif kind == "P":
                pad_count += 1

    return {
        "num_samples": float(limit),
        "counted_steps": float(counted_steps),
        "text_or_w_count": float(text_or_w_count),
        "text_or_w_pct": float("nan") if counted_steps == 0 else 100.0 * text_or_w_count / counted_steps,
        "pad_count": float(pad_count),
        "pad_pct": float("nan") if counted_steps == 0 else 100.0 * pad_count / counted_steps,
    }


def render_aggregate_summary(summary: dict[str, float]) -> str:
    return "\n".join(
        [
            "## Aggregate Target Summary",
            "",
            "Counts exclude the left-padding region.",
            "",
            f"- summary_samples: `{int(summary['num_samples'])}`",
            f"- counted_steps: `{int(summary['counted_steps'])}`",
            f"- text_or_w_count: `{int(summary['text_or_w_count'])}`",
            f"- text_or_w_pct: `{summary['text_or_w_pct']:.2f}%`",
            f"- pad_count: `{int(summary['pad_count'])}`",
            f"- pad_pct: `{summary['pad_pct']:.2f}%`",
            "",
        ]
    )


def build_report(
    *,
    cfg: dict[str, Any],
    split: str,
    sample_index: int,
    sample: dict[str, Any],
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
    max_steps: int | None,
) -> str:
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    if delay_ms % step_ms != 0:
        raise ValueError(f"--delay-ms must be a multiple of step_ms={step_ms}.")
    delay_steps = delay_ms // step_ms
    left_pad_steps = int(cfg["dataset"].get("left_pad_steps", 0))
    latents = sample["projected"]
    real_steps = int(latents.shape[0])

    aligned = align_sample(
        cfg=cfg,
        sample=sample,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
    )

    token_ids = aligned.token_ids.tolist()
    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    num_stream_steps = len(token_ids)
    num_pairs = len(labels)
    shown_steps = num_pairs if max_steps is None else min(num_pairs, int(max_steps))
    target_summary = summarize_targets_after_left_padding(
        token_ids,
        left_pad_steps=left_pad_steps,
        special_tokens=special_tokens,
    )

    lines = [
        "# Alignment Inspection",
        "",
        f"- key: `{sample['key']}`",
        f"- split: `{split}`",
        f"- sample_index: `{sample_index}`",
        f"- real_steps: `{real_steps}`",
        f"- left_pad_steps: `{left_pad_steps}`",
        f"- delay_ms: `{delay_ms}`",
        f"- delay_steps: `{delay_steps}`",
        f"- aligned_stream_steps: `{num_stream_steps}`",
        f"- shifted_training_pairs: `{num_pairs}`",
        f"- shown_pairs: `{shown_steps}`",
        "",
        "## Target Summary",
        "",
        "Counts exclude the left-padding region.",
        "",
        f"- counted_steps: `{int(target_summary['counted_steps'])}`",
        f"- text_or_w_count: `{int(target_summary['text_or_w_count'])}`",
        f"- text_or_w_pct: `{target_summary['text_or_w_pct']:.2f}%`",
        f"- pad_count: `{int(target_summary['pad_count'])}`",
        f"- pad_pct: `{target_summary['pad_pct']:.2f}%`",
        "",
        "## Transcript",
        "",
        str(sample["transcription"]),
        "",
        "## Word Timestamps",
        "",
    ]

    timestamps = sample.get("timestamps") or []
    if timestamps:
        lines.extend(
            f"- `{item.get('start', '?')}`-`{item.get('end', '?')}`: {item.get('text', '')}"
            for item in timestamps
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            render_tokenizer_spacing_diagnostics(
                sample=sample,
                token_ids=token_ids,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Shifted Training Pairs",
            "",
            "| step | ms | audio | latent_idx | input_token | target_step | target_token |",
            "|---:|---:|---|---:|---|---:|---|",
        ]
    )

    for step in range(shown_steps):
        region, latent_idx = audio_region(step, left_pad_steps=left_pad_steps, real_steps=real_steps)
        input_id = int(input_ids[step])
        target_id = int(labels[step])
        lines.append(
            "| "
            f"{step} | "
            f"{step * step_ms} | "
            f"{region} | "
            f"{latent_idx} | "
            f"{render_token(input_id, tokenizer, special_tokens)} | "
            f"{step + 1} | "
            f"{render_token(target_id, tokenizer, special_tokens)} |"
        )

    if shown_steps < num_pairs:
        lines.extend(["", f"_Truncated: {num_pairs - shown_steps} additional shifted pairs not shown._"])

    if len(aligned.token_ids) != len(aligned.audio_features):
        raise RuntimeError("Alignment invariant failed: token_ids and audio_features lengths differ.")
    if len(input_ids) != len(labels) or len(labels) != len(aligned.audio_features) - 1:
        raise RuntimeError("Shifted alignment invariant failed.")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_path)
    if args.country is not None:
        cfg["dataset"]["country"] = args.country

    resolved_tokenizer = load_tokenizer(cfg["tokenizer"])
    tokenizer = resolved_tokenizer.tokenizer
    special_tokens = SpecialTokenIds(
        bos=resolved_tokenizer.bos_token_id,
        eos=resolved_tokenizer.eos_token_id,
        pad_wait=resolved_tokenizer.pad_wait_token_id,
        word_start=resolved_tokenizer.word_start_token_id,
    )

    dataset, sample_index = build_inspection_dataset(
        cfg=cfg,
        split=args.split,
        sample_index=int(args.index),
        sample_key=args.key,
        one_shard=bool(args.one_shard),
    )

    sample = dataset[sample_index]
    delay_ms = int(args.delay_ms if args.delay_ms is not None else cfg["dataset"].get("delay_max_ms", 2400))
    aggregate_summary = aggregate_target_summary(
        cfg=cfg,
        dataset=dataset,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
        num_samples=args.summary_samples,
    )

    report = build_report(
        cfg=cfg,
        split=args.split,
        sample_index=sample_index,
        sample=sample,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
        max_steps=args.max_steps,
    )
    report = report + "\n" + render_aggregate_summary(aggregate_summary)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[INSPECT] wrote {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
