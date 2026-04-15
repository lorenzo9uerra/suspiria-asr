#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paired latents+transcriptions dataset locally and optionally upload it "
            "to a Hugging Face dataset repo."
        )
    )
    parser.add_argument("--latents-dir", default=None, help="Local latents root directory")
    parser.add_argument("--latents-repo-id", default=None, help="HF dataset repo containing latents")
    parser.add_argument("--latents-revision", default=None, help="Optional latents repo revision")

    parser.add_argument(
        "--transcriptions-dir",
        default=None,
        help="Local directory containing transcribe.py JSONL outputs",
    )
    parser.add_argument(
        "--transcriptions-repo-id",
        default=None,
        help="HF dataset repo containing transcribe.py outputs",
    )
    parser.add_argument(
        "--transcriptions-revision",
        default=None,
        help="Optional transcriptions repo revision",
    )

    parser.add_argument("--output-dir", required=True, help="Local paired dataset staging directory")
    parser.add_argument("--repo-id", default=None, help="Optional target HF dataset repo for upload")
    parser.add_argument("--revision", default=None, help="Optional upload revision")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip pairing/staging and upload an existing output-dir as-is",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target dataset repo as private if it does not exist",
    )
    parser.add_argument("--country", default=None, help="Optional country filter")
    parser.add_argument("--split", default=None, help="Optional split filter")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 4) // 2)),
        help="Parallel workers for large-folder upload",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=60,
        help="Seconds between upload progress reports",
    )
    parser.add_argument(
        "--hf-xet-high-performance",
        action="store_true",
        help="Set HF_XET_HIGH_PERFORMANCE=1 for faster large-folder uploads",
    )
    parser.add_argument(
        "--hf-xet-cache-dir",
        default=None,
        help="Optional local cache directory for hf_xet during upload",
    )
    return parser.parse_args()


def snapshot_to_local_dir(
    *,
    local_dir: str | None,
    repo_id: str | None,
    revision: str | None,
    allow_patterns: list[str],
) -> Path:
    if local_dir is not None:
        path = Path(local_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    if repo_id is None:
        raise ValueError("Either a local directory or a repo id must be provided.")

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
        )
    ).resolve()


def infer_country_split_from_path(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    if len(parts) >= 2 and path.suffix in {".pt", ".parquet"}:
        return parts[-3] if len(parts) >= 3 else None, parts[-2] if len(parts) >= 2 else None

    if len(parts) >= 3 and path.suffix == ".jsonl":
        return parts[-3], parts[-2]

    stem = path.stem
    if "_" in stem:
        country, split = stem.rsplit("_", 1)
        return country, split

    if len(parts) >= 2 and path.suffix == ".jsonl":
        return parts[-2], path.stem

    return None, None


def load_transcriptions(root: Path, *, country_filter: str | None, split_filter: str | None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in {"_progress.jsonl", "_progress.json"} or path.stem == "_progress":
            continue
        country, split = infer_country_split_from_path(path.relative_to(root))
        if country_filter is not None and country not in (None, country_filter):
            continue
        if split_filter is not None and split not in (None, split_filter):
            continue

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = str(row["key"])
                text = row.get("new_transcription", row.get("transcription", row.get("text")))
                timestamps = row.get("timestamps")
                records[key] = {
                    "key": key,
                    "transcription": text,
                    "timestamps": timestamps,
                    "country": row.get("country", country),
                    "split": row.get("split", split),
                    "source_jsonl": str(path.relative_to(root)),
                    "line_no": line_no,
                }
    return records


def load_latent_entries(
    root: Path,
    *,
    country_filter: str | None,
    split_filter: str | None,
) -> list[dict[str, Any]]:
    manifest_entries: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "key" not in row:
                    continue
                rel_value = row.get("latent_shard_path", row.get("latent_path"))
                row_idx = row.get("latent_row_idx")
                if rel_value is None:
                    continue
                country = row.get("country")
                split = row.get("split")
                if country_filter is not None and country not in (None, country_filter):
                    continue
                if split_filter is not None and split not in (None, split_filter):
                    continue

                rel_path = Path(rel_value)
                abs_path = root / rel_path
                if not abs_path.exists():
                    abs_path = path.parent / rel_path
                if not abs_path.exists():
                    raise FileNotFoundError(f"Latent shard not found for manifest row: {rel_path}")

                manifest_entries.append(
                    {
                        "key": str(row["key"]),
                        "country": country,
                        "split": split,
                        "src_path": abs_path.resolve(),
                        "relative_path": rel_path,
                        "latent_row_idx": None if row_idx is None else int(row_idx),
                        "num_frames": row.get("num_frames"),
                        "speaker_prefix_frames": row.get("speaker_prefix_frames"),
                    }
                )

    if manifest_entries:
        return manifest_entries

    scanned_entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.parquet")):
        rel_path = path.relative_to(root)
        country, split = infer_country_split_from_path(rel_path)
        if country_filter is not None and country not in (None, country_filter):
            continue
        if split_filter is not None and split not in (None, split_filter):
            continue
        scanned_entries.append(
            {
                "key": path.stem,
                "country": country,
                "split": split,
                "src_path": path.resolve(),
                "relative_path": rel_path,
                "latent_row_idx": None,
            }
        )
    return scanned_entries


def stage_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except OSError as e:
        raise OSError(
            f"Failed to hardlink {src} -> {dst}. "
            "Hardlink-only staging is enabled, so source and output directories must be on the same filesystem."
        ) from e


def maybe_login_for_upload(api: HfApi) -> None:
    try:
        user = api.whoami()
    except Exception as e:
        raise RuntimeError(
            "You are not logged in to Hugging Face.\n"
            "Run: huggingface-cli login\n"
            "Then rerun this script."
        ) from e

    print(f"[HF] Logged in as: {user.get('name') or user.get('fullname') or user.get('email')}")


def configure_upload_env(args: argparse.Namespace) -> None:
    if args.hf_xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        print("[HF] Enabled HF_XET_HIGH_PERFORMANCE=1")
    if args.hf_xet_cache_dir is not None:
        cache_dir = Path(args.hf_xet_cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_XET_CACHE"] = str(cache_dir)
        print(f"[HF] Using HF_XET_CACHE={cache_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not args.skip_build:
        latents_root = snapshot_to_local_dir(
            local_dir=args.latents_dir,
            repo_id=args.latents_repo_id,
            revision=args.latents_revision,
            allow_patterns=["*.parquet", "*.jsonl"],
        )
        transcriptions_root = snapshot_to_local_dir(
            local_dir=args.transcriptions_dir,
            repo_id=args.transcriptions_repo_id,
            revision=args.transcriptions_revision,
            allow_patterns=["*.jsonl"],
        )

        transcriptions = load_transcriptions(
            transcriptions_root,
            country_filter=args.country,
            split_filter=args.split,
        )
        latent_entries = load_latent_entries(
            latents_root,
            country_filter=args.country,
            split_filter=args.split,
        )

        manifests_out_dir = output_dir / "manifests"
        output_dir.mkdir(parents=True, exist_ok=True)

        matched = 0
        missing_transcriptions = 0
        manifest_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for entry in latent_entries:
            record = transcriptions.get(entry["key"])
            if record is None:
                missing_transcriptions += 1
                continue

            country = entry.get("country") or record.get("country") or "unknown_country"
            split = entry.get("split") or record.get("split") or "unknown_split"
            dst_rel = Path("latents") / entry["relative_path"]
            dst_abs = output_dir / dst_rel
            stage_file(entry["src_path"], dst_abs)

            payload = {
                "key": entry["key"],
                "country": country,
                "split": split,
                "latent_shard_path": str(dst_rel),
                "latent_row_idx": entry.get("latent_row_idx"),
                "transcription": record["transcription"],
            }
            if entry.get("num_frames") is not None:
                payload["num_frames"] = entry["num_frames"]
            if entry.get("speaker_prefix_frames") is not None:
                payload["speaker_prefix_frames"] = entry["speaker_prefix_frames"]
            if record.get("timestamps") is not None:
                payload["timestamps"] = record["timestamps"]
            manifest_rows[(country, split)].append(payload)
            matched += 1

        written_manifests: list[Path] = []
        for (country, split), rows in sorted(manifest_rows.items()):
            manifest_path = manifests_out_dir / country / split / "paired_manifest.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as out_f:
                for row in rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary_path = manifest_path.parent / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "country": country,
                        "split": split,
                        "num_paired": len(rows),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            written_manifests.append(manifest_path)

        summary = {
            "latents_root": str(latents_root),
            "transcriptions_root": str(transcriptions_root),
            "num_transcriptions": len(transcriptions),
            "num_latents": len(latent_entries),
            "num_paired": matched,
            "num_missing_transcriptions": missing_transcriptions,
            "manifests": [str(path.relative_to(output_dir)) for path in written_manifests],
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(
            "[PAIR] "
            f"latents={len(latent_entries)} transcriptions={len(transcriptions)} "
            f"paired={matched} missing_transcriptions={missing_transcriptions}"
        )
        for manifest_path in written_manifests:
            print(f"[PAIR] Wrote {manifest_path}")
    else:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"--skip-build was set but output-dir does not exist: {output_dir}"
            )
        print(f"[PAIR] Skipping build, uploading existing staged folder: {output_dir}")

    if args.repo_id is None:
        return

    configure_upload_env(args)
    api = HfApi()
    maybe_login_for_upload(api)

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=str(output_dir),
        repo_type="dataset",
        revision=args.revision,
        private=args.private,
        allow_patterns=["*.parquet", "*.jsonl", "*.json"],
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=args.report_every,
    )
    print(f"[HF] Uploaded {output_dir} -> {args.repo_id}")


if __name__ == "__main__":
    main()
