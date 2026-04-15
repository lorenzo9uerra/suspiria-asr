from __future__ import annotations

import json, os, re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import Audio, load_dataset
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import HfApi, HfFileSystem, create_repo
from huggingface_hub.errors import EntryNotFoundError
from tqdm import tqdm

from qwen_asr import Qwen3ASRModel


# ---------------------------------------------------------------------------
# IterableDataset wrapper
# ---------------------------------------------------------------------------

class AudioIterableDataset(IterableDataset):
    """
    Wraps a HF streaming split into a torch IterableDataset.
    Each item is a dict: {"key": str, "array": np.ndarray, "sampling_rate": int}
    """

    def __init__(
        self,
        hf_split,
        audio_col: str,
        max_steps: Optional[int] = None,
        batch_size: int = 1,
        skip_keys: Optional[set[str]] = None,
        skip_prefix_samples: int = 0,
    ):
        self.hf_split = hf_split
        self.audio_col = audio_col
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.skip_keys = skip_keys or set()
        self.skip_prefix_samples = int(skip_prefix_samples)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for step, ex in enumerate(self.hf_split):
            if self.max_steps is not None and step >= self.max_steps * self.batch_size:
                break
            if step < self.skip_prefix_samples:
                continue
            k = ex.get("key")
            if k is None:
                continue
            if str(k) in self.skip_keys:
                continue
            try:
                a = ex[self.audio_col]
                wav = a["array"]
                sr = int(a["sampling_rate"])
            except Exception as e:
                print(f"[WARN] Skipping sample key={k}: failed to decode audio ({e})")
                continue
            if wav is None:
                print(f"[WARN] Skipping sample key={k}: decoded audio is empty")
                continue
            yield {"key": str(k), "array": wav, "sampling_rate": sr}


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of samples into a batch dict (keeps numpy arrays as-is)."""
    return {
        "keys": [s["key"] for s in samples],
        "audios": [(s["array"], s["sampling_rate"]) for s in samples],
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_asr(cfg: DictConfig) -> Qwen3ASRModel:
    dtype_str = str(cfg.vllm.forced_aligner_dtype).lower()
    if dtype_str in ("float16", "fp16", "torch.float16"):
        fa_dtype = torch.float16
    elif dtype_str in ("bfloat16", "bf16", "torch.bfloat16"):
        fa_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported forced_aligner_dtype: {cfg.vllm.forced_aligner_dtype}")

    return Qwen3ASRModel.LLM(
        model=str(cfg.asr.model),
        gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        max_model_len=int(cfg.vllm.max_model_len),
        max_inference_batch_size=int(cfg.vllm.max_inference_batch_size),
        max_new_tokens=int(cfg.vllm.max_new_tokens),
        enforce_eager=bool(cfg.vllm.enforce_eager),
        forced_aligner=str(cfg.asr.forced_aligner),
        forced_aligner_kwargs=dict(
            dtype=fa_dtype,
            device_map=str(cfg.vllm.forced_aligner_device_map),
        ),
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_data_files(cfg: DictConfig) -> Dict[str, str]:
    country = str(cfg.dataset.country)
    files: Dict[str, str] = {}
    for split in cfg.dataset.splits:
        pattern = cfg.dataset.patterns[split]
        files[str(split)] = f"{country}/{pattern}"
    return files


def filter_existing_splits(repo_id: str, data_files: Dict[str, str]) -> Dict[str, str]:
    fs = HfFileSystem()
    kept: Dict[str, str] = {}
    for split, pattern in data_files.items():
        matches = fs.glob(f"datasets/{repo_id}/{pattern}")
        if len(matches) > 0:
            kept[split] = pattern
        else:
            print(f"[WARN] Split '{split}' missing (no files match '{pattern}'). Skipping.")
    return kept


def list_split_shards(repo_id: str, pattern: str) -> list[str]:
    fs = HfFileSystem()
    matches = sorted(fs.glob(f"datasets/{repo_id}/{pattern}"))
    prefix = f"datasets/{repo_id}/"
    return [m[len(prefix):] if m.startswith(prefix) else m for m in matches]


# ---------------------------------------------------------------------------
# HF auth helpers
# ---------------------------------------------------------------------------

def ensure_hf_logged_in_if_upload_enabled(cfg: DictConfig) -> None:
    if "upload" not in cfg or not bool(cfg.upload.enabled):
        return

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()
    except Exception as e:
        raise RuntimeError(
            "upload.enabled=true but you are not logged in to Hugging Face.\n"
            "Run: huggingface-cli login\n"
            "Then rerun this script."
        ) from e

    print(f"[HF] Logged in as: {user.get('name') or user.get('fullname') or user.get('email')}")


def upload_file_if_enabled(cfg: DictConfig, *, path_in_repo: str, local_path: str) -> None:
    if "upload" not in cfg or not bool(cfg.upload.enabled):
        return

    repo_id = str(cfg.upload.repo_id)
    private = bool(cfg.upload.private)

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] Uploaded {local_path} -> {repo_id}/{path_in_repo}")


def list_remote_transcription_shards(
    cfg: DictConfig,
    *,
    country: str,
    split: str,
    file_prefix: str,
) -> list[str]:
    if "upload" not in cfg or not bool(cfg.upload.enabled):
        return []
    if not bool(cfg.upload.get("skip_existing", False)):
        return []

    repo_id = str(cfg.upload.repo_id)
    fs = HfFileSystem()
    remote_pattern = f"datasets/{repo_id}/{country}/{split}/{file_prefix}-part-*.jsonl"
    try:
        return sorted(fs.glob(remote_pattern))
    except FileNotFoundError:
        return []


def get_remote_existing_keys_and_next_part_index(
    cfg: DictConfig,
    *,
    country: str,
    split: str,
    file_prefix: str,
) -> tuple[set[str], int]:
    shard_paths = list_remote_transcription_shards(
        cfg,
        country=country,
        split=split,
        file_prefix=file_prefix,
    )
    if not shard_paths:
        return set(), 0

    fs = HfFileSystem()
    existing_keys: set[str] = set()
    max_part_idx = -1
    pattern = re.compile(rf"{re.escape(file_prefix)}-part-(\d+)\.jsonl$")

    for shard_path in shard_paths:
        name = Path(shard_path).name
        match = pattern.match(name)
        if match is not None:
            max_part_idx = max(max_part_idx, int(match.group(1)))
        with fs.open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get("key")
                if key is not None:
                    existing_keys.add(str(key))
    print(
        f"[HF] Resuming from remote shards for {country}/{split}: "
        f"{len(shard_paths)} files, {len(existing_keys)} existing keys"
    )
    return existing_keys, max_part_idx + 1


def load_remote_progress(
    cfg: DictConfig,
    *,
    country: str,
    split: str,
) -> Optional[dict[str, Any]]:
    if "upload" not in cfg or not bool(cfg.upload.enabled):
        return None
    if not bool(cfg.upload.get("skip_existing", False)):
        return None

    repo_id = str(cfg.upload.repo_id)
    fs = HfFileSystem()
    remote_path = f"datasets/{repo_id}/{country}/{split}/_progress.json"
    try:
        with fs.open(remote_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, EntryNotFoundError):
        return None


def write_progress_file(
    path: str,
    *,
    country: str,
    split: str,
    last_completed_source_shard: str,
    next_part_idx: int,
) -> None:
    payload = {
        "country": country,
        "split": split,
        "last_completed_source_shard": last_completed_source_shard,
        "next_part_idx": next_part_idx,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def flush_pending_rows(
    cfg: DictConfig,
    *,
    pending_rows: list[dict[str, Any]],
    split_out_dir: str,
    country: str,
    split_name: str,
    file_prefix: str,
    next_part_idx: int,
) -> tuple[list[dict[str, Any]], int]:
    if not pending_rows:
        return pending_rows, next_part_idx

    shard_name = f"{file_prefix}-part-{next_part_idx:06d}.jsonl"
    shard_path = os.path.join(split_out_dir, shard_name)
    write_jsonl_chunk(shard_path, pending_rows)
    upload_file_if_enabled(
        cfg,
        path_in_repo=f"{country}/{split_name}/{shard_name}",
        local_path=shard_path,
    )
    return [], next_part_idx + 1


def write_jsonl_chunk(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="transcription")
def main(cfg: DictConfig) -> None:
    print("=== Effective config ===")
    print(OmegaConf.to_yaml(cfg))

    ensure_hf_logged_in_if_upload_enabled(cfg)

    data_files = make_data_files(cfg)
    data_files = filter_existing_splits(str(cfg.dataset.repo_id), data_files)

    if not data_files:
        raise RuntimeError("No split files found for the given country/patterns.")

    audio_col = str(cfg.audio.audio_column)
    asr = build_asr(cfg)

    batch_size = int(cfg.batching.dataset_batch_size)
    num_workers = int(cfg.batching.get("num_workers", 1))
    prefetch_factor = int(cfg.batching.get("prefetch_factor", 4))
    max_steps_per_split = cfg.batching.max_steps_per_split
    language = cfg.asr.language if cfg.asr.language not in ("null", None) else None
    return_ts = bool(cfg.asr.return_timestamps)

    out_dir = str(cfg.output.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    country = str(cfg.dataset.country)
    upload_every_batches = cfg.upload.get("upload_every_batches") if "upload" in cfg else None
    file_prefix = str(cfg.output.get("file_prefix", "transcriptions"))

    for split_name in data_files.keys():
        split_name = str(split_name)
        split_out_dir = os.path.join(out_dir, country, split_name)
        os.makedirs(split_out_dir, exist_ok=True)
        print(f"\n=== Processing split={split_name} -> {split_out_dir} ===")
        split_shards = list_split_shards(str(cfg.dataset.repo_id), data_files[split_name])
        if not split_shards:
            print(f"[WARN] No source parquet shards found for split={split_name}. Skipping.")
            continue
        existing_keys, next_part_idx = get_remote_existing_keys_and_next_part_index(
            cfg,
            country=country,
            split=split_name,
            file_prefix=file_prefix,
        )
        progress = load_remote_progress(cfg, country=country, split=split_name)
        start_shard_idx = 0
        if progress is not None:
            last_completed = progress.get("last_completed_source_shard")
            if isinstance(last_completed, str) and last_completed in split_shards:
                start_shard_idx = split_shards.index(last_completed) + 1
                next_part_idx = max(int(progress.get("next_part_idx", next_part_idx)), next_part_idx)
                print(
                    f"[HF] Resuming split={split_name} from source shard "
                    f"{start_shard_idx}/{len(split_shards)}"
                )
            elif last_completed:
                print(
                    f"[WARN] Remote progress points to missing shard '{last_completed}'. "
                    "Starting from the first shard."
                )

        pending_rows: list[dict[str, Any]] = []
        pending_batches = 0
        new_records_written = 0
        processed_batches = 0
        progress_path = os.path.join(split_out_dir, "_progress.json")

        with tqdm(unit="sample", desc=split_name, dynamic_ncols=True) as pbar:
            stop_split_early = False
            for source_shard in split_shards[start_shard_idx:]:
                shard_ds = load_dataset(
                    str(cfg.dataset.repo_id),
                    data_files={split_name: [source_shard]},
                    streaming=bool(cfg.dataset.streaming),
                )[split_name]
                if bool(cfg.audio.cast_audio_column):
                    target_sr = int(cfg.audio.target_sampling_rate)
                    shard_ds = shard_ds.cast_column(audio_col, Audio(sampling_rate=target_sr))

                dataset = AudioIterableDataset(
                    hf_split=shard_ds,
                    audio_col=audio_col,
                    max_steps=None,
                    batch_size=batch_size,
                    skip_keys=existing_keys,
                    skip_prefix_samples=0,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    collate_fn=collate_fn,
                    persistent_workers=num_workers > 0,
                )

                for batch in loader:
                    keys: List[str] = batch["keys"]
                    audios: List[Tuple[np.ndarray, int]] = batch["audios"]

                    langs = [language] * len(audios) if language is not None else None

                    results = asr.transcribe(
                        audio=audios,
                        language=langs,
                        return_time_stamps=bool(return_ts),
                    )

                    for k, r in zip(keys, results):
                        text = getattr(r, "text", None)
                        rec = {
                            "key": k,
                            "country": country,
                            "split": split_name,
                            "new_transcription": text,
                        }
                        if return_ts:
                            time_stamps = getattr(r, "time_stamps", None)
                            items = getattr(time_stamps, "items", [])
                            rec["timestamps"] = [
                                {
                                    "text": item.text,
                                    "start": item.start_time,
                                    "end": item.end_time,
                                }
                                for item in items
                            ]

                        pending_rows.append(rec)
                        existing_keys.add(k)
                        new_records_written += 1

                    pbar.update(len(keys))
                    pending_batches += 1
                    processed_batches += 1

                    if (
                        "upload" in cfg
                        and upload_every_batches not in (None, "null")
                        and int(upload_every_batches) > 0
                        and pending_rows
                        and pending_batches >= int(upload_every_batches)
                    ):
                        pending_rows, next_part_idx = flush_pending_rows(
                            cfg,
                            pending_rows=pending_rows,
                            split_out_dir=split_out_dir,
                            country=country,
                            split_name=split_name,
                            file_prefix=file_prefix,
                            next_part_idx=next_part_idx,
                        )
                        pending_batches = 0

                    if max_steps_per_split is not None and processed_batches >= int(max_steps_per_split):
                        print(f"Stopping early for {split_name} due to max_steps_per_split={max_steps_per_split}")
                        stop_split_early = True
                        break

                pending_rows, next_part_idx = flush_pending_rows(
                    cfg,
                    pending_rows=pending_rows,
                    split_out_dir=split_out_dir,
                    country=country,
                    split_name=split_name,
                    file_prefix=file_prefix,
                    next_part_idx=next_part_idx,
                )
                pending_batches = 0

                if not stop_split_early:
                    write_progress_file(
                        progress_path,
                        country=country,
                        split=split_name,
                        last_completed_source_shard=source_shard,
                        next_part_idx=next_part_idx,
                    )
                    upload_file_if_enabled(
                        cfg,
                        path_in_repo=f"{country}/{split_name}/_progress.json",
                        local_path=progress_path,
                    )

                del loader, dataset, shard_ds

                if stop_split_early:
                    break

        print(f"Done split={split_name}.")
        if new_records_written > 0:
            print(f"[HF] Wrote {new_records_written} new transcriptions for split={split_name}.")
        elif existing_keys:
            print(f"[HF] No new keys to upload for split={split_name}.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
