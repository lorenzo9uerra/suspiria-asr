#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import queue
import threading
from functools import partial
from typing import Any

import safetensors
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import yaml
from datasets import Audio, load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm import tqdm

from models.mimi import MimiEncoder
from modules.dummy_quantizer import DummyQuantizer
from modules.mimi_transformer import ProjectedTransformer
from modules.seanet import SEANetEncoder


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Mimi latents from a Hugging Face dataset in batches."
    )
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).resolve().parent / "configs" / "encoding.yaml"),
        help="Path to the encoding YAML config",
    )

    return parser.parse_args()


def coerce_audio(audio_value: Any) -> tuple[torch.Tensor, int]:
    """
    Returns waveform as [C, T] float32 and sample rate.
    """
    if isinstance(audio_value, dict):
        wav = torch.as_tensor(audio_value["array"]).float()
        sr = int(audio_value["sampling_rate"])
    elif hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        wav = torch.as_tensor(samples.data).float()
        sr = int(samples.sample_rate)
    elif hasattr(audio_value, "data") and hasattr(audio_value, "sample_rate"):
        wav = torch.as_tensor(audio_value.data).float()
        sr = int(audio_value.sample_rate)
    else:
        raise TypeError(f"Unsupported audio type: {type(audio_value)}")

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2:
        # Heuristic for [T, C] -> [C, T]
        if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
            wav = wav.transpose(0, 1).contiguous()
    else:
        raise ValueError(f"Unsupported waveform shape: {tuple(wav.shape)}")

    if wav.numel() == 0:
        raise ValueError("Empty audio")

    return wav, sr


def ensure_channels(wav: torch.Tensor, channels: int) -> torch.Tensor:
    """
    Convert [C, T] to desired channel count.
    """
    if wav.shape[0] == channels:
        return wav
    if channels == 1:
        return wav.mean(dim=0, keepdim=True)
    if wav.shape[0] == 1 and channels > 1:
        return wav.repeat(channels, 1)
    raise ValueError(f"Cannot convert waveform with {wav.shape[0]} channels to {channels} channels")


def load_full_yaml_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(raw)}")
    return raw


def make_data_files(dataset_cfg: dict[str, Any]) -> dict[str, str]:
    country = str(dataset_cfg["country"])
    files: dict[str, str] = {}
    for split in dataset_cfg["splits"]:
        pattern = dataset_cfg["patterns"][split]
        files[str(split)] = f"{country}/{pattern}"
    return files


def filter_existing_splits(repo_id: str, data_files: dict[str, str]) -> dict[str, str]:
    fs = HfFileSystem()
    kept: dict[str, str] = {}
    for split, pattern in data_files.items():
        matches = fs.glob(f"datasets/{repo_id}/{pattern}")
        if matches:
            kept[split] = pattern
        else:
            print(f"[WARN] Split '{split}' missing (no files match '{pattern}'). Skipping.")
    return kept


def list_split_shards(repo_id: str, pattern: str) -> list[str]:
    fs = HfFileSystem()
    matches = sorted(fs.glob(f"datasets/{repo_id}/{pattern}"))
    prefix = f"datasets/{repo_id}/"
    return [m[len(prefix):] if m.startswith(prefix) else m for m in matches]


def resolve_checkpoint_path(
    *,
    candidate_path: str | None,
    repo_id: str | None,
    repo_filename: str | None,
) -> str | None:
    path_or_filename = candidate_path or repo_filename
    if path_or_filename is None:
        return None

    local_path = Path(path_or_filename).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    if repo_id is None:
        raise ValueError(
            f"Checkpoint {path_or_filename!r} was not found locally and no --mimi-repo was provided."
        )

    return hf_hub_download(repo_id=repo_id, filename=path_or_filename)


def get_mimi_state_dict(weights_path: str) -> dict[str, torch.Tensor]:
    if weights_path.endswith(".safetensors"):
        state_dict: dict[str, torch.Tensor] = {}
        with safetensors.safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mimi."):
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    raise ValueError(f"Unsupported checkpoint format in {weights_path}")


def remap_mimi_key(key: str, weights_path: str) -> str:
    weights_name = Path(weights_path).name

    key = key.replace("self_attn.in_projs.0.weight", "self_attn.in_proj.weight")
    key = key.replace("self_attn.out_projs.0.weight", "self_attn.out_proj.weight")
    key = key.replace("mimi.", "")
    return key


def load_mimi_encoder(
    *,
    mimi_config: dict[str, Any],
    device: str,
) -> MimiEncoder:
    encoder = SEANetEncoder(**mimi_config["seanet"])
    encoder_transformer = ProjectedTransformer(**mimi_config["transformer"])
    quantizer = DummyQuantizer(**mimi_config["quantizer"])

    model = MimiEncoder(
        encoder=encoder,
        quantizer=quantizer,
        channels=mimi_config["channels"],
        sample_rate=mimi_config["sample_rate"],
        frame_rate=mimi_config["frame_rate"],
        encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
        encoder_transformer=encoder_transformer,
    ).to(device="cpu")

    resolved_weights_path = resolve_checkpoint_path(
        candidate_path=mimi_config.get("weights_path"),
        repo_id=mimi_config.get("repo_id"),
        repo_filename=mimi_config.get("weights_filename"),
    )
    if resolved_weights_path is not None:
        mimi_state = get_mimi_state_dict(resolved_weights_path)
        mimi_state_remap = {
            remap_mimi_key(k, resolved_weights_path): v for k, v in mimi_state.items()
        }
        missing, unexpected = model.load_state_dict(mimi_state_remap, strict=False)
        if missing:
            print("Missing keys from mimi state dict:", missing)
        #if unexpected:
        #    print("Unexpected keys from mimi state dict:", unexpected)
    else:
        raise ValueError(
            "No Mimi checkpoint configured. Set `mimi.weights_path` in the config or pass "
            "`--mimi-weights-path`."
        )

    model.eval().to(device)
    return model


def maybe_compile_model_calls(model: MimiEncoder, *, encoding_cfg: dict[str, Any]) -> MimiEncoder:
    if not bool(encoding_cfg.get("compile_model_calls", False)):
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("encoding.compile_model_calls=true but this PyTorch build has no torch.compile")

    compile_kwargs = {
        "dynamic": bool(encoding_cfg.get("compile_dynamic", True)),
    }
    mode = encoding_cfg.get("compile_mode")
    if mode not in (None, "null"):
        compile_kwargs["mode"] = str(mode)

    model.encode_to_latent = torch.compile(model.encode_to_latent, **compile_kwargs)
    model.quantize = torch.compile(model.quantize, **compile_kwargs)
    print(
        "[TORCH] Compiled Mimi calls: encode_to_latent, quantize "
        f"(dynamic={compile_kwargs['dynamic']}"
        + (f", mode={compile_kwargs['mode']}" if "mode" in compile_kwargs else "")
        + ")"
    )
    return model


def trim_audio(wav: torch.Tensor, sr: int, max_seconds: float | None) -> torch.Tensor:
    if max_seconds is None:
        return wav
    max_len = int(max_seconds * sr)
    return wav[:, :max_len]


def load_existing_local_keys(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    keys: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key")
            if key is not None:
                keys.add(str(key))
    return keys


def infer_next_shard_idx(latent_dir: Path) -> int:
    max_idx = -1
    for path in latent_dir.glob("latents-*.parquet"):
        try:
            idx = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1


def resolve_manifest_path(*, output_cfg: dict[str, Any], country: str, split: str) -> Path:
    manifest_root = Path(
        output_cfg.get("manifest_dir", Path(output_cfg["latent_dir"]).expanduser() / "manifests")
    ).expanduser().resolve()
    return manifest_root / country / f"{split}.jsonl"


def resolve_progress_path(*, latent_dir: Path) -> Path:
    return latent_dir / "_progress.json"


def load_local_progress(progress_path: Path) -> dict[str, Any] | None:
    if not progress_path.exists():
        return None
    with progress_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_local_progress(
    progress_path: Path,
    *,
    country: str,
    split: str,
    last_completed_source_shard: str,
) -> None:
    payload = {
        "country": country,
        "split": split,
        "last_completed_source_shard": last_completed_source_shard,
    }
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class AsyncLatentSaver:
    def __init__(
        self,
        *,
        latents_root: Path,
        latent_dir: Path,
        out_f,
        samples_per_parquet: int,
        max_pending_jobs: int = 8,
    ) -> None:
        self.latents_root = latents_root
        self.latent_dir = latent_dir
        self.out_f = out_f
        self.samples_per_parquet = samples_per_parquet
        self.saved = 0
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=max_pending_jobs)
        self._sentinel = object()
        self._flush_token = object()
        self._error: BaseException | None = None
        self._current_rows: list[dict[str, Any]] = []
        self._next_shard_idx = infer_next_shard_idx(latent_dir)
        self._thread = threading.Thread(target=self._run, name="latent-saver", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                job = self._queue.get()
                try:
                    if job is self._sentinel:
                        self._flush_current_rows()
                        return
                    if job is self._flush_token:
                        self._flush_current_rows()
                        continue
                    self._save_job(job)
                finally:
                    self._queue.task_done()
        except BaseException as e:
            self._error = e
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def _save_job(self, job: dict[str, Any]) -> None:
        self._current_rows.append(job)
        if len(self._current_rows) >= self.samples_per_parquet:
            self._flush_current_rows()

    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def _flush_current_rows(self) -> None:
        if not self._current_rows:
            return

        shard_name = f"latents-{self._next_shard_idx:06d}.parquet"
        shard_path = self.latent_dir / shard_name
        rows_for_table = []
        manifest_rows = []
        for row_idx, job in enumerate(self._current_rows):
            rows_for_table.append(
                {
                    "key": job["key"],
                    "country": job["country"],
                    "split": job["split"],
                    "projected_bytes": self._serialize_tensor(job["projected"]),
                    "speaker_prefix_prequant_bytes": self._serialize_tensor(job["speaker_prefix_prequant"]),
                    "num_frames": int(job["num_frames"]),
                    "speaker_prefix_frames": int(job["speaker_prefix_frames"]),
                }
            )
            manifest_rows.append(
                {
                    "key": job["key"],
                    "country": job["country"],
                    "split": job["split"],
                    "latent_shard_path": str(shard_path.relative_to(self.latents_root)),
                    "latent_row_idx": row_idx,
                    "num_frames": int(job["num_frames"]),
                    "speaker_prefix_frames": int(job["speaker_prefix_frames"]),
                }
            )

        table = pa.Table.from_pylist(rows_for_table)
        pq.write_table(table, shard_path)
        for row in manifest_rows:
            self.out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.saved += len(self._current_rows)
        self.out_f.flush()
        self._current_rows = []
        self._next_shard_idx += 1

    def submit_many(self, jobs: list[dict[str, Any]]) -> None:
        self._check_error()
        for job in jobs:
            self._queue.put(job)
            self._check_error()

    def flush(self) -> None:
        self._check_error()
        self._queue.put(self._flush_token)
        self._queue.join()
        self.out_f.flush()
        self._check_error()

    def close(self) -> None:
        self.flush()
        self._queue.put(self._sentinel)
        self._thread.join()
        self._check_error()

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Async latent saver failed") from self._error


# -----------------------------------------------------------------------------
# Batch prep
# -----------------------------------------------------------------------------

class AudioShardIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        hf_split,
        key_column: str,
        max_samples: int | None,
        skip_keys: set[str] | None = None,
    ) -> None:
        self.hf_split = hf_split
        self.key_column = key_column
        self.max_samples = None if max_samples is None else int(max_samples)
        self.skip_keys = skip_keys or set()

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        emitted = 0
        yielded = 0
        for i, sample in enumerate(self.hf_split):
            if i % num_workers != worker_id:
                continue

            key = sample.get(self.key_column)
            if key is not None and str(key) in self.skip_keys:
                continue

            if self.max_samples is not None and emitted >= self.max_samples:
                break

            emitted += 1
            yielded += 1
            yield sample


def prepare_batch(
    samples: list[dict[str, Any]],
    *,
    audio_column: str,
    key_column: str,
    target_sr: int,
    target_channels: int,
    frame_size: int,
    max_seconds: float | None,
) -> dict[str, Any]:
    """
    Creates a padded batch:
      audio: [B, C, T_pad]
      valid_frames: list[int] using ceil(orig_len / frame_size)
    """
    keys: list[str] = []
    wavs: list[torch.Tensor] = []
    valid_frames: list[int] = []
    rounded_lengths: list[int] = []

    for sample in samples:
        key = str(sample[key_column])
        try:
            wav, sr = coerce_audio(sample[audio_column])
        except Exception as e:
            print(f"[WARN] Skipping sample key={key}: failed to load audio ({e})")
            continue

        if sr != target_sr:
            print(
                f"[WARN] Skipping sample key={key}: sampling rate mismatch after cast "
                f"(got {sr}, expected {target_sr})"
            )
            continue

        try:
            wav = ensure_channels(wav, target_channels)
            wav = trim_audio(wav, sr, max_seconds)
        except Exception as e:
            print(f"[WARN] Skipping sample key={key}: invalid audio shape/content ({e})")
            continue

        if wav.numel() == 0 or wav.shape[-1] == 0:
            print(f"[WARN] Skipping sample key={key}: decoded audio is empty")
            continue

        T = int(wav.shape[-1])
        n_frames = ceil_div(T, frame_size)
        T_rounded = n_frames * frame_size

        keys.append(key)
        wavs.append(wav)
        valid_frames.append(n_frames)
        rounded_lengths.append(T_rounded)

    if not wavs:
        return {
            "keys": [],
            "audio": None,
            "valid_frames": [],
        }

    max_len = max(rounded_lengths)
    padded = []
    for wav in wavs:
        pad_right = max_len - int(wav.shape[-1])
        if pad_right > 0:
            wav = F.pad(wav, (0, pad_right))
        padded.append(wav)

    batch_audio = torch.stack(padded, dim=0)  # [B, C, T]

    return {
        "keys": keys,
        "audio": batch_audio,
        "valid_frames": valid_frames,
    }


def collate_prepared_batch(
    samples: list[dict[str, Any]],
    *,
    audio_column: str,
    key_column: str,
    target_sr: int,
    target_channels: int,
    frame_size: int,
    max_seconds: float | None,
) -> dict[str, Any]:
    prepared = prepare_batch(
        samples,
        audio_column=audio_column,
        key_column=key_column,
        target_sr=target_sr,
        target_channels=target_channels,
        frame_size=frame_size,
        max_seconds=max_seconds,
    )
    prepared["raw_count"] = len(samples)
    return prepared

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_full_yaml_config(args.config_path)
    dataset_cfg = cfg["dataset"]
    audio_cfg = cfg["audio"]
    output_cfg = cfg["output"]
    encoding_cfg = cfg["encoding"]
    mimi_config = cfg["mimi"]

    repo_id = str(dataset_cfg["repo_id"])
    split = str(dataset_cfg["split"])
    country = str(dataset_cfg["country"])
    audio_column = str(audio_cfg["audio_column"])
    key_column = str(dataset_cfg["key_column"])
    audio_should_cast = bool(audio_cfg.get("cast_audio_column", True))
    dataset_streaming = bool(dataset_cfg.get("streaming", True))

    device_name = encoding_cfg.get("device")
    if device_name in (None, "null"):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(device_name))

    latent_dir = Path(output_cfg["latent_dir"]).expanduser().resolve() / country / split
    latent_dir.mkdir(parents=True, exist_ok=True)
    latents_root = Path(output_cfg["latent_dir"]).expanduser().resolve()

    manifest_path = resolve_manifest_path(output_cfg=output_cfg, country=country, split=split)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_cfg.get("data_files") is not None:
        configured_data_files = dataset_cfg["data_files"]
        data_files: dict[str, Any] = {split: configured_data_files}
    else:
        data_files = make_data_files(dataset_cfg)
        data_files = filter_existing_splits(repo_id, data_files)

    if not data_files:
        raise RuntimeError("No split files found for the given country/patterns.")
    if split not in data_files:
        raise RuntimeError(f"Requested split '{split}' not available. Available: {sorted(data_files)}")

    split_shards = list_split_shards(repo_id, str(data_files[split]))
    if not split_shards:
        raise RuntimeError(f"No source parquet shards found for split '{split}'.")

    model = load_mimi_encoder(
        mimi_config=mimi_config,
        device=str(device),
    )
    model = maybe_compile_model_calls(model, encoding_cfg=encoding_cfg)

    frame_size = model.frame_size
    speaker_prefix_frames = math.ceil(
        float(encoding_cfg["speaker_proj_seconds"]) * float(mimi_config["frame_rate"])
    )
    written = 0
    seen = 0
    pbar = tqdm(desc="Encoding Mimi latents", unit="utt")
    batch_size = int(encoding_cfg["batch_size"])
    num_workers = int(encoding_cfg.get("num_workers", 1))
    prefetch_factor = int(encoding_cfg.get("prefetch_factor", 2))
    persistent_workers = bool(encoding_cfg.get("persistent_workers", num_workers > 0))
    flush_every_source_shards = max(1, int(encoding_cfg.get("flush_every_source_shards", 1)))
    max_samples = encoding_cfg.get("max_samples")
    existing_local_keys = load_existing_local_keys(manifest_path)
    progress_path = resolve_progress_path(latent_dir=latent_dir)
    progress = load_local_progress(progress_path)
    start_shard_idx = 0
    if progress is not None:
        last_completed = progress.get("last_completed_source_shard")
        if isinstance(last_completed, str) and last_completed in split_shards:
            start_shard_idx = split_shards.index(last_completed) + 1
            print(
                f"[LOCAL] Resuming split={split} from source shard "
                f"{start_shard_idx}/{len(split_shards)}"
            )
        elif last_completed:
            print(
                f"[WARN] Local progress points to missing shard '{last_completed}'. "
                "Starting from the first shard."
            )
    if existing_local_keys:
        print(
            f"[LOCAL] Resuming from {len(existing_local_keys)} existing latent files in {latent_dir}"
        )

    with manifest_path.open("a", encoding="utf-8") as out_f:
        saver = AsyncLatentSaver(
            latents_root=latents_root,
            latent_dir=latent_dir,
            out_f=out_f,
            samples_per_parquet=int(output_cfg.get("samples_per_parquet", 1000)),
            max_pending_jobs=int(encoding_cfg.get("max_pending_save_batches", 8)),
        )
        stop_early = False
        remaining_samples = None if max_samples is None else int(max_samples)
        completed_shards_since_flush = 0
        last_completed_source_shard_to_checkpoint: str | None = None
        try:
            for shard_idx, source_shard in enumerate(split_shards[start_shard_idx:], start=start_shard_idx):
                shard_ds = load_dataset(
                    path=repo_id,
                    name=dataset_cfg.get("config"),
                    data_files={split: [source_shard]},
                    streaming=dataset_streaming,
                    cache_dir=encoding_cfg.get("cache_dir"),
                )[split]

                if audio_should_cast:
                    shard_ds = shard_ds.cast_column(audio_column, Audio(sampling_rate=int(mimi_config["sample_rate"])))

                shard_dataset = AudioShardIterableDataset(
                    hf_split=shard_ds,
                    key_column=key_column,
                    max_samples=remaining_samples if num_workers <= 1 else None,
                    skip_keys=existing_local_keys,
                )
                collate_fn = partial(
                    collate_prepared_batch,
                    audio_column=audio_column,
                    key_column=key_column,
                    target_sr=int(mimi_config["sample_rate"]),
                    target_channels=int(mimi_config["channels"]),
                    frame_size=frame_size,
                    max_seconds=encoding_cfg.get("max_seconds"),
                )
                loader = DataLoader(
                    shard_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                    persistent_workers=persistent_workers if num_workers > 0 else False,
                    collate_fn=collate_fn,
                )

                for prepared in loader:
                    seen += int(prepared["raw_count"])
                    jobs = process_batch(
                        prepared=prepared,
                        model=model,
                        device=device,
                        speaker_prefix_frames=speaker_prefix_frames,
                        split=split,
                        country=country,
                    )
                    saver.submit_many(jobs)
                    batch_written = len(jobs)
                    written += batch_written
                    existing_local_keys.update(job["key"] for job in jobs)
                    pbar.update(int(prepared["raw_count"]))
                    if remaining_samples is not None:
                        remaining_samples -= int(prepared["raw_count"])
                        if remaining_samples <= 0:
                            stop_early = True
                    if stop_early:
                        break

                del loader, shard_dataset
                if not stop_early:
                    completed_shards_since_flush += 1
                    last_completed_source_shard_to_checkpoint = source_shard
                    if completed_shards_since_flush >= flush_every_source_shards:
                        saver.flush()
                        write_local_progress(
                            progress_path,
                            country=country,
                            split=split,
                            last_completed_source_shard=last_completed_source_shard_to_checkpoint,
                        )
                        completed_shards_since_flush = 0
                        last_completed_source_shard_to_checkpoint = None

                if stop_early:
                    break

            if not stop_early and last_completed_source_shard_to_checkpoint is not None:
                saver.flush()
                write_local_progress(
                    progress_path,
                    country=country,
                    split=split,
                    last_completed_source_shard=last_completed_source_shard_to_checkpoint,
                )
        finally:
            saver.close()
            out_f.flush()
            pbar.close()

    print(f"done. country={country} split={split} seen={seen} written={written} manifest={manifest_path}")


def process_batch(
    *,
    prepared: dict[str, Any],
    model: MimiEncoder,
    device: torch.device,
    speaker_prefix_frames: int,
    split: str,
    country: str,
) -> list[dict[str, Any]]:
    if prepared["audio"] is None:
        return []

    audio = prepared["audio"].to(device)  # [B, C, T]
    keys = prepared["keys"]
    valid_frames = prepared["valid_frames"]

    with torch.inference_mode():
        prequant = model.encode_to_latent(audio)   # [B, D, F]
        projected = model.quantize(prequant)       # [B, D', F]

    prequant = prequant.cpu()
    projected = projected.cpu()

    jobs: list[dict[str, Any]] = []
    for i, key in enumerate(keys):
        n_frames = int(valid_frames[i])
        prefix_frames = min(int(speaker_prefix_frames), n_frames)

        item_prequant = prequant[i, :, :prefix_frames].transpose(0, 1).contiguous()
        item_projected = projected[i, :, :n_frames].transpose(0, 1).contiguous()

        jobs.append(
            {
                "key": key,
                "country": country,
                "split": split,
                "projected": item_projected,
                "speaker_prefix_prequant": item_prequant,
                "num_frames": n_frames,
                "speaker_prefix_frames": prefix_frames,
            }
        )
    return jobs


if __name__ == "__main__":
    main()
