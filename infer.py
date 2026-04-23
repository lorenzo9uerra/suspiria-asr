#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
import wave

import hydra
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.signal import resample_poly

from preprocessing.encode_latents import load_mimi_encoder
from training.data.collator import SpecialTokenIds
from training.tokenizer import load_tokenizer
from training.utils.config import resolve_device
from training.utils.logging import silence_external_info_logs
from training.utils.model_builder import build_model
from training.utils.wer import generate_batch_greedy


silence_external_info_logs()


def resolve_weight_path(path_or_hf: str | Path) -> str:
    value = str(path_or_hf)
    local_path = Path(to_absolute_path(str(Path(value).expanduser())))
    if local_path.exists():
        return str(local_path.resolve())
    if value.startswith("hf://"):
        return str(download_hf_path(value.removeprefix("hf://")))
    parts = value.split("/")
    if len(parts) >= 3 and not value.startswith(("/", "./", "../")):
        return str(download_hf_path(value))
    return str(local_path)


def download_hf_path(hf_path: str) -> Path:
    parts = hf_path.split("/")
    if len(parts) < 3:
        raise ValueError("HF paths must look like `hf://owner/repo/file` or `hf://owner/repo/file@revision`.")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    revision = None
    if "@" in filename:
        filename, revision = filename.rsplit("@", 1)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def read_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    path = Path(to_absolute_path(str(Path(path).expanduser())))
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = int(wav_file.getframerate())
            channels = int(wav_file.getnchannels())
            sample_width = int(wav_file.getsampwidth())
            raw = wav_file.readframes(wav_file.getnframes())
        if sample_width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width {sample_width} bytes: {path}")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return torch.from_numpy(samples).unsqueeze(0), sample_rate

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError("soundfile is required to read non-WAV audio files.") from e
    data, sample_rate = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    return wav, int(sample_rate)


def convert_audio(wav: torch.Tensor, *, from_rate: int, to_rate: int, to_channels: int) -> torch.Tensor:
    if wav.ndim != 2:
        raise ValueError(f"Expected [C, T] audio, got {tuple(wav.shape)}")
    if wav.shape[0] != to_channels:
        if to_channels == 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.shape[0] == 1:
            wav = wav.repeat(to_channels, 1)
        else:
            raise ValueError(f"Cannot convert {wav.shape[0]} channels to {to_channels}")
    if from_rate != to_rate:
        gcd = int(np.gcd(int(from_rate), int(to_rate)))
        up = int(to_rate) // gcd
        down = int(from_rate) // gcd
        wav_np = resample_poly(wav.detach().cpu().numpy(), up, down, axis=-1)
        wav = torch.from_numpy(wav_np).to(dtype=torch.float32)
    return wav.float().contiguous()


def pad_to_frame_multiple(wav: torch.Tensor, frame_size: int) -> tuple[torch.Tensor, int]:
    num_frames = int((wav.shape[-1] + frame_size - 1) // frame_size)
    target_len = num_frames * frame_size
    if target_len == wav.shape[-1]:
        return wav, num_frames
    padded = torch.zeros(wav.shape[0], target_len, dtype=wav.dtype)
    padded[:, : wav.shape[-1]] = wav
    return padded, num_frames


def resolve_delay_ms(value: object, cfg: dict[str, Any]) -> int:
    if value is not None:
        return int(value)
    return int(cfg.get("timeline", {}).get("delay_ms", 2400))


def model_builder_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    timeline = cfg.get("timeline", {})
    return {
        "model": cfg["model"],
        "tokenizer": cfg["tokenizer"],
        "loss": cfg.get("loss", {}),
        "runtime": cfg.get("runtime", {}),
        "dataset": {
            "step_ms": int(timeline.get("step_ms", 80)),
            "left_pad_steps": int(timeline.get("left_pad_steps", 0)),
        },
    }


def build_special_tokens(resolved_tokenizer) -> SpecialTokenIds:
    return SpecialTokenIds(
        bos=int(resolved_tokenizer.bos_token_id),
        eos=int(resolved_tokenizer.eos_token_id),
        pad_wait=int(resolved_tokenizer.pad_wait_token_id),
        word_start=int(resolved_tokenizer.word_start_token_id),
    )


def load_decoder_safetensors(decoder: torch.nn.Module, weights_path: str | Path) -> None:
    state_dict = safetensors.torch.load_file(str(weights_path))
    missing, unexpected = decoder.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Decoder weight mismatch: missing={missing} unexpected={unexpected}")


@hydra.main(version_base=None, config_path="configs/inference", config_name="offline-ita")
def main(cfg: DictConfig) -> None:
    infer_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(infer_cfg, dict):
        raise ValueError("Expected mapping config for inference.")
    audio_path = infer_cfg["audio"]
    tokenizer_path = infer_cfg["tokenizer"]["name"]
    decoder_weights = infer_cfg["decoder"]["weights_path"]
    train_cfg = model_builder_cfg(infer_cfg)

    runtime_cfg = dict(train_cfg.get("runtime", {}))
    inference_runtime_cfg = infer_cfg.get("runtime", {})
    device_value = inference_runtime_cfg.get("device")
    if device_value is not None:
        runtime_cfg["device"] = device_value
    device = resolve_device(runtime_cfg)

    tokenizer_cfg = dict(infer_cfg["tokenizer"])
    tokenizer_cfg["name"] = tokenizer_path
    resolved_tokenizer = load_tokenizer(tokenizer_cfg)
    tokenizer = resolved_tokenizer.tokenizer
    special_tokens = build_special_tokens(resolved_tokenizer)

    decoder = build_model(
        train_cfg,
        vocab_size=len(tokenizer),
        device=device,
        special_tokens=special_tokens,
    )
    load_decoder_safetensors(decoder, resolve_weight_path(decoder_weights))
    decoder.eval().to(device=device, dtype=torch.bfloat16)

    mimi_cfg = dict(infer_cfg["mimi"])
    if mimi_cfg.get("weights_path") is not None:
        mimi_cfg["weights_path"] = resolve_weight_path(mimi_cfg["weights_path"])
    mimi = load_mimi_encoder(mimi_config=mimi_cfg, device=str(device))
    mimi.eval().to(device=device)

    wav, source_rate = read_audio(audio_path)
    wav = convert_audio(
        wav,
        from_rate=source_rate,
        to_rate=int(mimi_cfg["sample_rate"]),
        to_channels=int(mimi_cfg["channels"]),
    )
    timeline_cfg = infer_cfg.get("timeline", {})
    max_audio_seconds = timeline_cfg.get("max_audio_seconds")
    if max_audio_seconds is not None:
        wav = wav[:, : int(float(max_audio_seconds) * int(mimi_cfg["sample_rate"]))]
    wav, num_frames = pad_to_frame_multiple(wav, int(mimi.frame_size))

    with torch.inference_mode():
        audio = wav.unsqueeze(0).to(device=device)
        prequant = mimi.encode_to_latent(audio)
        projected = mimi.quantize(prequant)[0, :, :num_frames].transpose(0, 1).contiguous().cpu()

    step_ms = int(timeline_cfg.get("step_ms", 80))
    delay_ms = resolve_delay_ms(timeline_cfg.get("delay_ms"), train_cfg)
    if delay_ms % step_ms != 0:
        raise ValueError(f"delay-ms={delay_ms} must be divisible by step_ms={step_ms}")
    delay_steps = delay_ms // step_ms
    flush_steps_value = timeline_cfg.get("flush_steps")
    flush_steps = (
        delay_steps + int(timeline_cfg.get("extra_flush_steps", 128))
        if flush_steps_value is None
        else int(flush_steps_value)
    )
    max_decode_steps = timeline_cfg.get("max_decode_steps")
    max_decode_steps = None if max_decode_steps is None else int(max_decode_steps)

    print(
        f"[INFER] audio={audio_path} source_sr={source_rate} "
        f"mimi_sr={mimi_cfg['sample_rate']} frames={projected.shape[0]} delay_ms={delay_ms}"
    )
    transcript = generate_batch_greedy(
        decoder,
        [{"key": Path(audio_path).name, "projected": projected}],
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        device=device,
        data_dtype=torch.bfloat16,
        left_pad_steps=int(timeline_cfg.get("left_pad_steps", 0)),
        delay_steps=delay_steps,
        flush_steps=flush_steps,
        max_decode_steps=max_decode_steps,
    )[0]
    print(transcript)


if __name__ == "__main__":
    main()
