from __future__ import annotations

from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from models.decoder import DecoderLM
from training.config import DecoderConfig
from training.utils.config import resolve_torch_dtype


def build_model(cfg: dict[str, Any], *, vocab_size: int, device: torch.device) -> DecoderLM:
    model_cfg = DecoderConfig(
        vocab_size=vocab_size,
        audio_input_dim=int(cfg["model"]["audio_input_dim"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        num_kv_heads=int(cfg["model"]["num_kv_heads"]),
        ffw_hidden_size=int(cfg["model"]["ffw_hidden_size"]),
        attention_window=int(cfg["model"].get("attention_window", 8192)),
        rope_theta=float(cfg["model"].get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(cfg["model"].get("rms_norm_eps", 1e-5)),
        max_position_embeddings=int(cfg["model"].get("max_position_embeddings", 16384)),
        time_condition_dim=int(cfg["model"].get("time_condition_dim", 32)),
        time_embedding_theta=float(cfg["model"].get("time_embedding_theta", 10_000.0)),
        tie_word_embeddings=bool(cfg["model"].get("tie_word_embeddings", True)),
    )
    model_dtype = resolve_torch_dtype(
        cfg["runtime"].get("model_dtype", "bf16"),
        default=torch.bfloat16,
    )
    return DecoderLM(model_cfg).to(device=device, dtype=model_dtype)


def _strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    state_dict = payload.get("model", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported model state type: {type(state_dict)}")
    return _strip_prefix_if_present(state_dict, "module.")


def load_pretrained_model_weights(
    model: torch.nn.Module,
    *,
    weights_path: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    resolved_path = Path(weights_path).expanduser().resolve()
    if str(resolved_path).endswith(".safetensors"):
        payload = safetensors.torch.load_file(str(resolved_path))
    else:
        payload = torch.load(resolved_path, map_location="cpu")
    state_dict = _extract_model_state_dict(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)
