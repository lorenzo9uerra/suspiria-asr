from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def _is_embedding_parameter(name: str) -> bool:
    return name == "embed_tokens.weight" or name.endswith(".embed_tokens.weight")


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    params_total = 0
    params_trainable = 0
    params_no_embed = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        params_total += count
        if not param.requires_grad:
            continue
        params_trainable += count
        if not _is_embedding_parameter(name):
            params_no_embed += count
    return {
        "params_total": params_total,
        "params_trainable": params_trainable,
        "params_no_embed": params_no_embed,
    }


def estimate_flops_per_token(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    observed_max_seq_len: int | None = None,
) -> dict[str, int | float | None]:
    counts = count_parameters(model)
    model_cfg = cfg["model"]
    scaling_cfg = cfg.get("scaling", {})
    hidden_size = int(model_cfg["hidden_size"])
    num_heads = int(model_cfg["num_heads"])
    num_layers = int(model_cfg["num_layers"])
    head_dim = hidden_size // num_heads
    attention_window = int(model_cfg.get("attention_window", 8192))
    configured_context = scaling_cfg.get("flops_context_tokens")
    context = observed_max_seq_len if configured_context in (None, "", "null", "None") else int(configured_context)
    if context is None or context <= 0:
        context = attention_window if attention_window > 0 else int(model_cfg.get("max_position_embeddings", 0))
    effective_context = min(attention_window, int(context)) if attention_window > 0 else int(context)

    parameter_flops = 6 * int(counts["params_no_embed"])
    attention_flops = 12 * num_layers * num_heads * head_dim * effective_context
    flops_per_token = int(parameter_flops + attention_flops)
    return {
        **counts,
        "flops_per_token": flops_per_token,
        "parameter_flops_per_token": int(parameter_flops),
        "attention_flops_per_token": int(attention_flops),
        "effective_context": int(effective_context),
        "observed_max_seq_len": None if observed_max_seq_len is None else int(observed_max_seq_len),
        "head_dim": int(head_dim),
    }


def build_scaling_payload(
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    step: int,
    target_tokens: int,
    tokens_seen: int,
    observed_max_seq_len: int | None,
    best_train_loss: float | None,
    best_val_loss: float | None,
    best_val_step: int | None,
    best_val_tokens_seen: int | None,
    best_val_metrics: dict[str, float] | None,
    final_val_metrics: dict[str, float] | None,
    final_test_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    flops = estimate_flops_per_token(
        model,
        cfg,
        observed_max_seq_len=observed_max_seq_len,
    )
    flops_per_token = int(flops["flops_per_token"])
    token_overshoot = max(0, int(tokens_seen) - int(target_tokens))
    token_overshoot_ratio = float(token_overshoot) / float(max(1, int(target_tokens)))
    return {
        "scaling_enabled": True,
        "step": int(step),
        "model_name": cfg.get("scaling", {}).get("model_name"),
        "target_tokens": int(target_tokens),
        "tokens_seen": int(tokens_seen),
        "token_overshoot": int(token_overshoot),
        "token_overshoot_ratio": token_overshoot_ratio,
        "params_total": int(flops["params_total"]),
        "params_trainable": int(flops["params_trainable"]),
        "params_no_embed": int(flops["params_no_embed"]),
        "flops_per_token": flops_per_token,
        "declared_compute_flops": int(target_tokens) * flops_per_token,
        "actual_execution_flops": int(tokens_seen) * flops_per_token,
        "flops_estimate": flops,
        "best_train_loss": _finite_float_or_none(best_train_loss),
        "best_val_loss": _finite_float_or_none(best_val_loss),
        "best_val_step": None if best_val_step is None else int(best_val_step),
        "best_val_tokens_seen": None if best_val_tokens_seen is None else int(best_val_tokens_seen),
        "best_val_metrics": best_val_metrics,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "model_args": cfg.get("model", {}),
        "config": cfg,
    }


def save_scaling_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
