from __future__ import annotations

import math
from typing import Any

import torch

from training.data.collator import SpecialTokenIds
from training.utils.config import resolve_torch_dtype
from training.utils.metrics import (
    MetricCounts,
    compute_batch_metric_counts,
    finalize_metric_counts,
    merge_metric_counts,
)
from training.utils.wer import WERNormalizer, compute_wer, generate_batch_greedy, wer_stats

@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    special_tokens: SpecialTokenIds,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    counts = MetricCounts()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(batch)
        batch_counts = compute_batch_metric_counts(
            outputs["logits"],
            batch["packed_labels"],
            special_tokens=special_tokens,
            loss_value=float(outputs["loss"].detach().cpu()),
            unweighted_loss_value=float(outputs["unweighted_loss"].detach().cpu()),
            loss_sum=float(outputs["loss_sum"].detach().cpu()),
            loss_weight_sum=float(outputs["loss_weight_sum"].detach().cpu()),
            unweighted_loss_sum=float(outputs["unweighted_loss_sum"].detach().cpu()),
            token_count=int(outputs["token_count"].detach().cpu()),
        )
        merge_metric_counts(counts, batch_counts)

    if was_training:
        model.train()

    if counts.batch_count == 0:
        return {
            "loss": float("nan"),
            "unweighted_loss": float("nan"),
            "perplexity": float("nan"),
            "num_batches": 0.0,
            "num_tokens": 0.0,
        }
    return finalize_metric_counts(counts)


def select_eval_model(
    model: torch.nn.Module,
    *,
    ema,
    cfg: dict[str, Any],
) -> torch.nn.Module:
    eval_cfg = cfg.get("evaluation", {})
    if bool(eval_cfg.get("use_ema_for_eval", True)) and ema is not None:
        return ema.model
    return model


def _resolve_wer_delays_ms(wer_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> list[int]:
    raw_delay = wer_cfg.get("delay_ms", dataset_cfg.get("delay_max_ms", 2400))
    if isinstance(raw_delay, (list, tuple)):
        delays = [int(value) for value in raw_delay]
    else:
        delays = [int(raw_delay)]
    if not delays:
        raise ValueError("wer.delay_ms must contain at least one delay value.")
    return delays


@torch.no_grad()
def evaluate_wer(
    model: torch.nn.Module,
    dataloader,
    *,
    tokenizer,
    special_tokens: SpecialTokenIds,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, float]:
    wer_cfg = cfg.get("wer", {})
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    delays_ms = _resolve_wer_delays_ms(wer_cfg, cfg["dataset"])
    for delay_ms in delays_ms:
        if int(delay_ms) % step_ms != 0:
            raise ValueError("Each wer.delay_ms value must be a multiple of dataset.step_ms.")

    max_batches = wer_cfg.get("max_batches")
    max_batches = None if max_batches in (None, "null") else int(max_batches)
    max_decode_steps = wer_cfg.get("max_decode_steps")
    max_decode_steps = None if max_decode_steps in (None, "null") else int(max_decode_steps)
    data_dtype = resolve_torch_dtype(
        cfg["runtime"].get("data_dtype", "bf16"),
        default=torch.bfloat16,
    )
    assert data_dtype is not None

    normalizer = WERNormalizer(remove_diacritics=bool(wer_cfg.get("remove_diacritics", False)))
    metrics: dict[str, float] = {}
    flush_steps_cfg = wer_cfg.get("flush_steps")
    extra_flush_steps = int(wer_cfg.get("extra_flush_steps", 128))

    for delay_idx, delay_ms in enumerate(delays_ms):
        delay_steps = int(delay_ms) // step_ms
        flush_steps = delay_steps + extra_flush_steps if flush_steps_cfg in (None, "null") else int(flush_steps_cfg)
        total_errors = 0
        total_ref_words = 0
        total_samples = 0

        for batch_idx, samples in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            hypotheses = generate_batch_greedy(
                model,
                samples,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                device=device,
                data_dtype=data_dtype,
                left_pad_steps=int(cfg["dataset"].get("left_pad_steps", 0)),
                delay_steps=delay_steps,
                flush_steps=flush_steps,
                max_decode_steps=max_decode_steps,
            )
            for sample, hypothesis in zip(samples, hypotheses, strict=True):
                errors, ref_words = wer_stats(
                    str(sample["transcription"]),
                    hypothesis,
                    normalizer,
                )
                total_errors += int(errors)
                total_ref_words += int(ref_words)
                total_samples += 1

        suffix = f"delay_{delay_idx}_{int(delay_ms)}ms"
        metrics[f"wer/{suffix}"] = compute_wer(total_errors, total_ref_words)
        metrics[f"wer_errors/{suffix}"] = float(total_errors)
        metrics[f"wer_ref_words/{suffix}"] = float(total_ref_words)
        metrics[f"wer_num_samples/{suffix}"] = float(total_samples)
        metrics[f"wer_flush_steps/{suffix}"] = float(flush_steps)

    if len(delays_ms) == 1:
        suffix = f"delay_0_{int(delays_ms[0])}ms"
        metrics["wer"] = metrics[f"wer/{suffix}"]
        metrics["wer_errors"] = metrics[f"wer_errors/{suffix}"]
        metrics["wer_ref_words"] = metrics[f"wer_ref_words/{suffix}"]
        metrics["wer_num_samples"] = metrics[f"wer_num_samples/{suffix}"]
        metrics["wer_flush_steps"] = metrics[f"wer_flush_steps/{suffix}"]

    return metrics
