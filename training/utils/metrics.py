from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from training.data.collator import SpecialTokenIds


@dataclass
class MetricCounts:
    loss_sum: float = 0.0
    loss_weight_sum: float = 0.0
    unweighted_loss_sum: float = 0.0
    token_count: int = 0
    batch_count: int = 0
    valid_count: int = 0
    correct_count: int = 0
    top5_correct_count: int = 0
    non_pad_count: int = 0
    non_pad_correct_count: int = 0
    pad_count: int = 0
    pad_correct_count: int = 0
    text_count: int = 0
    text_correct_count: int = 0
    bos_count: int = 0
    bos_correct_count: int = 0
    eos_count: int = 0
    eos_correct_count: int = 0
    word_start_count: int = 0
    word_start_correct_count: int = 0
    emit_tp: int = 0
    emit_fp: int = 0
    emit_fn: int = 0
    text_required_pred_pad_count: int = 0
    pad_required_pred_text_count: int = 0
    word_start_tp: int = 0
    word_start_fp: int = 0
    word_start_fn: int = 0


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def _f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        return float("nan")
    return 2.0 * precision * recall / (precision + recall)


def _build_masks(
    labels: torch.Tensor,
    pred_ids: torch.Tensor,
    *,
    special_tokens: SpecialTokenIds,
) -> dict[str, torch.Tensor]:
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    pad_mask = valid_mask & (labels == special_tokens.pad_wait)
    word_start_mask = valid_mask & (labels == special_tokens.word_start)
    bos_mask = valid_mask & (labels == special_tokens.bos)
    eos_mask = valid_mask & (labels == special_tokens.eos)

    special_label_mask = (
        (labels == special_tokens.bos)
        | (labels == special_tokens.eos)
        | (labels == special_tokens.pad_wait)
        | (labels == special_tokens.word_start)
    )
    text_mask = valid_mask & (~special_label_mask)

    pred_text_mask = valid_mask & (
        (pred_ids != special_tokens.bos)
        & (pred_ids != special_tokens.eos)
        & (pred_ids != special_tokens.pad_wait)
        & (pred_ids != special_tokens.word_start)
    )
    return {
        "valid": valid_mask,
        "pad": pad_mask,
        "word_start": word_start_mask,
        "bos": bos_mask,
        "eos": eos_mask,
        "text": text_mask,
        "pred_text": pred_text_mask,
    }


@torch.no_grad()
def compute_batch_metric_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    special_tokens: SpecialTokenIds,
    loss_value: float,
    unweighted_loss_value: float | None = None,
    loss_sum: float | None = None,
    loss_weight_sum: float | None = None,
    unweighted_loss_sum: float | None = None,
    token_count: int | None = None,
) -> MetricCounts:
    pred_ids = logits.argmax(dim=-1)
    top5_ids = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
    masks = _build_masks(labels, pred_ids, special_tokens=special_tokens)
    valid_mask = masks["valid"]
    correct = (pred_ids == labels) & valid_mask
    top5_correct = (top5_ids == labels.unsqueeze(-1)).any(dim=-1) & valid_mask

    emit_gold = masks["text"]
    emit_pred = masks["pred_text"]
    emit_tp = int((emit_gold & emit_pred).sum().item())
    emit_fp = int(((~emit_gold) & valid_mask & emit_pred).sum().item())
    emit_fn = int((emit_gold & (~emit_pred)).sum().item())

    word_start_pred = valid_mask & (pred_ids == special_tokens.word_start)
    word_start_tp = int((masks["word_start"] & word_start_pred).sum().item())
    word_start_fp = int(((~masks["word_start"]) & valid_mask & word_start_pred).sum().item())
    word_start_fn = int((masks["word_start"] & (~word_start_pred)).sum().item())

    valid_count = int(valid_mask.sum().item())
    resolved_token_count = valid_count if token_count is None else int(token_count)
    resolved_loss_weight_sum = 1.0 if loss_weight_sum is None else float(loss_weight_sum)
    resolved_loss_sum = float(loss_value) * resolved_loss_weight_sum if loss_sum is None else float(loss_sum)
    resolved_unweighted_loss_sum = (
        float(loss_value if unweighted_loss_value is None else unweighted_loss_value) * resolved_token_count
        if unweighted_loss_sum is None
        else float(unweighted_loss_sum)
    )

    return MetricCounts(
        loss_sum=resolved_loss_sum,
        loss_weight_sum=resolved_loss_weight_sum,
        unweighted_loss_sum=resolved_unweighted_loss_sum,
        token_count=resolved_token_count,
        batch_count=1,
        valid_count=valid_count,
        correct_count=int(correct.sum().item()),
        top5_correct_count=int(top5_correct.sum().item()),
        non_pad_count=int((valid_mask & (~masks["pad"])).sum().item()),
        non_pad_correct_count=int((correct & (~masks["pad"])).sum().item()),
        pad_count=int(masks["pad"].sum().item()),
        pad_correct_count=int((correct & masks["pad"]).sum().item()),
        text_count=int(masks["text"].sum().item()),
        text_correct_count=int((correct & masks["text"]).sum().item()),
        bos_count=int(masks["bos"].sum().item()),
        bos_correct_count=int((correct & masks["bos"]).sum().item()),
        eos_count=int(masks["eos"].sum().item()),
        eos_correct_count=int((correct & masks["eos"]).sum().item()),
        word_start_count=int(masks["word_start"].sum().item()),
        word_start_correct_count=int((correct & masks["word_start"]).sum().item()),
        emit_tp=emit_tp,
        emit_fp=emit_fp,
        emit_fn=emit_fn,
        text_required_pred_pad_count=int((masks["text"] & (pred_ids == special_tokens.pad_wait)).sum().item()),
        pad_required_pred_text_count=int((masks["pad"] & emit_pred).sum().item()),
        word_start_tp=word_start_tp,
        word_start_fp=word_start_fp,
        word_start_fn=word_start_fn,
    )


def merge_metric_counts(dst: MetricCounts, src: MetricCounts) -> MetricCounts:
    for field_name in dst.__dataclass_fields__:
        setattr(dst, field_name, getattr(dst, field_name) + getattr(src, field_name))
    return dst


def finalize_metric_counts(counts: MetricCounts) -> dict[str, float]:
    emit_precision = _safe_div(counts.emit_tp, counts.emit_tp + counts.emit_fp)
    emit_recall = _safe_div(counts.emit_tp, counts.emit_tp + counts.emit_fn)
    word_precision = _safe_div(counts.word_start_tp, counts.word_start_tp + counts.word_start_fp)
    word_recall = _safe_div(counts.word_start_tp, counts.word_start_tp + counts.word_start_fn)
    avg_loss = _safe_div(counts.loss_sum, counts.loss_weight_sum)
    avg_unweighted_loss = _safe_div(counts.unweighted_loss_sum, counts.token_count)
    try:
        perplexity = math.exp(avg_unweighted_loss) if not math.isnan(avg_unweighted_loss) else float("nan")
    except OverflowError:
        perplexity = float("inf")
    return {
        "loss": avg_loss,
        "unweighted_loss": avg_unweighted_loss,
        "perplexity": perplexity,
        "num_batches": float(counts.batch_count),
        "num_tokens": float(counts.token_count),
        "overall/token_accuracy": _safe_div(counts.correct_count, counts.valid_count),
        "overall/top5_accuracy": _safe_div(counts.top5_correct_count, counts.valid_count),
        "overall/non_pad_accuracy": _safe_div(counts.non_pad_correct_count, counts.non_pad_count),
        "overall/pad_accuracy": _safe_div(counts.pad_correct_count, counts.pad_count),
        "overall/text_token_accuracy": _safe_div(counts.text_correct_count, counts.text_count),
        "control/bos_accuracy": _safe_div(counts.bos_correct_count, counts.bos_count),
        "control/eos_accuracy": _safe_div(counts.eos_correct_count, counts.eos_count),
        "control/word_start_accuracy": _safe_div(counts.word_start_correct_count, counts.word_start_count),
        "emit_text/precision": emit_precision,
        "emit_text/recall": emit_recall,
        "emit_text/f1": _f1(emit_precision, emit_recall),
        "emit_text/text_required_pred_pad_rate": _safe_div(counts.text_required_pred_pad_count, counts.text_count),
        "emit_text/pad_required_pred_text_rate": _safe_div(counts.pad_required_pred_text_count, counts.pad_count),
        "word_start/precision": word_precision,
        "word_start/recall": word_recall,
        "word_start/f1": _f1(word_precision, word_recall),
    }
