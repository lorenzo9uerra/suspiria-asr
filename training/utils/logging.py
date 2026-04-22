from __future__ import annotations

import logging

import torch


def silence_external_info_logs() -> None:
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def format_param_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def print_model_parameter_summary(model: torch.nn.Module) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    total = trainable + frozen
    print(
        "[MODEL] "
        f"parameters total={format_param_count(total)} ({total:,}) "
        f"trainable={format_param_count(trainable)} ({trainable:,}) "
        f"frozen={format_param_count(frozen)} ({frozen:,})"
    )
