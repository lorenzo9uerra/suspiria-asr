from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    audio_input_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    ffw_hidden_size: int
    attention_window: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    time_condition_dim: int
    time_embedding_theta: float
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    pad_wait_token_id: int
    word_start_token_id: int
    loss_eos_factor: float
    loss_pad_wait_factor: float
    loss_word_start_factor: float


@dataclass(frozen=True)
class TrainingConfig:
    dataset: dict[str, Any]
    tokenizer: dict[str, Any]
    model: dict[str, Any]
    optimization: dict[str, Any]
    runtime: dict[str, Any]
