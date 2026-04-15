from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.config import DecoderConfig

try:
    from torch.nn.attention.varlen import varlen_attn
except Exception:
    varlen_attn = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * norm * self.weight).to(x.dtype)


def _apply_rope(x: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    total_tokens, num_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even.")
    half_dim = head_dim // 2
    device = x.device
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    x_even = x[..., ::2].float()
    x_odd = x[..., 1::2].float()
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = rot_even.to(x.dtype)
    out[..., 1::2] = rot_odd.to(x.dtype)
    return out


class VarLenSelfAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope_theta = float(config.rope_theta)
        self.attention_window = int(config.attention_window)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if varlen_attn is None:
            raise ImportError(
                "PyTorch varlen attention is unavailable. Install a PyTorch build exposing "
                "`torch.nn.attention.varlen.varlen_attn`."
            )

        max_len = int(seq_lens.max().item())

        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        # `position_ids` is packed but resets to 0 for every sequence; `cu_seqlens`
        # gives varlen_attn the sequence boundaries so packed samples never mix.
        q = _apply_rope(q, position_ids, self.rope_theta)
        k = _apply_rope(k, position_ids, self.rope_theta)

        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1).contiguous()
            v = v.repeat_interleave(repeat_factor, dim=1).contiguous()

        window_size = (-1, 0) if self.attention_window <= 0 else (self.attention_window, 0)
        attn_out = varlen_attn(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_len,
            max_len,
            window_size=window_size,
        )
        attn_out = attn_out.reshape(-1, self.hidden_size)
        return self.o_proj(attn_out)


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = VarLenSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffw_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffw_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffw_hidden_size, config.hidden_size, bias=False)
        self.ada_down = nn.Linear(config.hidden_size, config.time_condition_dim, bias=False)
        self.ada_up = nn.Linear(config.time_condition_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        time_condition: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.attn_norm(hidden_states)
        hidden_states = hidden_states + self.self_attn(
            attn_input,
            seq_lens=seq_lens,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
        )

        ffn_input = self.ffn_norm(hidden_states)
        scale = self.ada_up(F.gelu(self.ada_down(time_condition)))
        ffn_input = ffn_input * (1.0 + scale)
        gate = F.silu(self.gate_proj(ffn_input))
        up = self.up_proj(ffn_input)
        hidden_states = hidden_states + self.down_proj(gate * up)
        return hidden_states


class DecoderLM(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_proj = nn.Linear(config.audio_input_dim, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_token_loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=-100,
            reduction="none",
        )
        valid_mask = labels != -100
        if not valid_mask.any():
            zero = per_token_loss.sum()
            return zero, zero

        weights = torch.ones_like(per_token_loss)
        if self.config.loss_bos_factor != 1.0:
            weights = torch.where(
                labels == self.config.bos_token_id,
                weights * float(self.config.loss_bos_factor),
                weights,
            )
        if self.config.loss_eos_factor != 1.0:
            weights = torch.where(
                labels == self.config.eos_token_id,
                weights * float(self.config.loss_eos_factor),
                weights,
            )
        if self.config.loss_pad_wait_factor != 1.0:
            weights = torch.where(
                labels == self.config.pad_wait_token_id,
                weights * float(self.config.loss_pad_wait_factor),
                weights,
            )
        if self.config.loss_word_start_factor != 1.0:
            weights = torch.where(
                labels == self.config.word_start_token_id,
                weights * float(self.config.loss_word_start_factor),
                weights,
            )

        valid_mask_f = valid_mask.to(per_token_loss.dtype)
        weighted_loss = per_token_loss * weights * valid_mask_f
        unweighted_loss = per_token_loss * valid_mask_f
        normalizer = (weights * valid_mask_f).sum()
        unweighted_normalizer = valid_mask_f.sum()
        return (
            weighted_loss.sum() / normalizer,
            unweighted_loss.sum() / unweighted_normalizer,
        )

    def _compute_time_embedding(self, delay_steps: torch.Tensor) -> torch.Tensor:
        device = delay_steps.device
        half_dim = self.config.hidden_size // 2
        inv_freq = torch.exp(
            -math.log(float(self.config.time_embedding_theta))
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(1, half_dim)
        )
        per_sample = delay_steps.to(torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)
        return torch.cat([torch.cos(per_sample), torch.sin(per_sample)], dim=-1)

    def _expand_time_condition(self, delay_steps: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        sample_embed = self._compute_time_embedding(delay_steps)
        repeated = []
        for idx, length in enumerate(seq_lens.tolist()):
            repeated.append(sample_embed[idx : idx + 1].expand(length, -1))
        return torch.cat(repeated, dim=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["packed_input_ids"]
        audio_features = batch["packed_audio_features"]
        seq_lens = batch["seq_lens"]
        packed_position_ids = batch["packed_position_ids"]
        cu_seqlens = batch["cu_seqlens"]

        hidden_states = self.embed_tokens(input_ids) + self.audio_proj(audio_features)
        time_condition = self._expand_time_condition(batch["delay_steps"], seq_lens)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                seq_lens=seq_lens,
                position_ids=packed_position_ids,
                cu_seqlens=cu_seqlens,
                time_condition=time_condition,
            )

        hidden_states = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states)
        loss, unweighted_loss = self._compute_loss(logits, batch["packed_labels"])
        return {"loss": loss, "unweighted_loss": unweighted_loss, "logits": logits}
