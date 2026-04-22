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
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads for GQA, got "
                f"num_heads={self.num_heads} num_kv_heads={self.num_kv_heads}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        if varlen_attn is None:
            raise ImportError(
                "PyTorch varlen attention is unavailable. Install a PyTorch build exposing "
                "`torch.nn.attention.varlen.varlen_attn`."
            )

        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        # `position_ids` is packed but resets to 0 for every sequence; `cu_seqlens`
        # gives varlen_attn the sequence boundaries so packed samples never mix.
        q = _apply_rope(q, position_ids, self.rope_theta)
        k = _apply_rope(k, position_ids, self.rope_theta)

        window_size = (-1, 0) if self.attention_window <= 0 else (self.attention_window, 0)
        attn_out = varlen_attn(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seq_len,
            max_seq_len,
            window_size=window_size,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        attn_out = attn_out.reshape(-1, self.hidden_size)
        return self.o_proj(attn_out)

    def forward_generate_step(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, self.num_kv_heads, self.head_dim)

        q = _apply_rope(q, position_ids, self.rope_theta)
        k = _apply_rope(k, position_ids, self.rope_theta)

        if cache is None:
            k_cache = k.unsqueeze(1)
            v_cache = v.unsqueeze(1)
        else:
            k_cache = torch.cat([cache["key"], k.unsqueeze(1)], dim=1)
            v_cache = torch.cat([cache["value"], v.unsqueeze(1)], dim=1)

        if self.attention_window > 0 and k_cache.shape[1] > self.attention_window:
            k_cache = k_cache[:, -self.attention_window :].contiguous()
            v_cache = v_cache[:, -self.attention_window :].contiguous()

        q = q.unsqueeze(2)
        attn_k = k_cache.transpose(1, 2)
        attn_v = v_cache.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            attn_k,
            attn_v,
            is_causal=False,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, self.hidden_size)
        return self.o_proj(attn_out), {"key": k_cache, "value": v_cache}

    def forward_generate_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len = hidden_states.shape[:2]
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = _apply_rope(q.reshape(-1, self.num_heads, self.head_dim), position_ids.reshape(-1), self.rope_theta)
        k = _apply_rope(k.reshape(-1, self.num_kv_heads, self.head_dim), position_ids.reshape(-1), self.rope_theta)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v_cache = v

        attn_k = k.transpose(1, 2)
        attn_v = v_cache.transpose(1, 2)
        attn_mask = None
        is_causal = True
        if self.attention_window > 0:
            positions = torch.arange(seq_len, device=hidden_states.device)
            query_pos = positions[:, None]
            key_pos = positions[None, :]
            attn_mask = (key_pos <= query_pos) & (key_pos >= query_pos - self.attention_window + 1)
            attn_mask = attn_mask[None, None, :, :]
            is_causal = False
        attn_out = F.scaled_dot_product_attention(
            q,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        k_cache = k
        if self.attention_window > 0 and k_cache.shape[1] > self.attention_window:
            k_cache = k_cache[:, -self.attention_window :].contiguous()
            v_cache = v_cache[:, -self.attention_window :].contiguous()

        return self.o_proj(attn_out), {"key": k_cache, "value": v_cache}


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
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
        time_condition: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.attn_norm(hidden_states)
        hidden_states = hidden_states + self.self_attn(
            attn_input,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
        )

        ffn_input = self.ffn_norm(hidden_states)
        scale = self.ada_up(F.gelu(self.ada_down(time_condition)))
        ffn_input = ffn_input * (1.0 + scale)
        gate = F.silu(self.gate_proj(ffn_input))
        up = self.up_proj(ffn_input)
        hidden_states = hidden_states + self.down_proj(gate * up)
        return hidden_states

    def forward_generate_step(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        time_condition: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_input = self.attn_norm(hidden_states)
        attn_out, new_cache = self.self_attn.forward_generate_step(
            attn_input,
            position_ids=position_ids,
            cache=cache,
        )
        hidden_states = hidden_states + attn_out

        ffn_input = self.ffn_norm(hidden_states)
        scale = self.ada_up(F.gelu(self.ada_down(time_condition)))
        ffn_input = ffn_input * (1.0 + scale)
        gate = F.silu(self.gate_proj(ffn_input))
        up = self.up_proj(ffn_input)
        hidden_states = hidden_states + self.down_proj(gate * up)
        return hidden_states, new_cache

    def forward_generate_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        time_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_input = self.attn_norm(hidden_states)
        attn_out, new_cache = self.self_attn.forward_generate_prefill(
            attn_input,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attn_out

        ffn_input = self.ffn_norm(hidden_states)
        scale = self.ada_up(F.gelu(self.ada_down(time_condition))).unsqueeze(1)
        ffn_input = ffn_input * (1.0 + scale)
        gate = F.silu(self.gate_proj(ffn_input))
        up = self.up_proj(ffn_input)
        hidden_states = hidden_states + self.down_proj(gate * up)
        return hidden_states, new_cache


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

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        per_token_loss = F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )
        weights = torch.ones_like(per_token_loss)
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

        weighted_loss = per_token_loss * weights
        loss_sum = weighted_loss.sum()
        loss_weight_sum = weights.sum()
        unweighted_loss_sum = per_token_loss.sum()
        token_count = torch.tensor(labels.numel(), device=labels.device, dtype=per_token_loss.dtype)
        return {
            "loss": loss_sum / loss_weight_sum,
            "unweighted_loss": unweighted_loss_sum / token_count,
            "loss_sum": loss_sum,
            "loss_weight_sum": loss_weight_sum,
            "unweighted_loss_sum": unweighted_loss_sum,
            "token_count": token_count,
        }

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
        max_seq_len = int(batch["max_seq_len"].item())

        hidden_states = self.embed_tokens(input_ids) + self.audio_proj(audio_features)
        time_condition = self._expand_time_condition(batch["delay_steps"], seq_lens).to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=packed_position_ids,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                time_condition=time_condition,
            )

        hidden_states = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states)
        loss_outputs = self._compute_loss(logits, batch["packed_labels"])
        loss_outputs["logits"] = logits
        return loss_outputs

    @torch.no_grad()
    def forward_generate_prefill(
        self,
        *,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        position_ids: torch.Tensor,
        delay_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids) + self.audio_proj(audio_features)
        time_condition = self._compute_time_embedding(delay_steps).to(hidden_states.dtype)

        new_cache = []
        for layer in self.layers:
            hidden_states, next_layer_cache = layer.forward_generate_prefill(
                hidden_states,
                position_ids=position_ids,
                time_condition=time_condition,
            )
            new_cache.append(next_layer_cache)

        hidden_states = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states)
        return logits, new_cache

    @torch.no_grad()
    def forward_generate_step(
        self,
        *,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        position_ids: torch.Tensor,
        delay_steps: torch.Tensor,
        kv_cache: list[dict[str, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids) + self.audio_proj(audio_features)
        time_condition = self._compute_time_embedding(delay_steps).to(hidden_states.dtype)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)
        new_cache = []
        for layer, layer_cache in zip(self.layers, kv_cache, strict=True):
            hidden_states, next_layer_cache = layer.forward_generate_step(
                hidden_states,
                position_ids=position_ids,
                time_condition=time_condition,
                cache=layer_cache,
            )
            new_cache.append(next_layer_cache)

        hidden_states = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden_states)
        return logits, new_cache
