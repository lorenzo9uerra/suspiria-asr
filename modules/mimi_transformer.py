
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing_extensions import Self

from modules.layer_scale import LayerScale
from modules.rope import RotaryEmbedding
from modules.stateful_module import StatefulModule
from modules.transformer import StreamingMultiheadAttention
from utils.config import FlowLMTransformerConfig


class MimiStreamingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        out_dim = 3 * embed_dim

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)


    def forward(self, query: torch.Tensor, attn_mask: None = None) -> torch.Tensor:
        B, T = query.shape[:2]

        projected = self.in_proj(query)
        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

        # RoPE expects [B, T, H, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q, k = self.rope(q, k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # positions (stateless): keys at 0..T-1 for every batch item
        pos = torch.arange(T, device=q.device, dtype=torch.long)          # [T]
        pos_k = pos[None, None, :].expand(B, 1, T)                        # [B, 1, T]
        pos_q = pos[None, :, None].expand(B, T, 1)                        # [B, T, 1]

        delta = pos_q - pos_k                                              # [B, T, T]

        attn_mask = (delta >= 0) & (delta < self.context)                  # [B, T, T]
        attn_mask = attn_mask[:, None, :, :]                               # [B, 1, T, T]

        x = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_proj(x)
        return x

class StreamingTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
        attention_kind: str = "mimi",
    ):
        super().__init__()
        # Redefine self_attn to our streaming multi-head attention
        if attention_kind == "mimi":
            # TODO: we should actually use StreamingMultiheadAttention here and add context window
            # support. And we should then delete MimiStreamingMultiheadAttention.
            # The implementation is really close.
            self.self_attn = MimiStreamingMultiheadAttention(
                context=context, rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        else:
            self.self_attn = StreamingMultiheadAttention(
                rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)


    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(F.gelu(self.linear1(x)))
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, attn_mask) 
        return x_orig.to(update) + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = self._sa_block(x, attn_mask)
        x = self._ff_block(x)
        return x


class StreamingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int | list[int] = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period

        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    attention_kind=kind,
                )
            )

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig) -> Self:
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            kind="flow_lm",
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class ProjectedTransformer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float,
        context: int,
        max_period: float,
        dim_feedforward: int,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
        )
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def forward(self, x, attn_mask: None = None):
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, attn_mask)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = y.transpose(1, 2)
            ys.append(y)
        return ys
    



if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- fake batch ----
    B = 2
    T = 8
    d_model = 32
    num_heads = 4
    context = 2
    max_period = 10_000.0

    rope = RotaryEmbedding(max_period=max_period).to(device)

    attn = MimiStreamingMultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        context=context,
        rope=rope,
    ).to(device)

    x = torch.randn(B, T, d_model, device=device)

    # ---- stateless forward ----
    with torch.no_grad():
        y = attn(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)

    assert y.shape == (B, T, d_model)
    assert torch.isfinite(y).all(), "NaNs or infs in attention output!"

    print("Stateless Mimi attention forward pass OK ✅")
