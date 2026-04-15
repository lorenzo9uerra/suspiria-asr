import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.rope import RotaryEmbedding


def _materialize_causal_mask(
    shape: tuple[int, ...], shift: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    dtype = torch.float32

    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    tensor = torch.full(shape, dtype=dtype, fill_value=1, device=device)
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    mask = torch.log(mask)
    return mask.to(dtype)


class StreamingMultiheadAttention(nn.Module):
    """Similar to `nn.MultiheadAttention` but with support for streaming.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        rope: RotaryEmbedding class

    """

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        mult = 1
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, mult * embed_dim, bias=False)


    def forward(self, query: torch.Tensor, attn_mask: torch.Tensor):
        # TODO: check attn_mask is correct

        projected = self.in_proj(query)
        # Reshape from (b, t, p*h*d) to (b, t, p, h, d) where p=3, h=num_heads
        b, t, _ = projected.shape
        d = self.embed_dim // self.num_heads
        packed = projected.view(b, t, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)
        q, k = self.rope(q,k)

        # Provide correct mask
        #attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2)
        # Reshape from (b, t, h, d) to (b, t, h*d)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- fake batch ----
    B = 2
    T = 8
    d_model = 32
    num_heads = 4
    max_period = 10_000.0

    rope = RotaryEmbedding(max_period=max_period).to(device)
    mask = torch.tril(torch.ones(T, T)).bool().to(device)
    print(mask)
    print(mask.shape)
    attn = StreamingMultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        rope=rope,
    ).to(device)

    x = torch.randn(B, T, d_model, device=device)

    # ---- stateless forward ----
    with torch.no_grad():
        y = attn(x, mask)

    print("input shape :", x.shape)
    print("output shape:", y.shape)

    assert y.shape == (B, T, d_model)
    assert torch.isfinite(y).all(), "NaNs or infs in attention output!"

    print("Stateless attention forward pass OK ✅")
