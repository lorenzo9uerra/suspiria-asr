import math
import torch
from torch import nn


def apply_rope(q: torch.Tensor, k: torch.Tensor, max_period: int | float = 10_000):
    """
    RoPE with offset fixed to 0 (training-time), computed on the fly.

    Args:
        q: [B, T, H, D]
        k: [B, T, Hk, D]
        max_period: base for rotation frequencies

    Returns:
        (q_rot, k_rot) with same shapes as inputs.
    """
    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert (B, T, D) == (Bk, Tk, Dk), (q.shape, k.shape)
    assert D > 0 and D % 2 == 0
    assert max_period > 0

    # freqs: [D/2] in fp32
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(float(max_period)) * 2.0 / D))

    # positions with offset=0: [T]
    ts = torch.arange(T, device=q.device, dtype=torch.float32)

    # angles: [T, D/2] -> broadcast later
    angles = ts[:, None] * freqs[None, :]
    rotr = torch.cos(angles)[None, :, None, :]  # [1, T, 1, D/2]
    roti = torch.sin(angles)[None, :, None, :]  # [1, T, 1, D/2]

    # pack into complex pairs
    q2 = q.view(B, T, H, D // 2, 2)
    k2 = k.view(B, T, Hk, D // 2, 2)

    qr = q2[..., 0].float()
    qi = q2[..., 1].float()
    kr = k2[..., 0].float()
    ki = k2[..., 1].float()

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    out_dtype = q.dtype
    q_out = torch.stack((qor.to(out_dtype), qoi.to(out_dtype)), dim=-1).view(B, T, H, D)
    k_out = torch.stack((kor.to(out_dtype), koi.to(out_dtype)), dim=-1).view(B, T, Hk, D)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """RoPE with offset fixed to 0, computed on the fly."""

    def __init__(self, max_period: float | int = 10_000.0):
        super().__init__()
        self.max_period = float(max_period)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return apply_rope(q, k, max_period=self.max_period)