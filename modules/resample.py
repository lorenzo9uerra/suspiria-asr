import torch
from torch import nn

from modules.conv import StreamingConv1d

class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    """

    def __init__(self, stride: int, dimension: int):
        super().__init__()
        self.conv = StreamingConv1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    def __init__(self, stride: int, dimension: int):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        del model_state
        return self.convtr(x)
