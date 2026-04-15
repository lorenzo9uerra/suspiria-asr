import warnings
import torch
from torch import nn
from torch.nn import functional as F

from modules.stateful_module import StatefulModule

class StreamingConv1d(StatefulModule):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ["constant", "replicate"], pad_mode
        self.pad_mode = pad_mode
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @property
    def _stride(self) -> int:
        return self.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        dilation = self.conv.dilation[0]
        return (self._kernel_size - 1) * dilation + 1  # effective kernel size with dilations

    def init_state(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        stride = self._stride
        # Effective kernel size accounting for dilation.
        kernel = self._effective_kernel_size
        previous = torch.zeros(batch_size, self.conv.in_channels, kernel - stride, device=device)
        first = torch.ones(batch_size, dtype=torch.bool, device=device)
        return dict(previous=previous, first=first)

    def forward(self, x, model_state: dict | None):
        B, C, T = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        if model_state is None:
            state = self.init_state(B, 0, device=x.device)
        else:
            state = self.get_state(model_state)
        TP = state["previous"].shape[-1]
        if TP and self.pad_mode == "replicate":
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            state["previous"][:] = torch.where(
                state["first"].view(-1, 1, 1), init, state["previous"]
            )
        if TP:
            x = torch.cat([state["previous"], x], dim=-1)
        y = self.conv(x)
        if TP:
            state["previous"][:] = x[..., -TP:]
            if self.pad_mode == "replicate":
                state["first"] = torch.zeros_like(state["first"])
        return y
