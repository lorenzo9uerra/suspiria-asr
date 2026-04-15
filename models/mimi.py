import logging

import torch
from torch import nn

from modules.dummy_quantizer import DummyQuantizer
from modules.mimi_transformer import ProjectedTransformer
from modules.resample import ConvDownsample1d
from modules.seanet import SEANetEncoder

logger = logging.getLogger()


class MimiEncoder(nn.Module):
    def __init__(
        self,
        encoder: SEANetEncoder,
        quantizer: DummyQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        encoder_transformer: ProjectedTransformer,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_transformer = encoder_transformer
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoder_frame_rate = encoder_frame_rate

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(dimension, int), (
            f"Dimension should be int, got {dimension} of type {type(dimension)}."
        )
        self.dimension = dimension

        if encoder_frame_rate != frame_rate:
            assert self.encoder_frame_rate > self.frame_rate, "Cannot upsample with conv."
            downsample_stride = self.encoder_frame_rate / self.frame_rate
            assert downsample_stride == int(downsample_stride), (
                f"Only integer strides are supported, got {downsample_stride}"
            )
            self.downsample = ConvDownsample1d(int(downsample_stride), dimension=dimension)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def _to_framerate(self, x: torch.Tensor):
        # Convert from the encoder frame rate to the overall framerate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        return self.downsample(x, model_state=None)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()


    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.
           This requires the batch of samples already padded to the same length and multiple of frame size.   
           The outputs is converted to the encoder frame rate so given input of shape [B, C, T],
           it return tensor of shape [B, C, T'] where T' = T / frame_size.
           frame_size = sample_rate / frame_size.
           Encoder with streaming convolution is used with None state.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
            emb (torch.Tensor): Float tensor of shape [B, C, T']

        """
        assert x.dim() == 3, (
            f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        )

        emb = self.encoder(x, model_state=None)
        (emb,) = self.encoder_transformer(emb)
        emb = self._to_framerate(emb)
        return emb
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use quantizer to project to latent dimension. 
        The output of this layer is used for teacher forcing training.
        
        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]
        
        Returns:
            emb (torch.Tensor) Float tensor of shape [B, C', T]

        """

        return self.quantizer.input_proj(x)
