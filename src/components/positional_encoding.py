import torch.nn as nn
from jaxtyping import Float
import torch
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        # raise NotImplementedError("This is your homework.")
        self.num_octaves = num_octaves

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        frequencies = [2 ** i for i in range(self.num_octaves)] # num_octaves
        encoded = [torch.sin(f * samples) for f in frequencies] + \
            [torch.cos(f * samples) for f in frequencies] # 2 * num_octaves
        output = torch.cat(encoded, dim=-1)
        return output
        # raise NotImplementedError("This is your homework.")

    def d_out(self, dimensionality: int):
        # raise NotImplementedError("This is your homework.")
        return 2 * self.num_octaves * dimensionality
        
