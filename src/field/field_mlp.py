from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
import torch.nn as nn

from .field import Field
from ..components import PositionalEncoding


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)
        # raise NotImplementedError("This is your homework.")
        if cfg.positional_encoding_octaves is not None:
            d_coordinate = PositionalEncoding(cfg.positional_encoding_octaves).d_out(d_coordinate)
            
        #mlp
        layers = [nn.Linear(d_coordinate, cfg.d_hidden), nn.ReLU()] if \
            cfg.positional_encoding_octaves is None else \
                [PositionalEncoding(cfg.positional_encoding_octaves), nn.Linear(d_coordinate, cfg.d_hidden), nn.ReLU()]
        for _ in range(cfg.num_hidden_layers - 1):
            layers.append(nn.Linear(cfg.d_hidden, cfg.d_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.d_hidden, d_out))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        # raise NotImplementedError("This is your homework.")
        return self.mlp(coordinates)
