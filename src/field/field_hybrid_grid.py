from jaxtyping import Float
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP



class FieldHybridGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a hybrid grid-mlp neural field. You should reuse FieldGrid from
        src/field/field_grid.py and FieldMLP from src/field/field_mlp.py in your
        implementation.

        Hint: Since you're reusing existing components, you only need to add one line
        each to __init__ and forward!
        """
        super().__init__(cfg, d_coordinate, d_out)
        d_grid_feature = cfg.d_grid_feature
        
        #grid
        self.grid_field = FieldGrid(
            cfg.grid,
            d_coordinate,
            d_grid_feature
        )
        
        #mlp
        self.mlp_field = FieldMLP(
            cfg.mlp,
            d_grid_feature,
            d_out
        )

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        # raise NotImplementedError("This is your homework.")
        out_grid_field = self.grid_field(coordinates)
        return self.mlp_field(out_grid_field)
