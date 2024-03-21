from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
import torch

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP
from ..components import PositionalEncoding


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3
        
        #positional encoding for z
        self.z_pos_embedding = PositionalEncoding(cfg.positional_encoding_octaves)
        zd_coordinate = PositionalEncoding(cfg.positional_encoding_octaves).d_out(1)
        
        #grid using x and y coordinates
        self.grid_field = FieldGrid(
            cfg.grid,
            2,
            cfg.d_grid_feature,
        )
        
        #concatenate the grid's outputs with the corresponding encoded Z values, then feed the result through the MLP
        self.mlp_field = FieldMLP(
            cfg.mlp,
            cfg.d_grid_feature + zd_coordinate,
            d_out
        )

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """
        
        #get coordinates
        z_coords = coordinates[:,-1] # N,
        z_coords = z_coords.unsqueeze(-1) # Nx1
        xy_coords = coordinates[:,:-1] # Nx2
        
        # Get the grid's outputs using the XY coordinates and the Z coordinates
        xy_grid_field = self.grid_field(xy_coords) # Nx d_grid_feature
        z_pos_embedding = self.z_pos_embedding(z_coords) # Nx zd_coordinate
        
        #concatenate the grid's outputs with the corresponding encoded Z values
        concat_field = torch.cat((xy_grid_field, z_pos_embedding), dim=-1) # Nx (d_grid_feature + zd_coordinate)
        out_mlp_field = self.mlp_field(concat_field)
        
        return out_mlp_field
