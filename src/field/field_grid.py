from jaxtyping import Float
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from .field import Field


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)
        
        # raise NotImplementedError("This is your homework.")
        C = d_out #predefined channel
        if d_coordinate == 2:
            self.grid_field = nn.Parameter(
                torch.randn(1, C, cfg.side_length, cfg.side_length)
            ) # 1, C, H, W since we dealing with 2D RGB image
        elif d_coordinate == 3:
            self.grid_field = nn.Parameter(
                torch.randn(1, C, cfg.side_length, cfg.side_length, cfg.side_length)
            ) # 1, C, D, H, W 
        else:
            raise ValueError("d_coordinate must be 2 or 3")

        self.d_coordinate = d_coordinate
        self.d_out = d_out
    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """
        num_coords, d_coords = coordinates.shape
        normalized_coord = coordinates * 2 - 1 # map to [-1,1], see https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        # raise NotImplementedError("This is your homework.")
        if self.d_coordinate == 2:
            grid = normalized_coord.view(1, num_coords, 1, d_coords) # 1, N, 1, D
            output = F.grid_sample(
                self.grid_field, # 1, C, H, W
                grid, # 1, N, 1, D
                mode='bilinear',
                align_corners=True
            ) # 1, C, N, 1
            output = output.squeeze(0).squeeze(-1).transpose(0,1) # N, C
        elif self.d_coordinate == 3:
            grid = normalized_coord.view(1, num_coords, 1, 1, d_coords) # 1, N, 1, 1, D
            output = F.grid_sample(
                self.grid_field, # 1, C, D, H, W
                grid, # 1, N, 1, 1, D
                mode='bilinear',
                align_corners=True
            ) # 1, C, N, 1, 1
            output = output.squeeze(0).squeeze(-1).squeeze(-1).transpose(0,1) # N, C
        else:
            raise ValueError("d_coordinate must be 2 or 3")
        return output
