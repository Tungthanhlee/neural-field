from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn
import torch

from .field.field import Field


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        # generate samples
        xyz_sample_locations, sample_boundaries = self.generate_samples(
            origins, directions, near, far, self.cfg.num_samples
        )
        
        # evaluate neural field
        out_field = self.field(xyz_sample_locations) # [batch, sample, 4]
        
        # map to valid output ranges
        out_field[..., :3] = torch.sigmoid(out_field[..., :3]) # [batch, sample, 3]
        
        # compute alpha values
        alphas = self.compute_alpha_values(
            out_field[..., -1],
            sample_boundaries,
        )
        
        # composite these alpha values together with the evaluated colors
        c = self.alpha_composite(
            alphas,
            out_field[..., :3],
        )
        
        return c

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """

        #compute steps between near and far
        sample_boundaries = torch.linspace(near, far, num_samples+1, device=origins.device) # [sample+1]
        
        #map steps to sample locations
        mid_points_sample = (sample_boundaries[:-1] + sample_boundaries[1:]) / 2 # [sample]
        xyz_sample_locations = origins[:, None] + directions[:, None] * mid_points_sample[..., None] # [batch, sample, 3]
        
        return xyz_sample_locations, sample_boundaries

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """

        segment_length = boundaries[..., 1:] - boundaries[..., :-1] # [batch, sample]
        alpha = 1 - torch.exp(-sigma * segment_length) # [batch, sample]
        return alpha

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """

        #compute transmittance
        T = torch.cumprod(1 - alphas, dim=-1) # [batch, sample]
        
        #compute weight
        w = alphas * T # [batch, sample]
        
        #compute expected radiance along the ray: c = sum_i=1^n w_i * colors_i
        c = torch.sum(w[..., None] * colors, dim=1) # [batch, 3] sum over samples
        
        return c
