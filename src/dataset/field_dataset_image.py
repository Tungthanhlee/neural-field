from jaxtyping import Float
from omegaconf import DictConfig
import torch
from torch import Tensor
import torchvision.io as io
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms.functional import to_tensor

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)
        # raise NotImplementedError("This is your homework.")
        # self.image = (io.read_image(cfg.path) / 255.).unsqueeze(0) # N, C, H, W
        self.image = to_tensor(Image.open(cfg.path)).unsqueeze(0).cuda()
        self.cfg = cfg

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """
        #check if coordinate and image in the same device
        assert coordinates.device == self.image.device
        N, C, H, W = self.image.shape

        num_coords, d_coordinate = coordinates.shape
        normalized_coord = coordinates * 2 - 1 #map to [-1, 1]
        grid = normalized_coord.view(1, num_coords, 1, d_coordinate) # 1, N, 1, D
        
        sample = F.grid_sample(
            self.image, # 1, C, H, W
            grid, # 1, N, 1, D
            mode='bilinear',
            align_corners=True
        ) # 1, C, N, 1
        return sample.squeeze(0).squeeze(-1).transpose(0,1) # N, C
        # raise NotImplementedError("This is your homework.")
        

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        # raise NotImplementedError("This is your homework.")
        return self.image.shape[2:]
        
