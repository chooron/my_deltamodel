import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class Calibrate(torch.nn.Module):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        num_basins: int = 100,
        num_start: int = 10,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = "Calibrate"
        self.params = nn.Parameter(
            torch.randn(num_basins, ny, num_start, device=device)
        )

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        return cls(
            nx=config["nx2"],
            ny=config["ny"],
            num_basins=config["batch_size"],
            num_start=config["nmul"],
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        return None, F.sigmoid(self.params)
