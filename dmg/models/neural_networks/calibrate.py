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
            num_basins=559, # todo 当流域总数改变时这个需要改变
            num_start=config["nmul"],
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        batch_indices = x['batch_sample']
        cur_params = self.params[batch_indices]
        return None, F.sigmoid(cur_params)
