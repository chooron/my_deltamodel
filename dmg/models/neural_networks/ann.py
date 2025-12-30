import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class AnnModel(torch.nn.Module):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = "AnnModel"
        self.hidden_size = hidden_size
        self.core = nn.Sequential(
            nn.Linear(nx, hidden_size),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_size, ny),
        ).to(torch.device(device))

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        return cls(
            nx=config["nx2"],
            ny=config["ny"],
            hidden_size=config["hidden_size"],
            dr=config["dr"],
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        nn_out = self.core(x["c_nn_norm"])
        return None, F.sigmoid(nn_out)
