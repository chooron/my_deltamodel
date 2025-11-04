from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmg.models.neural_networks.layers.ann import AnnModel
from dmg.models.neural_networks.layers.hope import Hope
from dmg.models.neural_networks.layers.ema import (
    ExponentialMovingAverage,
    SimpleMovingAverage,
)


class HopeMlpV2(torch.nn.Module):
    def __init__(
        self,
        *,
        nx1: int,
        ny1: int,
        hiddeninv1: int,
        nx2: int,
        ny2: int,
        hiddeninv2: int,
        dr1: Optional[float] = 0.5,
        dr2: Optional[float] = 0.5,
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = "HopeMlpModel"
        config = {
            "lr_min": 0.001,
            "lr": 0.01,
            "lr_dt": 0.0,
            "min_dt": 0.001,
            "max_dt": 0.01,  # change to 0.01
            "wd": 0.01,  # change to 0.01
            "d_state": 256,
            "cfr": 1.2,
            "cfi": 0.8,
            "use_gated": False,
            "out_activation": 'glu',
        }
        self.hope_layer = Hope(
            input_size=nx1,
            hidden_size=hiddeninv1,
            dropout=dr1,
            cfg=config,
            prenorm=False,
            n_layers=3,
        )
        self.fc = nn.Linear(hiddeninv1, ny1, bias=True)
        # self.norm = nn.LayerNorm(hiddeninv1)
        # self.ma = SimpleMovingAverage(
        #     channels=hiddeninv1, kernel_size=11, per_channel=True
        # )
        # self.a = 0.5
        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

    @classmethod
    def build_by_config(cls, config: dict, device: Optional[str] = "cpu"):
        return cls(
            nx1=config["nx"],
            nx2=config["nx2"],
            ny1=config["ny1"],
            ny2=config["ny2"],
            hiddeninv1=config["hope_hidden_size"],
            hiddeninv2=config["mlp_hidden_size"],
            dr1=config["hope_dropout"],
            dr2=config["mlp_dropout"],
            device=device,
        )
        

    def forward(
        self,
        data_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = data_dict["xc_nn_norm"]
        z2 = data_dict["c_nn_norm"]
        hope_out = self.hope_layer(
            torch.permute(z1, (1, 0, 2))
        ).permute((1, 0, 2))  # dim: timesteps, gages, params
        # fc_out = self.fc(self.ma(self.norm(hope_out)))
        fc_out = self.fc(hope_out)
        ann_out = self.ann(z2)
        return (F.tanh(fc_out) + 1) / 2, F.sigmoid(ann_out)
