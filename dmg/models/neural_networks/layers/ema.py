import torch
import torch.nn as nn
from typing import Optional

class ExponentialMovingAverage(nn.Module):
    def __init__(
        self,
        alpha: float = 0.33,
        per_channel: bool = True,
        learnable: bool = True,
        channels: Optional[int] = None,
    ):
        """
        alpha: 初始平滑系数 (0,1]
        per_channel: 若 True，则为每个通道维护独立 alpha
        learnable: 若 True，则把 alpha 设为可学习参数（内部用 sigmoid 映射）
        channels: 当 per_channel 或 learnable=True 时需要提供 channels
        """
        super().__init__()
        assert 0 < alpha <= 1.0
        self.per_channel = per_channel
        self.learnable = learnable

        if learnable:
            assert channels is not None
            # 用 raw_param 经 sigmoid 映射到 (0,1)
            init_raw = torch.logit(torch.tensor(alpha))
            if per_channel:
                self.raw = nn.Parameter(init_raw.repeat(channels))
            else:
                self.raw = nn.Parameter(init_raw.repeat(1))
        else:
            if per_channel:
                assert channels is not None
                self.register_buffer("alpha", torch.full((channels,), alpha))
            else:
                self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        device = x.device

        if self.learnable:
            a = torch.sigmoid(self.raw)
        else:
            a = self.alpha

        a = a.to(device)

        if a.numel() == 1:
            a = a.repeat(C)

        a = a.view(-1)  # (C,)

        # iterative EMA using list to avoid inplace
        y_list = [x[:, 0, :]]
        one_minus_a = 1.0 - a
        for t in range(1, T):
            y_t = a * x[:, t, :] + one_minus_a * y_list[-1]
            y_list.append(y_t)
        y = torch.stack(y_list, dim=1)
        return y