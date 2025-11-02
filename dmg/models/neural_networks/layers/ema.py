import torch
import torch.nn as nn
from typing import Optional

class SimpleMovingAverage(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,  # 窗口大小，奇数推荐以中心对齐
        per_channel: bool = True,  # 若True，每个通道独立
        channels: int = 1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for symmetric padding"
        self.kernel_size = kernel_size
        self.per_channel = per_channel
        if per_channel:
            assert channels is not None
            groups = channels
        else:
            groups = 1
        
        # Conv1d: 输入通道=channels, 输出=channels, groups=groups确保独立
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding, 保持序列长度
            bias=False,
            groups=groups
        )
        # 初始化为均匀权重 (1/kernel_size)
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        # x: (B, T, C) -> 需permute到 (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)  # (B, C, T)
        y = self.conv(x)
        return y.permute(0, 2, 1)  # back to (B, T, C)

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