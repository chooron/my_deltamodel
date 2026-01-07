import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class DplUHBase(nn.Module, ABC):
    """
    可微单位线基类 (Grouped Conv1d + Flip修复)
    """

    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        # 注册时间索引 [1, 2, ..., max_lag]
        self.register_buffer(
            "t_idx",
            torch.arange(1, self.max_lag + 1, dtype=torch.float32).view(
                1, 1, -1
            ),
        )
    @abstractmethod
    def get_weights(self, params):
        raise NotImplementedError

    def forward(self, flux_in, params):
        # 维度适配
        if params.dim() == 1:
            params = params.unsqueeze(-1)
        batch_size, time_steps = flux_in.shape

        # 1. 生成并归一化权重
        raw_weights = self.get_weights(params)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)

        # 2. 翻转权重以匹配物理卷积定义
        flipped_weights = torch.flip(norm_weights, dims=[-1])

        # 3. 准备卷积输入 (Causal Padding)
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))

        # 4. 执行卷积
        flux_out = F.conv1d(
            input=padded_x, weight=flipped_weights, groups=batch_size
        )

        return flux_out.view(batch_size, time_steps)
