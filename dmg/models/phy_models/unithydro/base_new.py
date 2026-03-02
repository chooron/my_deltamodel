import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class DplUHBase(nn.Module, ABC):
    """
    可微单位线基类 (S-Curve Difference Method + Causal Conv)
    """
    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        
        # [修改] 生成整数时间点 0, 1, 2, ..., max_lag
        # 用来计算累积曲线 S(0), S(1), S(2)...
        self.register_buffer(
            "t_seq", 
            torch.arange(0, self.max_lag + 1, dtype=torch.float32).view(1, 1, -1)
        )

    def _get_t_seq(self, params):
        # 动态获取以匹配 device
        return torch.arange(0, self.max_lag + 1, device=params.device, dtype=params.dtype).view(1, 1, -1)

    @abstractmethod
    def get_s_curve(self, params, t_seq):
        """
        计算累积 S 曲线
        Returns S(t) for t in [0, 1, ..., max_lag]
        """
        raise NotImplementedError

    def forward(self, flux_in, params):
        if params.dim() == 1:
            params = params.unsqueeze(-1)
        batch_size, time_steps = flux_in.shape

        # 1. 获取时间序列 0, 1, ..., max_lag
        t_seq = self._get_t_seq(params)

        # 2. 计算 S-Curve [Batch, 1, max_lag+1]
        s_curve = self.get_s_curve(params, t_seq)
        
        # 3. 差分计算权重 w(t) = S(t) - S(t-1)
        # s_curve[..., 1:] 取索引 1..L (对应 t=1..L)
        # s_curve[..., :-1] 取索引 0..L-1 (对应 t=0..L-1)
        # 结果长度为 max_lag
        raw_weights = s_curve[..., 1:] - s_curve[..., :-1]
        
        # 4. 归一化 (可选，S-Curve 理论上最终趋于1，但截断可能导致不足1)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)

        # 5. 翻转 + 卷积 (手动左填充)
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        
        x = flux_in.view(1, batch_size, time_steps)
        pad_size = self.max_lag - 1
        padded_x = F.pad(x, (pad_size, 0))

        flux_out = F.conv1d(
            input=padded_x, 
            weight=flipped_weights, 
            groups=batch_size,
            padding=0
        )

        return flux_out.view(batch_size, time_steps)