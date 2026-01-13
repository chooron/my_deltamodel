import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


# class DplUHBase(nn.Module, ABC):
#     """
#     可微单位线基类 (Grouped Conv1d + Flip修复)
#     """

#     def __init__(self, max_lag, epsilon=1e-6):
#         super().__init__()
#         self.max_lag = int(max_lag)
#         self.epsilon = epsilon
#         # 注册时间索引 [1, 2, ..., max_lag]
#         self.register_buffer(
#             "t_idx",
#             torch.arange(1, self.max_lag + 1, dtype=torch.float32).view(
#                 1, 1, -1
#             ),
#         )
#     @abstractmethod
#     def get_weights(self, params):
#         raise NotImplementedError

#     def forward(self, flux_in, params):
#         # 维度适配
#         if params.dim() == 1:
#             params = params.unsqueeze(-1)
#         batch_size, time_steps = flux_in.shape

#         # 1. 生成并归一化权重
#         raw_weights = self.get_weights(params)
#         sum_w = raw_weights.sum(dim=-1, keepdim=True)
#         norm_weights = raw_weights / (sum_w + self.epsilon)

#         # 2. 翻转权重以匹配物理卷积定义
#         flipped_weights = torch.flip(norm_weights, dims=[-1])

#         # 3. 准备卷积输入 (Causal Padding)
#         x = flux_in.view(1, batch_size, time_steps)
#         padded_x = F.pad(x, (self.max_lag - 1, 0))

#         # 4. 执行卷积
#         flux_out = F.conv1d(
#             input=padded_x, weight=flipped_weights, groups=batch_size
#         )

#         return flux_out.view(batch_size, time_steps)

class DplUHBase(nn.Module, ABC):
    """
    可微单位线基类 (Grouped Conv1d)
    【严格对齐版】
    1. 时间采样：从 0.5 开始 (与 Reference uh_gamma 对齐)
    2. 填充逻辑：Conv1d symmetric padding + Slicing (与 Reference uh_conv 对齐)
    """

    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        
        # [关键对齐 1] 时间索引 t
        # 参考代码: t = torch.arange(0.5, lenF * 1.0)
        # 解释: 物理上降雨发生在 dt 期间，取中点 0.5 代表该时段的平均响应。
        # 如果不改这里，波形会整体滞后 0.5 个时间步。
        self.register_buffer(
            "t_idx",
            torch.arange(0.5, self.max_lag * 1.0, dtype=torch.float32).view(
                1, 1, -1
            ),
        )

    @abstractmethod
    def get_weights(self, params):
        """
        子类实现具体分布（如 Gamma），返回未归一化的权重。
        Shape: [batch_size, 1, max_lag]
        """
        raise NotImplementedError

    def forward(self, flux_in, params):
        # 维度适配
        if params.dim() == 1:
            params = params.unsqueeze(-1)
        
        batch_size, time_steps = flux_in.shape

        # --- 步骤 1: 权重生成 (保持不变) ---
        raw_weights = self.get_weights(params)
        
        # 归一化 (Mass Balance)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)

        # --- 步骤 2: 权重翻转 (严格对齐 Reference) ---
        # Reference: torch.flip(w, [2])
        # 卷积操作即积分，需要将核翻转
        flipped_weights = torch.flip(norm_weights, dims=[-1])

        # --- 步骤 3: 卷积与填充 (严格对齐 Reference) ---
        # 准备输入: [1, batch, time] 以利用 groups=batch 进行并行计算
        x = flux_in.view(1, batch_size, time_steps)
        
        # 定义 Padding 大小
        # Reference: m = UH.shape[-1]; padd = m - 1
        padd = self.max_lag - 1

        # 执行卷积 (Symetric Padding)
        # Reference: F.conv1d(..., padding=padd, ...)
        # 注意: padding=padd 会在序列的“左边”和“右边”都补 padd 个 0
        flux_out = F.conv1d(
            input=x, 
            weight=flipped_weights, 
            groups=batch_size,
            padding=padd 
        )

        # --- 步骤 4: 裁剪 (严格对齐 Reference) ---
        # Reference: if padd != 0: y = y[:, :, 0:-padd]
        # 解释: 因为是对称填充，右边多补了 0，导致输出变长，必须切掉多余部分
        # 才能保证输出长度与输入长度一致 (Causal)
        if padd > 0:
            flux_out = flux_out[:, :, 0:-padd]

        return flux_out.view(batch_size, time_steps)