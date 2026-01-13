import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

# ==========================================
# 1. 你的代码实现 (保持原样)
# ==========================================

class DplUHBase(nn.Module, ABC):
    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        self.register_buffer(
            "t_idx",
            torch.arange(1, self.max_lag + 1, dtype=torch.float32).view(1, 1, -1),
        )
    @abstractmethod
    def get_weights(self, params):
        raise NotImplementedError

    def forward(self, flux_in, params):
        if params.dim() == 1:
            params = params.unsqueeze(-1)
        batch_size, time_steps = flux_in.shape
        raw_weights = self.get_weights(params)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        flux_out = F.conv1d(input=padded_x, weight=flipped_weights, groups=batch_size)
        return flux_out.view(batch_size, time_steps)

class DplHalf1(DplUHBase):
    """ GR4J UH1 (Half Bell Curve) """
    def get_weights(self, params):
        d_base = params
        d_base = torch.clamp(d_base, min=1e-3)
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.5)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        return weights

class DplFull2(DplUHBase):
    """ GR4J UH2 (Full Bell Curve) """
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)
        ratio = self.t_idx.to(d_base.device) / d_base.unsqueeze(-1)
        s_part1 = 0.5 * ratio.pow(2.5)
        term_b = torch.clamp(2.0 - ratio, min=0.0)
        s_part2 = 1.0 - 0.5 * term_b.pow(2.5)
        s_curve = torch.where(ratio <= 1.0, s_part1, s_part2)
        s_curve = torch.clamp(s_curve, max=1.0)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        return weights

# ==========================================
# 2. 测试与绘图脚本
# ==========================================

def test_uh_shapes():
    # 设置测试参数
    max_lag = 30
    n_steps = 40
    batch_size = 3
    
    # 模拟 3 组不同的参数 x4 (汇流时间)
    # Case 1: x4 = 5.5 (正常)
    # Case 2: x4 = 12.0 (较长)
    # Case 3: x4 = 2.0 (极短)
    x4_params = torch.tensor([5.5, 12.0, 2.0]).view(batch_size, 1)
    
    # 构造脉冲输入 (在 t=0 时刻有 1mm 降雨，其余为 0)
    # 这样模型的输出就是单位线本身
    pulse_in = torch.zeros(batch_size, n_steps)
    pulse_in[:, 0] = 1.0  # Impulse at t=0
    
    # 初始化模型
    model_uh1 = DplHalf1(max_lag=max_lag)
    model_uh2 = DplFull2(max_lag=max_lag) # UH2 的总长是 2*x4
    
    # 运行模型
    with torch.no_grad():
        out_uh1 = model_uh1(pulse_in, x4_params)
        out_uh2 = model_uh2(pulse_in, x4_params)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: UH1 (Half Bell)
    ax = axes[0]
    time_axis = np.arange(n_steps)
    for i in range(batch_size):
        x4_val = x4_params[i].item()
        ax.plot(time_axis, out_uh1[i].numpy(), marker='o', label=f'x4 = {x4_val}')
        # 画出 x4 的理论截止线
        ax.axvline(x=x4_val, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title("[Result] DplHalf1 (GR4J UH1)")
    ax.set_xlabel("Time Step (days)")
    ax.set_ylabel("Weight / Flow Response")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: UH2 (Full Bell)
    ax = axes[1]
    for i in range(batch_size):
        x4_val = x4_params[i].item()
        ax.plot(time_axis, out_uh2[i].numpy(), marker='o', label=f'x4 = {x4_val}')
        # 画出 x4 (峰值) 和 2*x4 (结束) 的理论线
        ax.axvline(x=x4_val, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=2*x4_val, color='red', linestyle='--', alpha=0.5)
        
    ax.set_title("[Result] DplFull2 (GR4J UH2)")
    ax.set_xlabel("Time Step (days)")
    ax.set_ylabel("Weight / Flow Response")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_uh_shapes()