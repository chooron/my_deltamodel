import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. 基础环境 (DplUHBase 包含 Flip 修复)
# ==============================================================================
class DplUHBase(nn.Module):
    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        self.register_buffer('t_idx', torch.arange(1, self.max_lag + 1).float().view(1, 1, -1))
    
    def forward(self, flux_in, params, return_weights=False):
        if params.dim() == 1: params = params.unsqueeze(-1)
        raw_weights = self.get_weights(params)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        
        batch_size, time_steps = flux_in.shape
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        flux_out = F.conv1d(padded_x, flipped_weights, groups=batch_size)
        
        flux_out = flux_out.view(batch_size, time_steps)
        if return_weights: return flux_out, norm_weights.view(batch_size, -1)
        return flux_out
    
    def get_weights(self, params): raise NotImplementedError

# 插入刚才定义的 DplExp5
class DplExp5(DplUHBase):
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)
        scale_factor = 7.0 / d_base.unsqueeze(-1)
        scaled_t = self.t_idx * scale_factor
        clamped_t = torch.clamp(scaled_t, max=7.0)
        s_curve = 1.0 - torch.exp(-clamped_t)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        return weights

# ==============================================================================
# 1. 权重数值验证
# ==============================================================================
def test_weights_numeric():
    print("--- 🚀 测试 DplExp5 (Exponential Decay) 数值精度 ---")
    
    # 参数设定
    d_base_val = 3.8
    max_lag = 10
    
    # 实例化
    model = DplExp5(max_lag=max_lag)
    params = torch.tensor([[d_base_val]])
    dummy = torch.zeros(1, 10)
    
    # 计算
    with torch.no_grad():
        _, w_tensor = model(dummy, params, return_weights=True)
    weights = w_tensor[0].numpy()
    
    # 目标值 (MATLAB 注释)
    targets = [0.841, 0.133, 0.021, 0.004]
    
    print(f"参数: d_base = {d_base_val}")
    print(f"{'Day':<5} | {'PyTorch':<10} | {'Target':<10} | {'Diff':<10} | {'Status'}")
    print("-" * 55)
    
    for t, ref in enumerate(targets):
        val = weights[t]
        diff = abs(val - ref)
        # 允许 0.005 的误差 (因为 MATLAB 代码对尾部做了特殊归一化处理，而我们是全局归一化)
        status = "✅" if diff < 0.005 else "❌"
        print(f"{t+1:<5} | {val:.4f}     | {ref:<10} | {diff:.4f}     | {status}")
        
    print("-" * 55)

# ==============================================================================
# 2. 形状可视化验证
# ==============================================================================
def test_shape_visual():
    print("\n--- 📊 生成形状验证图 ---")
    d_base = 5.0
    model = DplExp5(max_lag=15)
    params = torch.tensor([[d_base]])
    dummy = torch.zeros(1, 10)
    
    with torch.no_grad():
        _, w = model(dummy, params, return_weights=True)
    w = w[0].numpy()
    
    plt.figure(figsize=(10, 5))
    x = range(1, 16)
    
    # 绘制柱状图
    plt.bar(x, w, color='orange', alpha=0.6, label='PyTorch Weights', edgecolor='black')
    
    # 绘制理论衰减曲线用于对比
    # y = exp(-x), x mapped from [0, d_base] to [0, 7]
    t_smooth = np.linspace(0, 15, 100)
    # 理论上 continuous pdf 应该是 exp(-t * 7/d) * (7/d)
    # 但由于 unit hydrograph 是积分形式，我们只看趋势是否一致
    
    plt.title(f'Exponential Decay Unit Hydrograph (d_base={d_base})')
    plt.xlabel('Time Step')
    plt.ylabel('Weight Fraction')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig("tmp.png")
    print("绘图完成。应看到典型的指数下降趋势，且在 t=5 之后截断为 0。")

if __name__ == "__main__":
    test_weights_numeric()
    test_shape_visual()