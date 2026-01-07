import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. 基础环境
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
        
        # Delay 模型同样需要 Flip 才能保证物理因果方向正确
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        
        batch_size, time_steps = flux_in.shape
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        flux_out = F.conv1d(padded_x, flipped_weights, groups=batch_size)
        
        flux_out = flux_out.view(batch_size, time_steps)
        if return_weights: return flux_out, norm_weights.view(batch_size, -1)
        return flux_out
    
    def get_weights(self, params): raise NotImplementedError

# 插入 DplDelay8
class DplDelay8(DplUHBase):
    def get_weights(self, params):
        t_delay = torch.clamp(params, min=0.0)
        center = t_delay + 1.0
        dist = torch.abs(self.t_idx - center.unsqueeze(-1))
        weights = F.relu(1.0 - dist)
        return weights

# ==============================================================================
# 1. 数值验证
# ==============================================================================
def test_delay_numeric():
    print("--- 🚀 测试 DplDelay8 (Pure Delay) ---")
    
    # 目标参数
    d_val = 3.8
    model = DplDelay8(max_lag=10)
    params = torch.tensor([[d_val]])
    dummy = torch.zeros(1, 10)
    
    with torch.no_grad():
        _, w_tensor = model(dummy, params, return_weights=True)
    
    w = w_tensor[0].numpy()
    
    # 目标值 (MATLAB 注释)
    # Lag 0, 1, 2 -> 0
    # Lag 3 (Index 4) -> 0.2
    # Lag 4 (Index 5) -> 0.8
    targets = [0.0, 0.0, 0.0, 0.20, 0.80, 0.0]
    
    print(f"参数: t_delay = {d_val}")
    print(f"{'Idx':<5} | {'Lag':<5} | {'PyTorch':<10} | {'Target':<10} | {'Diff':<10} | {'Check'}")
    print("-" * 65)
    
    for i, ref in enumerate(targets):
        lag = i # Index 1 corresponds to Lag 0
        val = w[i]
        diff = abs(val - ref)
        status = "✅" if diff < 1e-6 else "❌"
        print(f"{i+1:<5} | {lag:<5} | {val:.6f}   | {ref:.6f}   | {diff:.2e}   | {status}")
        
    # 计算加权平均滞后时间 (Center of Mass)
    # Expected: 3.8
    # Calculated: sum(Lag * Weight)
    lags = np.arange(len(w))
    weighted_lag = np.sum(w * lags)
    print(f"\n[重心检查] Weighted Lag: {weighted_lag:.6f} (Target: {d_val})")
    
    if abs(weighted_lag - d_val) < 1e-5:
        print(">> 结果: ✅ 滞后重心精确匹配。")
    else:
        print(">> 结果: ❌ 重心偏移。")

# ==============================================================================
# 2. 脉冲响应可视化 (移动效果)
# ==============================================================================
def test_delay_visual():
    print("\n--- 📊 生成位移验证图 ---")
    
    # 生成一个位于 t=5 的脉冲
    T = 20
    inflow = torch.zeros(1, T)
    inflow[0, 5] = 1.0 # t=5 (Day 6)
    
    # 设置延迟 3.5 天
    delay_val = 3.5
    model = DplDelay8(max_lag=10)
    params = torch.tensor([[delay_val]])
    
    with torch.no_grad():
        outflow = model(inflow, params)
    
    in_np = inflow[0].numpy()
    out_np = outflow[0].numpy()
    
    # 理论峰值位置: 5 + 3.5 = 8.5
    # 应该在 t=8 (Day 9) 和 t=9 (Day 10) 各有一半
    
    plt.figure(figsize=(10, 4))
    plt.stem(range(T), in_np, linefmt='b-', markerfmt='bo', basefmt=' ', label='Input (t=5)')
    plt.stem(range(T), out_np, linefmt='r-', markerfmt='ro', basefmt=' ', label=f'Output (Delay={delay_val})')
    
    plt.title(f'Pure Delay Unit Hydrograph (Shift = {delay_val})')
    plt.xlabel('Time Step')
    plt.ylabel('Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(T))
    plt.show()

if __name__ == "__main__":
    test_delay_numeric()
    test_delay_visual()