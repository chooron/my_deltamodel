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
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        batch_size, time_steps = flux_in.shape
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        flux_out = F.conv1d(padded_x, flipped_weights, groups=batch_size)
        flux_out = flux_out.view(batch_size, time_steps)
        if return_weights: return flux_out, norm_weights.view(batch_size, -1)
        return flux_out
    
    def get_weights(self, params): raise NotImplementedError

# 插入 DplUniform7
class DplUniform7(DplUHBase):
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)
        ratio = self.t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        return s_curve - s_curve_padded[..., :-1]

# ==============================================================================
# 1. 数值验证
# ==============================================================================
def test_uniform_numeric():
    print("--- 🚀 测试 DplUniform7 (Uniform) ---")
    
    d_val = 3.8
    max_lag = 10
    
    model = DplUniform7(max_lag=max_lag)
    params = torch.tensor([[d_val]])
    dummy = torch.zeros(1, 10)
    
    with torch.no_grad():
        _, w_tensor = model(dummy, params, return_weights=True)
    w = w_tensor[0].numpy()
    
    # 理论计算
    # 前3步满载: 1/3.8
    full_step = 1.0 / 3.8
    # 第4步剩余: 1 - 3 * full_step
    remainder = 1.0 - 3.0 * full_step
    
    targets = [full_step, full_step, full_step, remainder, 0.0]
    
    print(f"参数: d_base = {d_val}")
    print(f"{'Day':<5} | {'PyTorch':<10} | {'Math(True)':<10} | {'Diff':<10} | {'Check'}")
    print("-" * 60)
    
    for t, ref in enumerate(targets):
        val = w[t]
        diff = abs(val - ref)
        status = "✅" if diff < 1e-6 else "❌"
        # 只要是很小的数都算0
        if ref == 0.0 and val < 1e-6: status = "✅"
            
        print(f"{t+1:<5} | {val:.6f}   | {ref:.6f}   | {diff:.2e}   | {status}")

    # 形状检查: 应该是平顶的
    if abs(w[0] - w[1]) < 1e-6 and abs(w[1] - w[2]) < 1e-6:
        print("\n>> 结果: ✅ 形状正确 (前几步权重相等，呈均匀分布)。")
    else:
        print("\n>> 结果: ❌ 形状错误。")

# ==============================================================================
# 2. 可视化形状
# ==============================================================================
def test_uniform_visual():
    print("\n--- 📊 形状可视化 ---")
    d_base = 5.5 # 5天半
    model = DplUniform7(max_lag=10)
    params = torch.tensor([[d_base]])
    
    with torch.no_grad():
        _, w = model(torch.zeros(1,10), params, return_weights=True)
    w = w[0].numpy()
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, 11), w, color='teal', alpha=0.6, edgecolor='black', label=f'd_base={d_base}')
    
    # 画一条理论线
    plt.axhline(y=1.0/d_base, color='red', linestyle='--', label=f'Height = 1/{d_base}')
    
    plt.title('Uniform Unit Hydrograph')
    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    test_uniform_numeric()
    test_uniform_visual()