import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. 基础环境构建 (确保独立可运行)
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
        
        # 翻转权重 (关键)
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        
        batch_size, time_steps = flux_in.shape
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        flux_out = F.conv1d(padded_x, flipped_weights, groups=batch_size)
        
        flux_out = flux_out.view(batch_size, time_steps)
        if return_weights: return flux_out, norm_weights.view(batch_size, -1)
        return flux_out
    
    def get_weights(self, params): raise NotImplementedError

# 将刚才定义的 DplTri3 放入
class DplTri3(DplUHBase):
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)
        ratio = self.t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.0) # Power 2.0
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        return s_curve - s_curve_padded[..., :-1]

# ==============================================================================
# 1. 验证逻辑：Python复刻MATLAB循环算法 (Ground Truth)
# ==============================================================================
def matlab_uh3_simulator(d_base):
    """
    完全照搬 MATLAB uh_3_half 的 Loop 逻辑来生成真值
    """
    delay = d_base
    # tt = 1:ceil(delay)
    tt = np.arange(1, int(np.ceil(delay)) + 1)
    
    # ff = 1/(0.5*delay^2)
    ff = 1.0 / (0.5 * delay**2)
    
    UH = np.zeros(len(tt))
    
    for i, t in enumerate(tt): # t is 1-based time
        if t <= delay:
            # UH(t) = ff.*(0.5*t^2 - 0.5*(t-1)^2);
            val = ff * (0.5*(t**2) - 0.5*((t-1)**2))
        else:
            # UH(t) = ff.*(0.5*delay^2 - 0.5*(t-1)^2);
            val = ff * (0.5*(delay**2) - 0.5*((t-1)**2))
        UH[i] = val
        
    return UH

# ==============================================================================
# 2. 运行测试
# ==============================================================================
def run_test():
    print("--- 🚀 测试 DplTri3 (Half Triangle / Linear) ---")
    
    # 设定测试参数
    d_base = 3.8
    max_lag = 10
    
    # --- A. 计算 MATLAB 真值 ---
    matlab_vals = matlab_uh3_simulator(d_base)
    print(f"\n[基准值] 基于 MATLAB 逻辑计算 (d_base={d_base}):")
    print(np.round(matlab_vals, 4))
    
    # --- B. 计算 PyTorch 值 ---
    model = DplTri3(max_lag=max_lag)
    params = torch.tensor([[d_base]])
    dummy = torch.zeros(1, 10)
    
    with torch.no_grad():
        _, pt_weights_tensor = model(dummy, params, return_weights=True)
    
    pt_vals = pt_weights_tensor[0].numpy()[:len(matlab_vals)]
    
    # --- C. 逐项对比 ---
    print(f"\n[数值对账] PyTorch vs MATLAB Logic:")
    print(f"{'Time':<5} | {'PyTorch':<10} | {'MATLAB Sim':<10} | {'Diff':<10} | {'Check'}")
    print("-" * 55)
    
    all_pass = True
    for t in range(len(matlab_vals)):
        p = pt_vals[t]
        m = matlab_vals[t]
        diff = abs(p - m)
        status = "✅" if diff < 1e-6 else "❌"
        if diff >= 1e-6: all_pass = False
        print(f"{t+1:<5} | {p:.6f}   | {m:.6f}   | {diff:.2e}   | {status}")
        
    if all_pass:
        print("\n>> 结果: ✅ 验证通过！PyTorch S-Curve公式法与 MATLAB 循环法完全一致。")
    else:
        print("\n>> 结果: ❌ 验证失败，请检查公式。")

    # --- D. 绘图：形状检查 ---
    # 使用一个大一点的 d_base 方便观察"直角三角形"形状
    d_viz = 8.5 
    params_viz = torch.tensor([[d_viz]])
    with torch.no_grad():
        _, w_viz = model(dummy, params_viz, return_weights=True)
    w_viz = w_viz[0].numpy()
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, max_lag+1), w_viz, color='purple', alpha=0.6, label='PyTorch Weights')
    plt.plot(range(1, max_lag+1), w_viz, 'o-', color='purple')
    
    # 画理论线性参考线 (y = kx)
    # PDF斜率 k = 2 / d^2
    k = 2 / (d_viz**2)
    x_ref = np.linspace(0, d_viz, 100)
    y_ref = k * x_ref
    plt.plot(x_ref, y_ref, 'k--', label='Theoretical Linear PDF', alpha=0.5)
    
    plt.title(f'[DplTri3] Half Triangle Shape Check (d_base={d_viz})')
    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()