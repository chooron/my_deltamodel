import torch
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
import torch.nn as nn

# --- 保持 DplUHBase 和 DplGamma6 不变 ---
# (为了节省篇幅，这里假设类定义已经存在，和上一条回复一样)
class DplUHBase(nn.Module):
    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        self.register_buffer('t_idx', torch.arange(1, self.max_lag + 1).float().view(1, 1, -1))
    
    def forward(self, flux_in, params, return_weights=False):
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

class DplGamma6(DplUHBase):
    def get_weights(self, params):
        n = torch.clamp(params[:, 0:1], min=0.1, max=20.0)
        k = torch.clamp(params[:, 1:2], min=1e-3)
        x_val = self.t_idx / k.unsqueeze(-1)
        s_curve = torch.special.gammainc(n.unsqueeze(-1), x_val)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        return s_curve - s_curve_padded[..., :-1]

# ==============================================================================
# 🛠️ 修正后的测试脚本
# ==============================================================================
def test_gamma_uh_fixed():
    print("--- 🚀 测试 DplGamma6 (修正归一化对比逻辑) ---")
    
    params_np = np.array([
        [1.0, 3.79999995], # 使用你报错里的精确浮点数
        [2.5, 1.5]
    ], dtype=np.float32)
    
    max_lag = 15
    
    # --- PyTorch 计算 ---
    model = DplGamma6(max_lag=max_lag)
    params_pt = torch.from_numpy(params_np)
    dummy = torch.zeros(2, 10)
    with torch.no_grad():
        _, pt_weights = model(dummy, params_pt, return_weights=True)
    pt_weights = pt_weights.numpy()

    # --- SciPy 计算 (并手动执行归一化!) ---
    for i in range(2):
        n_val, k_val = params_np[i]
        
        # 1. 计算原始概率
        t_steps = np.arange(1, max_lag + 1)
        cdf_t = stats.gamma.cdf(t_steps, a=n_val, scale=k_val)
        cdf_prev = stats.gamma.cdf(t_steps - 1, a=n_val, scale=k_val)
        w_raw = cdf_t - cdf_prev
        
        # 2. [关键步骤] 对 SciPy 结果也进行归一化，模拟 PyTorch 行为
        # 这一步是为了补偿被截断的尾部 (Tail truncation compensation)
        scipy_sum = w_raw.sum()
        w_norm = w_raw / scipy_sum
        
        print(f"\n[Case {i+1}] n={n_val:.2f}, k={k_val:.2f}")
        print(f"尾部丢失质量(Tail Loss): {(1-scipy_sum)*100:.4f}% -> 已通过归一化补偿")
        print(f"{'Time':<5} | {'PyTorch':<10} | {'SciPy(Norm)':<12} | {'Diff':<10} | {'Check'}")
        print("-" * 60)
        
        for t in range(5):
            p = pt_weights[i, t]
            s = w_norm[t] # 使用归一化后的 SciPy 值
            diff = abs(p - s)
            status = "✅" if diff < 1e-6 else "❌"
            print(f"{t+1:<5} | {p:.6f}   | {s:.6f}       | {diff:.2e}   | {status}")

if __name__ == "__main__":
    test_gamma_uh_fixed()