import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. 核心类定义 (已包含 torch.flip 修复)
# ==============================================================================

class DplUHBase(nn.Module):
    """
    可微单位线基类 (Grouped Conv1d + Flip修复)
    """
    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        # 注册时间索引 [1, 2, ..., max_lag]
        self.register_buffer(
            't_idx', 
            torch.arange(1, self.max_lag + 1, dtype=torch.float32).view(1, 1, -1)
        )

    def get_weights(self, params):
        raise NotImplementedError

    def forward(self, flux_in, params):
        # 维度适配
        if params.dim() == 1: params = params.unsqueeze(-1)
        batch_size, time_steps = flux_in.shape
        
        # 1. 生成并归一化权重
        raw_weights = self.get_weights(params)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)
        
        # 2. [关键修复] 翻转权重以匹配物理卷积定义
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        
        # 3. 准备卷积输入 (Causal Padding)
        x = flux_in.view(1, batch_size, time_steps)
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        
        # 4. 执行卷积
        flux_out = F.conv1d(
            input=padded_x, 
            weight=flipped_weights, 
            groups=batch_size
        )
        
        return flux_out.view(batch_size, time_steps)

class DplHalf1(DplUHBase):
    """ GR4J UH1: Half Bell Curve """
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)
        ratio = self.t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.5)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        return s_curve - s_curve_padded[..., :-1]

# ==============================================================================
# 2. 数据生成与绘图逻辑
# ==============================================================================

def generate_gaussian_runoff(length, center, sigma, magnitude):
    """ 生成一个高斯形状的单峰径流过程 """
    t = torch.arange(length, dtype=torch.float32)
    val = magnitude * torch.exp(-((t - center)**2) / (2 * sigma**2))
    return val.unsqueeze(0) # (Batch=1, Time)

def run_visual_test():
    print("正在生成单位线路由对比图...")
    
    # --- A. 设置参数 ---
    T = 60              # 总时长
    MAX_LAG = 20        # 单位线最大长度
    D_BASE = 6.5        # 汇流时间参数 (越大滞后越明显)
    
    # --- B. 生成数据 ---
    # 在第 10 天生成一个尖锐的洪峰
    inflow = generate_gaussian_runoff(length=T, center=10, sigma=2.0, magnitude=100.0)
    
    # --- C. 运行模型 ---
    model = DplHalf1(max_lag=MAX_LAG)
    params = torch.tensor([[D_BASE]])
    
    with torch.no_grad():
        outflow = model(inflow, params)
    
    # --- D. 绘图 ---
    in_data = inflow[0].numpy()
    out_data = outflow[0].numpy()
    
    # 计算关键指标用于标注
    peak_in_idx = np.argmax(in_data)
    peak_in_val = np.max(in_data)
    
    peak_out_idx = np.argmax(out_data)
    peak_out_val = np.max(out_data)
    
    lag_days = peak_out_idx - peak_in_idx
    attenuation = (peak_in_val - peak_out_val) / peak_in_val * 100
    
    plt.figure(figsize=(12, 6))
    
    # 1. 画输入径流 (Inflow) - 虚线+填充
    plt.plot(in_data, color='#1f77b4', linestyle='--', linewidth=2, label='Inflow (Original Runoff)')
    plt.fill_between(range(T), in_data, color='#1f77b4', alpha=0.1)
    
    # 2. 画输出径流 (Outflow) - 实线+填充
    plt.plot(out_data, color='#d62728', linewidth=3, label=f'Outflow (Routed, d_base={D_BASE})')
    plt.fill_between(range(T), out_data, color='#d62728', alpha=0.1)
    
    # 3. 标注峰值点
    plt.scatter(peak_in_idx, peak_in_val, color='#1f77b4', s=100, zorder=5)
    plt.scatter(peak_out_idx, peak_out_val, color='#d62728', s=100, zorder=5)
    
    # 4. 标注滞后 (Lag Arrow)
    plt.annotate(
        '', xy=(peak_out_idx, peak_out_val), xytext=(peak_in_idx, peak_out_val),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
    )
    plt.text((peak_in_idx + peak_out_idx)/2, peak_out_val + 2, 
             f'Lag: {lag_days} steps', ha='center', fontweight='bold')

    # 5. 标注削峰 (Attenuation)
    plt.text(peak_out_idx + 2, peak_out_val, 
             f'Peak: {peak_out_val:.1f}\n(-{attenuation:.1f}%)', 
             color='#d62728', va='center')

    plt.title(f'Unit Hydrograph Effect: Lag & Attenuation (d_base={D_BASE})', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Flow Magnitude', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    # 显示图像
    plt.savefig("temp.png")
    print("绘图完成。")

if __name__ == "__main__":
    run_visual_test()