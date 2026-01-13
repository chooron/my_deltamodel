import torch
import torch.nn.functional as F
import numpy as np

# 假设 DplUHBase 已经定义在你的环境中 (包含 torch.flip 修复的版本)
# 这里为了代码独立性，我再次简略声明一下，如果你已有文件可跳过此 Base 定义
class DplUHBase(torch.nn.Module):
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

# ==============================================================================
# 新增: DplTri4 (Full Triangle)
# ==============================================================================
class DplTri4(DplUHBase):
    """
    Unit Hydrograph 4: Full Triangle (Linear Rise & Fall)
    对应 MATLAB: uh_4_full(d_base, delta_t)
    
    物理形状:
    等腰三角形，峰值位于 d_base / 2 处。
    S-Curve 解析解:
    - Phase 1 (t <= d/2): 2 * (t/d)^2
    - Phase 2 (t > d/2):  1 - 2 * (1 - t/d)^2
    """
    
    def get_weights(self, params):
        # 1. 预处理参数
        d_base = torch.clamp(params, min=1e-3)
        
        # 2. 计算归一化时间比率 r = t / d_base
        ratio = self.t_idx / d_base.unsqueeze(-1)
        
        # 3. 计算 S-Curve (分段函数)
        # ----------------------------------------------------
        # Phase 1: 上升段 (ratio <= 0.5)
        # S = 2 * r^2
        s1 = 2.0 * ratio.pow(2)
        
        # Phase 2: 下降段 (ratio > 0.5)
        # S = 1 - 2 * (1 - r)^2
        # 注意: 当 ratio > 1.0 时，(1-r)^2 会增大导致 S 减小，
        # 所以必须 clamp (1-ratio) 最小为 0，确保 ratio > 1 时 S 恒为 1.0
        term2 = torch.clamp(1.0 - ratio, min=0.0)
        s2 = 1.0 - 2.0 * term2.pow(2)
        
        # 组合两段
        s_curve = torch.where(ratio <= 0.5, s1, s2)
        
        # 再次截断 (消除浮点误差，确保不超过1)
        s_curve = torch.clamp(s_curve, max=1.0)
        
        # 4. 差分计算权重 UH(t) = S(t) - S(t-1)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights
    
def test_uh4_weights_only():
    print("--- 🚀 测试 DplTri4 (Full Triangle) 权重 ---")
    
    # 1. 设定参数
    d_base_val = 3.8
    max_lag = 10
    
    # 2. 初始化模型
    model = DplTri4(max_lag=max_lag)
    params = torch.tensor([[d_base_val]])
    dummy_input = torch.zeros(1, 10) # 仅用于占位
    
    # 3. 计算权重
    with torch.no_grad():
        # return_weights=True 会返回 (Batch, Lag) 的归一化权重
        _, w_tensor = model(dummy_input, params, return_weights=True)
    
    weights = w_tensor[0].numpy()
    
    # 4. 对比目标值 (来自 MATLAB 源码注释)
    # UH(1)=0.14, UH(2)=0.41, UH(3)=0.36, UH(4)=0.09
    targets = [0.14, 0.41, 0.36, 0.09]
    
    print(f"\n参数设定: d_base = {d_base_val}")
    print(f"{'Day':<5} | {'PyTorch':<10} | {'MATLAB Ref':<10} | {'Diff':<10} | {'Check'}")
    print("-" * 55)
    
    all_pass = True
    sum_check = 0.0
    
    for t, ref in enumerate(targets):
        val = weights[t]
        sum_check += val
        diff = abs(val - ref)
        
        # MATLAB 注释通常保留2位小数，我们允许 0.01 的显示误差
        status = "✅" if diff < 0.01 else "❌"
        if diff >= 0.01: all_pass = False
        
        print(f"{t+1:<5} | {val:.4f}     | {ref:<10} | {diff:.4f}     | {status}")
        
    # 检查后续是否归零
    tail_sum = weights[len(targets):].sum()
    print(f"\n[残余检查] Day 5-{max_lag} Sum: {tail_sum:.6f} (Should be 0)")
    
    # 检查总和
    total_sum = weights.sum()
    print(f"[守恒检查] Total Sum: {total_sum:.6f} (Should be 1.0)")
    
    if all_pass and abs(total_sum - 1.0) < 1e-5:
        print("\n>> 结果: ✅ DplTri4 权重计算与 MATLAB 标准值一致。")
    else:
        print("\n>> 结果: ❌ 验证失败，请检查逻辑。")

if __name__ == "__main__":
    test_uh4_weights_only()