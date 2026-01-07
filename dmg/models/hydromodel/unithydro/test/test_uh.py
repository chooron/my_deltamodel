import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. 核心类定义 (DplUHBase + DplHalf1)
# ==============================================================================

class DplUHBase(nn.Module):
    """
    可微单位线基类 (修复了卷积方向与滞后问题)
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
        """ 子类必须实现此方法 """
        raise NotImplementedError

    def forward(self, flux_in, params, return_weights=False):
        # 0. 参数维度安全检查 (Batch,) -> (Batch, 1)
        if params.dim() == 1:
            params = params.unsqueeze(-1)
            
        batch_size, time_steps = flux_in.shape
        
        # 1. 生成权重 (Batch, 1, Lag)
        raw_weights = self.get_weights(params)
        
        # 2. 归一化 (Mass Balance)
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        norm_weights = raw_weights / (sum_w + self.epsilon)
        
        # 3. 翻转权重 (CRITICAL FIX)
        # 物理卷积 y[t] = x[t]*w[0] + x[t-1]*w[1]...
        # PyTorch Conv1d 默认是互相关，必须翻转核才能实现因果卷积
        flipped_weights = torch.flip(norm_weights, dims=[-1])
        
        # 4. 准备卷积输入
        # View: (Batch, Time) -> (1, Batch, Time)
        x = flux_in.view(1, batch_size, time_steps)
        
        # 5. 因果填充 (Causal Padding)
        # 只在左侧填充 max_lag - 1，确保 t 时刻只能看到 t 及之前的数据
        padded_x = F.pad(x, (self.max_lag - 1, 0))
        
        # 6. 并行卷积 (Grouped Convolution)
        flux_out = F.conv1d(
            input=padded_x, 
            weight=flipped_weights, # 使用翻转后的权重
            groups=batch_size       # 每个样本独立卷积
        )
        
        # 7. 还原形状
        flux_out = flux_out.view(batch_size, time_steps)
        
        # 返回未翻转的 norm_weights 供检查，符合人类直觉
        if return_weights:
            return flux_out, norm_weights.view(batch_size, -1)
        
        return flux_out

class DplHalf1(DplUHBase):
    """ 
    GR4J UH1 (Half Bell Curve) 
    对应 MATLAB: uh_1_half
    """
    def get_weights(self, params):
        # params: d_base (Batch, 1)
        d_base = params
        
        # 保护机制: d_base 极小值
        d_base = torch.clamp(d_base, min=1e-3)
        
        # 计算 S-Curve
        # ratio: (1,1,L) / (B,1,1) -> (B,1,L)
        ratio = self.t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.5)
        
        # 差分计算 UH
        # UH(t) = SH(t) - SH(t-1)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]
        
        return weights

# ==============================================================================
# 2. 验证脚本 (包含数值对账与滞后检查)
# ==============================================================================

def run_verification():
    print("\n" + "="*60)
    print("🚀 开始 GR4J 单位线 (DplHalf1) 完整性测试")
    print("="*60)

    # 全局设置
    MAX_LAG = 15
    model = DplHalf1(max_lag=MAX_LAG)
    dummy_input = torch.zeros(1, 20) # 仅用于触发 forward，数值不重要

    # ------------------------------------------------------------------
    # 测试案例 A: 数值精度对账 (d_base = 3.8)
    # ------------------------------------------------------------------
    print("\n[测试 A] 数值对账: 对比 d_base=3.8 时的权重分布")
    print("目标值来源: MARRMoT 文档 (UH1=0.04, UH2=0.17, UH3=0.35, UH4=0.45)")
    
    d_base_val = 3.8
    params = torch.tensor([[d_base_val]])
    
    with torch.no_grad():
        _, w_tensor = model(dummy_input, params, return_weights=True)
    
    weights = w_tensor[0].numpy()
    targets = [0.04, 0.17, 0.35, 0.45] # 文档参考值
    
    print("-" * 65)
    print(f"{'Day':<5} | {'PyTorch Value':<15} | {'MATLAB Ref':<12} | {'Diff':<10} | {'Check'}")
    print("-" * 65)
    
    all_pass = True
    for i, ref_val in enumerate(targets):
        pt_val = weights[i]
        diff = abs(pt_val - ref_val)
        # 文档只保留了2位小数，所以允许 0.01 的显示误差
        status = "✅" if diff < 0.01 else "❌" 
        if diff >= 0.01: all_pass = False
        print(f"{i+1:<5} | {pt_val:.6f}        | {ref_val:<12} | {diff:.6f}   | {status}")
    
    print("-" * 65)
    if all_pass:
        print(">> 结果: 数值验证通过！PyTorch 实现与文档描述一致。")
    else:
        print(">> 结果: 数值存在差异，请检查。")

    # ------------------------------------------------------------------
    # 测试案例 B: 滞后修复检查 (d_base = 1.0)
    # ------------------------------------------------------------------
    print("\n[测试 B] 滞后性检查: 当 d_base=1.0 时")
    print("预期: 水应该在第 1 天全部流出 (UH[0] ≈ 1.0)，不应有滞后。")
    
    params_1 = torch.tensor([[1.0]])
    with torch.no_grad():
        _, w_1 = model(dummy_input, params_1, return_weights=True)
        
    w_vec = w_1[0].numpy()
    peak_idx = np.argmax(w_vec)
    peak_val = w_vec[peak_idx]
    
    print(f"权重向量(前5天): {w_vec[:5]}")
    print(f"最大权重位置(Lag): Day {peak_idx + 1}")
    print(f"最大权重值: {peak_val:.6f}")
    
    if peak_idx == 0 and peak_val > 0.99:
        print(">> 结果: ✅ 滞后问题已修复 (Lag=0)。")
    else:
        print(f">> 结果: ❌ 依然存在滞后! (Detected Lag={peak_idx} days)")

    # ------------------------------------------------------------------
    # 测试案例 C: 脉冲响应卷积检查 (Impulse Response)
    # ------------------------------------------------------------------
    print("\n[测试 C] 卷积逻辑检查: 脉冲响应")
    print("输入: [100, 0, 0, ...] (模拟单次瞬时降雨)")
    print("预期: 输出流量应严格等于 100 * Weights")
    
    # 构造脉冲
    impulse_in = torch.zeros(1, 10)
    impulse_in[0, 0] = 100.0
    
    with torch.no_grad():
        # 这里使用 d_base=3.8 进行测试
        q_out = model(impulse_in, params) # params is still 3.8
    
    q_vec = q_out[0].numpy()
    w_vec = weights # 来自测试A的权重
    
    # 检查第2天 (索引1)
    day_idx = 1
    expected_q = 100.0 * w_vec[day_idx]
    actual_q = q_vec[day_idx]
    
    print(f"Day {day_idx+1} Expected Q: {expected_q:.6f}")
    print(f"Day {day_idx+1} Actual Q  : {actual_q:.6f}")
    
    if abs(expected_q - actual_q) < 1e-5:
        print(">> 结果: ✅ 卷积计算正确。")
    else:
        print(">> 结果: ❌ 卷积计算错误 (Flip可能未生效)。")
        
def test_uh_2_full():
    print("--- 测试 GR4J UH2 (Full Curve) ---")
    
    # 1. 设置参数
    d_base_val = 3.8
    # UH2 的影响时间是 2 * d_base = 7.6 天，所以我们需要至少 8 天的窗口
    max_lag = 10 
    
    # 2. 初始化模型
    model = DplFull2(max_lag=max_lag)
    params = torch.tensor([[d_base_val]])
    dummy_in = torch.zeros(1, 10)
    
    # 3. 计算权重
    with torch.no_grad():
        _, w_tensor = model(dummy_in, params, return_weights=True)
    
    weights = w_tensor[0].numpy()
    
    # 4. 打印核对
    # MATLAB文档值
    targets = [0.02, 0.08, 0.18, 0.29, 0.24, 0.14, 0.05, 0.005]
    
    print(f"参数: d_base = {d_base_val}")
    print(f"{'Day':<5} | {'PyTorch':<10} | {'MATLAB Ref':<10} | {'Diff':<10}")
    print("-" * 45)
    
    for t, ref in enumerate(targets):
        val = weights[t]
        diff = abs(val - ref)
        status = "✅" if diff < 0.01 else "❌"
        print(f"{t+1:<5} | {val:.4f}     | {ref:<10} | {diff:.4f} {status}")

if __name__ == "__main__":
    run_verification()