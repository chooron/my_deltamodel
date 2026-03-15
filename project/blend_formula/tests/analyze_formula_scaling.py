"""
分析公式量级差异的根本原因，并提出参数范围调整建议。

基于 test_formula_magnitude.py 的测试结果，针对性分析问题公式。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from project.blend_formula.models.diff_blend_v2 import (
    quick_linear, quick_vic, quick_topmodel,
    base_linear, base_power,
)

torch.manual_seed(42)

# ===================================================================
# 问题 1: 快速流量级差异 (49x at S/Smax=0.2, 13x at S/Smax=0.5)
# ===================================================================

def analyze_quickflow():
    """分析快速流公式的量级差异"""
    print("\n" + "="*72)
    print("快速流量级差异分析")
    print("="*72)

    # 测试不同 S 值下的输出
    Smax = 200.0
    S_ratios = [0.2, 0.5, 0.8]

    # 参数范围 (新公式: S * (1-exp(-rate)))
    k_quick_range = 10.0 ** torch.linspace(-5, -1, 100)     # x4: [-5, -1]
    q_rate_range = torch.linspace(0.001, 0.3, 100)           # x5: [0.001, 0.3] q_rate
    n_range = torch.linspace(0.5, 2.0, 100)                  # x6: [0.5, 2.0]
    lam_range = torch.linspace(1.0, 5.0, 100)                # x7: [1, 5]

    for sr in S_ratios:
        S = Smax * sr
        print(f"\n--- S/Smax = {sr} (S={S:.1f}) ---")

        # 1. quick_linear: S * (1 - exp(-k))
        q1_vals = []
        for k in k_quick_range:
            q1 = quick_linear(torch.tensor([S]), torch.tensor([k]))
            q1_vals.append(q1.item())

        # 2. quick_vic: S * (1 - exp(-q_rate * ratio^n))
        q2_vals = []
        for q_rate in q_rate_range:
            for n in n_range:
                q2 = quick_vic(torch.tensor([S]), torch.tensor([Smax]),
                              torch.tensor([q_rate]), torch.tensor([n]))
                q2_vals.append(q2.item())

        # 3. quick_topmodel: S * (1 - exp(-q_rate * exp(-lam*(1-ratio)^n)))
        q3_vals = []
        for q_rate in q_rate_range:
            for lam in lam_range:
                q3 = quick_topmodel(torch.tensor([S]), torch.tensor([Smax]),
                                   torch.tensor([q_rate]), torch.tensor([n_range[0]]),
                                   torch.tensor([lam]))
                q3_vals.append(q3.item())

        print(f"  quick_linear:   mean={np.mean(q1_vals):.2f}, std={np.std(q1_vals):.2f}, max={np.max(q1_vals):.2f}")
        print(f"  quick_vic:      mean={np.mean(q2_vals):.2f}, std={np.std(q2_vals):.2f}, max={np.max(q2_vals):.2f}")
        print(f"  quick_topmodel: mean={np.mean(q3_vals):.2f}, std={np.std(q3_vals):.2f}, max={np.max(q3_vals):.2f}")
        print(f"  量级比: vic/linear={np.mean(q2_vals)/np.mean(q1_vals):.1f}x, "
              f"topmodel/linear={np.mean(q3_vals)/np.mean(q1_vals):.1f}x")


# ===================================================================
# 问题 2: 基流量级差异 (7-8x)
# ===================================================================

def analyze_baseflow():
    """分析基流公式的量级差异"""
    print("\n" + "="*72)
    print("基流量级差异分析")
    print("="*72)

    Smax = 200.0
    S_ratios = [0.1, 0.3, 0.5]

    k_base_range = 10.0 ** torch.linspace(-5, -2, 100)     # x11: [-5, -2]
    n_range = torch.linspace(0.5, 2.0, 100)                # x12: [0.5, 2.0]

    for sr in S_ratios:
        S = Smax * sr
        print(f"\n--- S/Smax = {sr} (S={S:.1f}) ---")

        # 1. base_linear: S * (1 - exp(-k))
        b1_vals = []
        for k in k_base_range:
            b1 = base_linear(torch.tensor([S]), torch.tensor([k]))
            b1_vals.append(b1.item())

        # 2. base_power: S * (1 - exp(-k * ratio^(n-1)))
        b2_vals = []
        for k in k_base_range:
            for n in n_range:
                b2 = base_power(torch.tensor([S]), torch.tensor([k]),
                               torch.tensor([n]), torch.tensor([Smax]))
                b2_vals.append(b2.item())

        print(f"  base_linear: mean={np.mean(b1_vals):.2f}, std={np.std(b1_vals):.2f}, max={np.max(b1_vals):.2f}")
        print(f"  base_power:  mean={np.mean(b2_vals):.2f}, std={np.std(b2_vals):.2f}, max={np.max(b2_vals):.2f}")
        print(f"  量级比: power/linear={np.mean(b2_vals)/np.mean(b1_vals):.1f}x")


# ===================================================================
# 建议的参数范围调整
# ===================================================================

def suggest_param_adjustments():
    """基于分析结果，建议参数范围调整"""
    print("\n" + "="*72)
    print("参数范围调整建议")
    print("="*72)

    suggestions = {
        "快速流": {
            "问题": "quick_topmodel 输出量级远小于其他两个公式",
            "原因": "topmodel 公式中 exp(-lam*(1-S/Smax)) 在 S/Smax 较小时衰减很快",
            "建议": [
                "1. 缩小 q_max (x5) 范围: [0, 100] -> [0, 20]，减少 vic 公式的极端值",
                "2. 调整 topmodel_lambda (x7): [5, 10] -> [2, 6]，减缓衰减速度",
                "3. 或者在公式中添加归一化系数"
            ]
        },
        "基流": {
            "问题": "base_power 在 n>1 且 S 较大时输出远大于 base_linear",
            "原因": "幂函数 S^n 在 n>1 时增长快，k 的范围 [exp(-5), exp(-2)] 跨度大",
            "建议": [
                "1. 缩小 k_base 范围: log_k_base (x11) [-5, -2] -> [-4, -2.5]",
                "2. 限制 n_base (x12): [0.5, 2.0] -> [0.5, 1.5]",
                "3. 或者在 base_power 中添加缩放因子"
            ]
        },
        "雨雪分割": {
            "问题": "在极端温度下某些公式输出接近 0，导致除零",
            "原因": "这是正常的物理行为（极寒时全是雪），不是 bug",
            "建议": [
                "在计算量级比时，跳过 mean < 1e-6 的情况",
                "或者使用 max(mean, 1e-6) 避免除零"
            ]
        }
    }

    for process, info in suggestions.items():
        print(f"\n【{process}】")
        print(f"  问题: {info['问题']}")
        print(f"  原因: {info['原因']}")
        print(f"  建议:")
        for s in info['建议']:
            print(f"    {s}")


# ===================================================================
# 主函数
# ===================================================================

def main():
    print("公式量级差异根因分析")
    print("基于 test_formula_magnitude.py 的测试结果\n")

    analyze_quickflow()
    analyze_baseflow()
    suggest_param_adjustments()

    print("\n" + "="*72)
    print("总结")
    print("="*72)
    print("主要问题:")
    print("  1. quick_topmodel 的 lambda 参数范围过大，导致在低含水率时输出过小")
    print("  2. base_power 的 k 和 n 参数范围过大，导致输出跨度大")
    print("  3. q_max 参数 [0, 100] 范围过大，vic 公式容易产生极端值")
    print("\n推荐方案:")
    print("  方案A (调整参数范围): 见上述建议，修改 PARAM_BOUNDS")
    print("  方案B (公式归一化): 在每个公式输出后除以其典型量级")
    print("  方案C (混合): 先调整明显不合理的参数范围，再用 balance loss 约束")


if __name__ == "__main__":
    main()
