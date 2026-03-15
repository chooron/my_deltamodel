"""
测试各候选公式在相同输入条件下的输出量级是否一致。

对每个水文过程，用相同的输入 (典型值 + 边界值) 调用所有候选公式，
打印输出统计 (min, max, mean, std)，并标记量级差异过大的情况。

用法:
    python project/blend_formula/tests/test_formula_magnitude.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# 导入所有候选公式
from project.blend_formula.models.diff_blend_v2 import (
    rainsnow_hbv, rainsnow_dingman, rainsnow_threshold,
    snobal_simple, snobal_hbv, snobal_hmets,
    inf_hmets, inf_vic_arno, inf_hbv,
    soilevap_all, soilevap_linear, soilevap_vic,
    quick_linear, quick_vic, quick_topmodel,
    base_linear, base_power,
)

torch.manual_seed(42)
B = 200  # 样本数

# ===================================================================
# 工具函数
# ===================================================================

def stats(name: str, t: torch.Tensor) -> dict:
    """返回 tensor 的统计摘要"""
    t = t.detach().float()
    return {
        "name": name,
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "median": t.median().item(),
    }

def print_table(title: str, rows: list[dict]):
    """打印对比表格"""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"  {'公式':<25s} {'min':>9s} {'max':>9s} {'mean':>9s} {'std':>9s} {'median':>9s}")
    print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for r in rows:
        print(f"  {r['name']:<25s} {r['min']:9.4f} {r['max']:9.4f} "
              f"{r['mean']:9.4f} {r['std']:9.4f} {r['median']:9.4f}")
    # 量级差异检查
    means = [r["mean"] for r in rows]
    if max(means) > 0:
        ratio = max(means) / (min(means) + 1e-12)
        if ratio > 5:
            print(f"  ⚠️  量级差异警告: max_mean/min_mean = {ratio:.1f}x")
        else:
            print(f"  ✓  量级一致: max_mean/min_mean = {ratio:.1f}x")


def sample_uniform(lo, hi):
    return torch.rand(B) * (hi - lo) + lo


# ===================================================================
# 参数范围 (来自 PARAM_BOUNDS)
# ===================================================================
BOUNDS = {
    "x1":  [0.0, 1.0],    "x2":  [0.3, 3.0],    "x3":  [0.5, 3.0],
    "x4":  [-5.0, -1.0],  "x5":  [0.001, 0.3],   "x6":  [0.5, 2.0],
    "x7":  [1.0, 5.0],    "x8":  [0.0, 3.0],     "x9":  [0.0, 0.05],
    "x10": [0.0, 0.45],   "x11": [-5.0, -2.0],   "x12": [0.5, 2.0],
    "x13": [0.0, 0.1],    "x14": [0.01, 0.3],    "x15": [0.005, 0.1],
    "x16": [-5.0, 2.0],   "x17": [0.3, 1.0],     "x18": [0.0, 5.0],
    "x19": [0.0, 0.4],    "x24": [1.5, 3.0],     "x25": [0.0, 5.0],
    "x26": [-1.0, 1.0],   "x27": [0.01, 0.2],    "x28": [0.00001, 0.02],
    "x31": [-3.0, 3.0],   "x32": [0.5, 4.0],     "x33": [0.8, 1.2],
    "x34": [0.8, 1.2],    "x35": [0.0, 0.02],    "x36": [0.3, 3.0],
}

def p(name):
    lo, hi = BOUNDS[name]
    return sample_uniform(lo, hi)

# ===================================================================
# 输入场景
# ===================================================================
SCENARIOS = OrderedDict({
    "典型降雨日": {"P": 10.0, "T": 10.0, "PET": 3.0},
    "大暴雨":     {"P": 80.0, "T": 15.0, "PET": 5.0},
    "小雨":       {"P": 1.0,  "T": 8.0,  "PET": 2.0},
    "冬季降雪":   {"P": 15.0, "T": -5.0, "PET": 0.5},
    "融雪期":     {"P": 5.0,  "T": 3.0,  "PET": 2.0},
    "干旱日":     {"P": 0.0,  "T": 30.0, "PET": 8.0},
})


# ===================================================================
# 测试各过程
# ===================================================================

def test_rainsnow(P, T):
    """雨雪分割: 3个公式"""
    P_t = torch.full((B,), P)
    T_t = torch.full((B,), T)
    tt = p("x31"); tti = p("x32")

    sf1, rf1 = rainsnow_hbv(P_t, T_t, tt, tti)
    sf2, rf2 = rainsnow_dingman(P_t, T_t, tt)
    sf3, rf3 = rainsnow_threshold(P_t, T_t, tt)

    snow_rows = [stats("hbv_snow", sf1), stats("dingman_snow", sf2), stats("threshold_snow", sf3)]
    rain_rows = [stats("hbv_rain", rf1), stats("dingman_rain", rf2), stats("threshold_rain", rf3)]
    return snow_rows, rain_rows


def test_infiltration(P_eff_val, S_ratio=0.5):
    """入渗: 3个公式"""
    Smax = sample_uniform(50, 500)
    S = Smax * S_ratio
    P_eff = torch.full((B,), P_eff_val)

    i1 = inf_hmets(P_eff, S, Smax, p("x1"))
    i2 = inf_vic_arno(P_eff, S, Smax, p("x2"))
    i3 = inf_hbv(P_eff, S, Smax, p("x3"))
    return [stats("inf_hmets", i1), stats("inf_vic_arno", i2), stats("inf_hbv", i3)]


def test_evaporation(PET_val, S_ratio=0.5):
    """蒸发: 3个公式"""
    Smax = sample_uniform(50, 500)
    S = Smax * S_ratio
    PET = torch.full((B,), PET_val)
    c = p("x8")
    fc = sample_uniform(0.3, 0.8)

    e1 = soilevap_all(PET, c, S)
    e2 = soilevap_linear(PET, c, S, fc * Smax)
    e3 = soilevap_vic(PET, c, S, Smax, p("x36"))
    return [stats("evap_all", e1), stats("evap_linear", e2), stats("evap_vic", e3)]

def test_quickflow(S_ratio=0.5):
    """快速流: 3个公式"""
    Smax = sample_uniform(50, 500)
    S = Smax * S_ratio
    k_quick = 10.0 ** p("x4")  # log_k_quick -> k_quick (与模型一致用 10**)

    q1 = quick_linear(S, k_quick)
    q2 = quick_vic(S, Smax, p("x5"), p("x6"))
    q3 = quick_topmodel(S, Smax, p("x5"), p("x6"), p("x7"))
    return [stats("quick_linear", q1), stats("quick_vic", q2), stats("quick_topmodel", q3)]


def test_baseflow(S_ratio=0.3):
    """基流: 2个公式"""
    Smax = sample_uniform(50, 500)
    S = Smax * S_ratio
    k_base = 10.0 ** p("x11")  # log_k_base -> k_base (与模型一致用 10**)

    b1 = base_linear(S, k_base)
    b2 = base_power(S, k_base, p("x12"), Smax)
    return [stats("base_linear", b1), stats("base_power", b2)]


def test_snowbal(snowfall_val, rainfall_val, T_val, S_snow_val=20.0):
    """雪平衡: 3个公式"""
    snowfall = torch.full((B,), snowfall_val)
    rainfall = torch.full((B,), rainfall_val)
    T = torch.full((B,), T_val)
    S_snow = torch.full((B,), S_snow_val)
    S_liq = torch.full((B,), 2.0)
    S_cum = torch.full((B,), 10.0)

    o1, *_ = snobal_simple(snowfall, rainfall, T, S_snow, S_cum,
                            p("x24"), p("x26"))
    o2, *_ = snobal_hbv(snowfall, rainfall, T, S_snow, S_liq, S_cum,
                         p("x24") + p("x25"), p("x26"), p("x18"),
                         p("x16"), p("x19"))
    o3, *_ = snobal_hmets(snowfall, rainfall, T, S_snow, S_liq, S_cum,
                           p("x24"), p("x24") + p("x25"), p("x26"),
                           p("x18"), p("x16"), p("x17"),
                           p("x13"), p("x13") + p("x14"),
                           p("x15"), p("x27"))
    return [stats("snobal_simple", o1), stats("snobal_hbv", o2), stats("snobal_hmets", o3)]


# ===================================================================
# 主函数
# ===================================================================

def main():
    print("候选公式输出量级对比测试")
    print(f"样本数 B={B}, 参数从 PARAM_BOUNDS 均匀采样\n")

    for scenario_name, vals in SCENARIOS.items():
        P, T, PET = vals["P"], vals["T"], vals["PET"]
        print(f"\n{'#'*72}")
        print(f"# 场景: {scenario_name}  (P={P}, T={T}, PET={PET})")
        print(f"{'#'*72}")

        # 1. 雨雪分割
        snow_rows, rain_rows = test_rainsnow(P, T)
        print_table(f"雨雪分割 - 降雪量 (P={P}, T={T})", snow_rows)
        print_table(f"雨雪分割 - 降雨量 (P={P}, T={T})", rain_rows)

        # 2. 雪平衡 (仅在有雪的场景测试)
        if T < 5:
            sf_mean = np.mean([r["mean"] for r in snow_rows])
            rf_mean = np.mean([r["mean"] for r in rain_rows])
            sn_rows = test_snowbal(max(sf_mean, 0.1), max(rf_mean, 0.1), T)
            print_table(f"雪平衡 - 出流 (T={T})", sn_rows)

        # 3. 入渗
        P_eff = P * 0.8  # 假设80%有效降水
        for sr in [0.2, 0.5, 0.8]:
            inf_rows = test_infiltration(P_eff, S_ratio=sr)
            print_table(f"入渗 (P_eff={P_eff:.1f}, S/Smax={sr})", inf_rows)

        # 4. 蒸发
        for sr in [0.2, 0.5, 0.8]:
            ev_rows = test_evaporation(PET, S_ratio=sr)
            print_table(f"蒸发 (PET={PET}, S/Smax={sr})", ev_rows)

        # 5. 快速流
        for sr in [0.2, 0.5, 0.8]:
            qf_rows = test_quickflow(S_ratio=sr)
            print_table(f"快速流 (S/Smax={sr})", qf_rows)

        # 6. 基流
        for sr in [0.1, 0.3, 0.5]:
            bf_rows = test_baseflow(S_ratio=sr)
            print_table(f"基流 (S/Smax={sr})", bf_rows)


if __name__ == "__main__":
    main()
