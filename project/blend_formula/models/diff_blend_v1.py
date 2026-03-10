"""
DiffBlendV1 - 可微分公式混合水文模型

基于 Raven Chapter 3 水文过程公式目录，对每个水文过程提供多个可选公式，
通过 Gumbel-Softmax / Softmax / Sparsemax / Entmax 权重进行可微分混合。

过程及选项数:
  - 雨雪分割 (3): RAINSNOW_HBV, RAINSNOW_DINGMAN, RAINSNOW_THRESHOLD
  - 雪平衡   (3): SNOBAL_SIMPLE, SNOBAL_HBV, SNOBAL_HMETS
  - 入渗     (3): INF_HMETS, INF_VIC_ARNO, INF_HBV
  - 蒸发     (3): SOILEVAP_ALL, SOILEVAP_LINEAR, SOILEVAP_VIC
  - 快速流   (3): QUICK_LINEAR, QUICK_VIC, QUICK_TOPMODEL
  - 基流     (2): BASE_LINEAR_ANALYTIC, BASE_POWER_LAW
  - 渗漏     (1): PERC_LINEAR (固定)
  - 汇流     (1): Gamma UH (固定)

共 36 个物理参数 + 过程权重 logits + 2 路由参数。
所有公式独立实现，全部基于 torch 可导。
"""

from __future__ import annotations
from typing import Any, Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import uh_conv, uh_gamma


# ===================================================================
# 权重激活工具
# ===================================================================

def activate_weights(
    logits: torch.Tensor,
    method: str = "softmax",
    tau: float = 1.0,
    training: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """对 logits 施加权重激活，返回归一化权重。"""
    logits = torch.clamp(logits, min=-10.0, max=10.0)
    if method == "gumbel_softmax":
        if training:
            return F.gumbel_softmax(logits, tau=tau, hard=False, dim=dim)
        return F.softmax(logits / tau, dim=dim)
    if method == "sparsemax":
        return _sparsemax(logits, dim=dim)
    if method == "entmax15":
        return _entmax15(logits, dim=dim)
    return F.softmax(logits / tau, dim=dim)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016)"""
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    n = logits.shape[dim]
    k_range = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = n
    k_range = k_range.view(shape)
    cumsum = sorted_logits.cumsum(dim=dim)
    support = (1 + k_range * sorted_logits) > cumsum
    k_z = support.sum(dim=dim, keepdim=True).float()
    tau_z = (cumsum.gather(dim, (k_z - 1).long().clamp(min=0)) - 1) / k_z
    return F.relu(logits - tau_z)


def _entmax15(logits: torch.Tensor, dim: int = -1, n_iter: int = 25) -> torch.Tensor:
    """Entmax 1.5 (Peters et al., 2019) - bisection"""
    lo = logits.min(dim=dim, keepdim=True).values - 1
    hi = logits.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        p = F.relu(logits - mid).pow(2.0)
        s = p.sum(dim=dim, keepdim=True)
        lo = torch.where(s > 1, mid, lo)
        hi = torch.where(s > 1, hi, mid)
    tau = (lo + hi) / 2
    return F.relu(logits - tau).pow(2.0)


# ===================================================================
# 平滑阈值函数 (独立实现，不依赖外部 flux 模块)
# ===================================================================

def _smooth_temperature(T: torch.Tensor, threshold: torch.Tensor, k: float = 5.0) -> torch.Tensor:
    """Sigmoid 平滑温度阈值: ~1 when T < threshold (雪), ~0 when T > threshold (雨)."""
    return torch.sigmoid(k * (threshold - T))


# ===================================================================
# 独立水文过程公式 (全部基于 torch，保证可导)
# 所有函数签名: (参数..., 状态..., 驱动...) -> flux tensor
# ===================================================================

# ---------- 1. 雨雪分割 (Precipitation Partitioning) ----------

def rainsnow_hbv(P, T, tt, tti, eps=1e-6):
    """HBV 线性过渡: snowfall = P * clamp((tt + tti/2 - T) / tti, 0, 1)"""
    snow_frac = torch.clamp((tt + tti / 2 - T) / (tti + eps), 0.0, 1.0)
    return P * snow_frac, P * (1.0 - snow_frac)

def rainsnow_dingman(P, T, ts):
    """Dingman 双侧修正: T << ts → 全雪 (snow_frac→1), T >> ts → 全雨 (snow_frac→0)。
    原单侧 relu 版在 T > ts 时 snow_frac 固定为 0.5 (物理错误), 已修复为对称双侧形式。
    指数 1.3 > 1 保证 diff=0 处梯度为零，无爆炸风险。
    """
    diff_cold = F.relu(ts - T)   # T < ts 时有效 (寒冷侧)
    diff_warm = F.relu(T - ts)   # T > ts 时有效 (温暖侧)
    snow_frac = torch.clamp(
        0.5 * (
            1.0
            + torch.exp(-2.2 * (diff_warm + 1e-6).pow(1.3))
            - torch.exp(-2.2 * (diff_cold + 1e-6).pow(1.3))
        ),
        0.0,
        1.0,
    )
    return P * snow_frac, P * (1.0 - snow_frac)

def rainsnow_threshold(P, T, tt):
    """平滑阈值: 基于 sigmoid 的雨雪分割"""
    snow_frac = _smooth_temperature(T, tt)
    return P * snow_frac, P * (1.0 - snow_frac)


# ---------- 2. 雪平衡 (Snow Balance) ----------

def snobal_simple(snowfall, rainfall, T, S_snow, S_cum, ddf, tt_melt, eps=1e-6):
    """SIMPLE_MELT: 单层雪桶, 无液态水储存"""
    melt_pot = ddf * F.relu(T - tt_melt)
    melt = torch.minimum(S_snow, melt_pot)
    outflow = melt + rainfall
    S_snow_new = S_snow + snowfall - melt
    S_cum_new = torch.where(S_snow > eps, S_cum + melt, torch.zeros_like(S_cum))
    return outflow, S_snow_new, S_cum_new, melt

def snobal_hbv(snowfall, rainfall, T, S_snow, S_liq, S_cum,
               ddf, tt_melt, cfr, tt_refreeze, swi, eps=1e-6):
    """SNOBAL_HBV: 含液态水保持和重冻"""
    melt_pot = ddf * F.relu(T - tt_melt)
    melt = torch.minimum(S_snow, melt_pot)
    refreeze = torch.minimum(S_liq, cfr * F.relu(tt_refreeze - T))
    outflow = F.relu(S_liq + rainfall + melt - swi * S_snow)
    S_snow_new = S_snow + snowfall - melt + refreeze
    # F.relu 防止 swi≈0 且 refreeze>0 时 S_liq 变负，负 S_liq 会使下一步 refreeze 为负并级联产生 NaN
    S_liq_new = F.relu(S_liq + melt + rainfall - refreeze - outflow)
    S_cum_new = torch.where(S_snow > eps, S_cum + melt, torch.zeros_like(S_cum))
    return outflow, S_snow_new, S_liq_new, S_cum_new, melt

def snobal_hmets(snowfall, rainfall, T, S_snow, S_liq, S_cum,
                 ddf_min, ddf_max, tt_melt, kf, tt_refreeze, refreeze_exp,
                 swi_min, swi_max, alpha_swi, dd_agg, eps=1e-6):
    """SNOBAL_HMETS: 变液态水容量"""
    ddf = torch.minimum(ddf_max, ddf_min * (1.0 + dd_agg * S_cum))
    melt_pot = ddf * F.relu(T - tt_melt)
    melt = torch.minimum(S_snow, melt_pot)
    refreeze = torch.minimum(S_liq, kf * (F.relu(tt_refreeze - T) + eps).pow(refreeze_exp))
    swi = torch.maximum(swi_min, swi_max * (1.0 - alpha_swi * S_cum))
    outflow = F.relu(S_liq + rainfall + melt - swi * S_snow)
    S_snow_new = S_snow + snowfall - melt + refreeze
    # F.relu 防止 S_liq 变负（同 snobal_hbv 的修复）
    S_liq_new = F.relu(S_liq + melt + rainfall - refreeze - outflow)
    S_cum_new = torch.where(S_snow > eps, S_cum + melt, torch.zeros_like(S_cum))
    return outflow, S_snow_new, S_liq_new, S_cum_new, melt


# ---------- 3. 入渗 (Infiltration) ----------

def inf_hmets(P_eff, S, Smax, c_runoff, eps=1e-6):
    """HMETS: INF = P_eff * max(1 - c_runoff * S/Smax, 0)"""
    return P_eff * F.relu(1.0 - c_runoff * S / (Smax + eps))

def inf_vic_arno(P_eff, S, Smax, b_exp, eps=1e-6):
    """VIC_ARNO: INF = P_eff * [1 - (1 - S/Smax)^b_exp]
    VIC 空间异质性产流：土壤越满 (S→Smax) 饱和面积越大，入渗比例越高，公式方向正确。
    ratio = 1 - S/Smax (剩余容量比)；当 S→Smax 时 ratio→0，b_exp<1 时梯度 ∝ ratio^(b-1) 爆炸，
    clamp(min=0.01) 将最大梯度限制在有限范围内。
    """
    ratio = torch.clamp(1.0 - S / (Smax + eps), 0.0, 1.0)
    return P_eff * (1.0 - ratio.clamp(min=0.01).pow(b_exp))

def inf_hbv(P_eff, S, Smax, beta, eps=1e-6):
    """HBV: INF = P_eff * [1 - (S/Smax)^beta]"""
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    # ratio.clamp(min=0.01) 防止 beta<1 时 ratio→0 的梯度 ∝ (eps)^(beta-1) 爆炸
    return P_eff * (1.0 - ratio.clamp(min=0.01).pow(beta))


# ---------- 4. 蒸发 (Soil Evaporation) ----------

def soilevap_all(PET, c_pet, S, eps=1e-6):
    """SOILEVAP_ALL: E = min(PET * c_pet, S)"""
    return torch.minimum(PET * c_pet, S)

def soilevap_linear(PET, c_pet, S, S_tension, eps=1e-6):
    """SOILEVAP_LINEAR: E = PET * c_pet * min(S/S_tension, 1)"""
    ratio = torch.clamp(S / (S_tension + eps), 0.0, 1.0)
    return torch.minimum(PET * c_pet * ratio, S)

def soilevap_vic(PET, c_pet, S, Smax, gamma, eps=1e-6):
    """SOILEVAP_VIC: E = PET * c_pet * (1 - (1-S/Smax)^gamma)"""
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    # (1-ratio).clamp(min=0.01) 防止 gamma<1 时 (1-ratio)→0 的梯度 ∝ (1-ratio)^(gamma-1) 爆炸
    evap = PET * c_pet * (1.0 - (1.0 - ratio).clamp(min=0.01).pow(gamma))
    return torch.minimum(evap, S)


# ---------- 5. 快速流 (Quickflow) ----------

def quick_linear_analytic(S, k_quick, eps=1e-6):
    """LINEAR_ANALYTIC: Q = S * (1 - exp(-k_quick))"""
    return S * (1.0 - torch.exp(-k_quick))

def quick_vic(S, Smax, q_max, n_q, eps=1e-6):
    """VIC: Q = q_max * (S/Smax)^n_q"""
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    return torch.minimum(q_max * (ratio + eps).pow(n_q), S)

def quick_topmodel(S, Smax, q_max, n_q, lam, eps=1e-6):
    """TOPMODEL: Q = q_max * exp(-lam * (1 - S/Smax))"""
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    return torch.minimum(q_max * torch.exp(-lam * (1.0 - ratio)), S)


# ---------- 6. 基流 (Baseflow) ----------

def base_linear_analytic(S, k_base, eps=1e-6):
    """LINEAR_ANALYTIC: Q_b = S * (1 - exp(-k_base))"""
    return S * (1.0 - torch.exp(-k_base))

def base_power_law(S, k_base, n_base, eps=1e-6):
    """POWER_LAW: Q_b = k_base * S^n_base"""
    return torch.minimum(k_base * (S + eps).pow(n_base), S)


# ---------- 7. 渗漏 (Percolation, 固定) ----------

def perc_linear(S, k_perc, eps=1e-6):
    """LINEAR: PERC = k_perc * S"""
    return torch.minimum(k_perc * S, S)


# ---------- 8. 汇流 (Convolution, Gamma UH) ----------
# 使用 dmg.models.hydrodl2 的 uh_gamma + uh_conv



# ===================================================================
# 主模型类
# ===================================================================

# 过程选项数量
PROCESS_OPTIONS = {
    "rainsnow": 3,    # HBV, Dingman, Threshold
    "snowbal": 3,     # Simple, HBV, HMETS
    "infiltration": 3,# HMETS, VIC_ARNO, HBV
    "evaporation": 3, # ALL, LINEAR, VIC
    "quickflow": 3,   # LINEAR_ANALYTIC, VIC, TOPMODEL
    "baseflow": 2,    # LINEAR_ANALYTIC, POWER_LAW
}
TOTAL_WEIGHT_LOGITS = sum(PROCESS_OPTIONS.values())  # 17
STEP_FLUX_KEYS = (
    "snowfall_opts",
    "rainfall_opts",
    "snow_outflow_opts",
    "melt_opts",
    "infiltration_opts",
    "surface_runoff",
    "evaporation_opts",
    "quickflow_opts",
    "baseflow_opts",
    "Q_surface",
    "Q_quick",
    "Q_base",
)
FLUX_OPTION_DIMS = {
    "snowfall_opts": 3,
    "rainfall_opts": 3,
    "snow_outflow_opts": 3,
    "melt_opts": 3,
    "infiltration_opts": 3,
    "evaporation_opts": 3,
    "quickflow_opts": 3,
    "baseflow_opts": 2,
}


class BlendStepForcing(NamedTuple):
    P_t: torch.Tensor
    T_t: torch.Tensor
    PET_t: torch.Tensor


class BlendStepState(NamedTuple):
    S_snow: torch.Tensor
    S_liq: torch.Tensor
    S_cum: torch.Tensor
    S_top: torch.Tensor
    S_phreatic: torch.Tensor


class BlendStepParams(NamedTuple):
    x33_rain_correction: torch.Tensor
    x34_snow_correction: torch.Tensor
    x31_rainsnow_temp: torch.Tensor
    x32_rainsnow_delta: torch.Tensor
    x24_min_melt_factor: torch.Tensor
    x26_dd_melt_temp: torch.Tensor
    max_melt_factor: torch.Tensor
    x18_refreeze_factor: torch.Tensor
    x16_refreeze_temp: torch.Tensor
    x19_snow_swi_hbv: torch.Tensor
    x17_refreeze_exp: torch.Tensor
    x13_swi_min: torch.Tensor
    swi_max: torch.Tensor
    x15_swi_reduct: torch.Tensor
    Smax_top: torch.Tensor
    x1_hmets_runoff_coeff: torch.Tensor
    x2_b_exp: torch.Tensor
    x3_hbv_beta: torch.Tensor
    x8_pet_correction: torch.Tensor
    field_capacity: torch.Tensor
    x28_perc_coeff_top: torch.Tensor
    k_quick: torch.Tensor
    x5_q_max: torch.Tensor
    x6_n_quick: torch.Tensor
    x7_topmodel_lambda: torch.Tensor
    Smax_phreatic: torch.Tensor
    x35_perc_coeff_phreatic: torch.Tensor
    k_base: torch.Tensor
    x12_n_base: torch.Tensor
    x27_dd_aggradation: torch.Tensor
    soilevap_vic_gamma: torch.Tensor


class BlendStepWeights(NamedTuple):
    rainsnow: torch.Tensor
    snowbal: torch.Tensor
    infiltration: torch.Tensor
    evaporation: torch.Tensor
    quickflow: torch.Tensor
    baseflow: torch.Tensor


class BlendStepOutput(NamedTuple):
    Q_total: torch.Tensor
    S_snow: torch.Tensor
    S_liq: torch.Tensor
    S_cum: torch.Tensor
    S_top: torch.Tensor
    S_phreatic: torch.Tensor
    snowfall_opts: torch.Tensor
    rainfall_opts: torch.Tensor
    snow_outflow_opts: torch.Tensor
    melt_opts: torch.Tensor
    infiltration_opts: torch.Tensor
    surface_runoff: torch.Tensor
    evaporation_opts: torch.Tensor
    quickflow_opts: torch.Tensor
    baseflow_opts: torch.Tensor
    Q_surface: torch.Tensor
    Q_quick: torch.Tensor
    Q_base: torch.Tensor


def diff_blend_step(
    forcing: BlendStepForcing,
    state: BlendStepState,
    params: BlendStepParams,
    weights: BlendStepWeights,
    eps: float = 1e-6,
) -> BlendStepOutput:
    """单时间步计算，返回总径流、更新状态和过程通量记录。"""

    P_t, T_t, PET_t = forcing
    S_snow, S_liq, S_cum, S_top, S_phreatic = state

    # ========== 1. 雨雪分割 (3 options) ==========
    sf1, rf1 = rainsnow_hbv(
        P_t,
        T_t,
        params.x31_rainsnow_temp,
        params.x32_rainsnow_delta,
    )
    sf2, rf2 = rainsnow_dingman(P_t, T_t, params.x31_rainsnow_temp)
    sf3, rf3 = rainsnow_threshold(P_t, T_t, params.x31_rainsnow_temp)
    # 先进行雨雪分割，再分别施加雨/雪校正系数
    sf1 = sf1 * params.x34_snow_correction
    sf2 = sf2 * params.x34_snow_correction
    sf3 = sf3 * params.x34_snow_correction
    rf1 = rf1 * params.x33_rain_correction
    rf2 = rf2 * params.x33_rain_correction
    rf3 = rf3 * params.x33_rain_correction

    w_rs = weights.rainsnow  # [B, 1, 3]
    snowfall_stack = torch.stack([sf1, sf2, sf3], dim=-1)  # [B, nmul, 3]
    rainfall_stack = torch.stack([rf1, rf2, rf3], dim=-1)
    snowfall = (snowfall_stack * w_rs).sum(-1)
    rainfall = (rainfall_stack * w_rs).sum(-1)

    # ========== 2. 雪平衡 (3 options) ==========
    w_sn = weights.snowbal  # [B, 1, 3]

    out1, Ss1, Sc1, melt1 = snobal_simple(
        snowfall,
        rainfall,
        T_t,
        S_snow,
        S_cum,
        params.x24_min_melt_factor,
        params.x26_dd_melt_temp,
        eps,
    )
    Sl1 = torch.zeros_like(S_liq)  # snobal_simple 无液态水层，重置为零避免非物理累积

    out2, Ss2, Sl2, Sc2, melt2 = snobal_hbv(
        snowfall,
        rainfall,
        T_t,
        S_snow,
        S_liq,
        S_cum,
        params.max_melt_factor,
        params.x26_dd_melt_temp,
        params.x18_refreeze_factor,
        params.x16_refreeze_temp,
        params.x19_snow_swi_hbv,
        eps,
    )

    out3, Ss3, Sl3, Sc3, melt3 = snobal_hmets(
        snowfall,
        rainfall,
        T_t,
        S_snow,
        S_liq,
        S_cum,
        params.x24_min_melt_factor,
        params.max_melt_factor,
        params.x26_dd_melt_temp,
        params.x18_refreeze_factor,
        params.x16_refreeze_temp,
        params.x17_refreeze_exp,
        params.x13_swi_min,
        params.swi_max,
        params.x15_swi_reduct,
        params.x27_dd_aggradation,
        eps,
    )

    snow_outflow_stack = torch.stack([out1, out2, out3], dim=-1)
    snow_outflow = (snow_outflow_stack * w_sn).sum(-1)

    S_snow_new = (torch.stack([Ss1, Ss2, Ss3], -1) * w_sn).sum(-1)
    S_liq_new = (torch.stack([Sl1, Sl2, Sl3], -1) * w_sn).sum(-1)
    S_cum_new = (torch.stack([Sc1, Sc2, Sc3], -1) * w_sn).sum(-1)
    melt_stack = torch.stack([melt1, melt2, melt3], -1)

    # ========== 3. 入渗 (3 options) ==========
    w_inf = weights.infiltration  # [B, 1, 3]

    inf1 = inf_hmets(snow_outflow, S_top, params.Smax_top, params.x1_hmets_runoff_coeff, eps)
    inf2 = inf_vic_arno(snow_outflow, S_top, params.Smax_top, params.x2_b_exp, eps)
    inf3 = inf_hbv(snow_outflow, S_top, params.Smax_top, params.x3_hbv_beta, eps)

    inf_stack = torch.stack([inf1, inf2, inf3], dim=-1)
    infiltration = (inf_stack * w_inf).sum(-1)
    surface_runoff = snow_outflow - infiltration

    # ========== 4. 蒸发 (3 options) ==========
    w_ev = weights.evaporation  # [B, 1, 3]

    ev1 = soilevap_all(PET_t, params.x8_pet_correction, S_top, eps)
    ev2 = soilevap_linear(
        PET_t,
        params.x8_pet_correction,
        S_top,
        params.field_capacity * params.Smax_top,
        eps,
    )
    ev3 = soilevap_vic(
        PET_t,
        params.x8_pet_correction,
        S_top,
        params.Smax_top,
        params.soilevap_vic_gamma,
        eps,
    )

    ev_stack = torch.stack([ev1, ev2, ev3], dim=-1)
    evaporation = (ev_stack * w_ev).sum(-1)

    # ========== 5. 渗漏 (表层 -> 潜水层, 固定) ==========
    percolation_top = perc_linear(S_top, params.x28_perc_coeff_top, eps)

    # ========== 更新表层土壤 ==========
    S_top_new = S_top + infiltration - evaporation - percolation_top
    S_top_new = F.relu(S_top_new)
    overflow_top = F.relu(S_top_new - params.Smax_top)
    S_top_new = S_top_new - overflow_top
    surface_runoff = surface_runoff + overflow_top

    # ========== 6. 快速流 (表层, 3 options) ==========
    w_qf = weights.quickflow  # [B, 1, 3]

    qf1 = quick_linear_analytic(S_top_new, params.k_quick, eps)
    qf2 = quick_vic(S_top_new, params.Smax_top, params.x5_q_max, params.x6_n_quick, eps)
    qf3 = quick_topmodel(
        S_top_new,
        params.Smax_top,
        params.x5_q_max,
        params.x6_n_quick,
        params.x7_topmodel_lambda,
        eps,
    )

    qf_stack = torch.stack([qf1, qf2, qf3], dim=-1)
    quickflow = (qf_stack * w_qf).sum(-1)
    S_top_final = F.relu(S_top_new - quickflow)

    # ========== 7. 潜水层: 渗漏入 + 基流出 ==========
    # perc_phreatic 代表深层渗漏至未建模的深层含水层 (deep drainage sink)，属于设计有意的水量漏损，不进入径流
    perc_phreatic = perc_linear(S_phreatic, params.x35_perc_coeff_phreatic, eps)

    w_bf = weights.baseflow  # [B, 1, 2]
    bf1 = base_linear_analytic(S_phreatic, params.k_base, eps)
    bf2 = base_power_law(S_phreatic, params.k_base, params.x12_n_base, eps)

    bf_stack = torch.stack([bf1, bf2], dim=-1)
    baseflow = (bf_stack * w_bf).sum(-1)

    S_phreatic_new = S_phreatic + percolation_top - perc_phreatic - baseflow
    S_phreatic_new = F.relu(S_phreatic_new)
    overflow_phreatic = F.relu(S_phreatic_new - params.Smax_phreatic)
    S_phreatic_new = S_phreatic_new - overflow_phreatic
    baseflow = baseflow + overflow_phreatic

    Q_total = surface_runoff + quickflow + baseflow

    return BlendStepOutput(
        Q_total=Q_total,
        S_snow=S_snow_new,
        S_liq=S_liq_new,
        S_cum=S_cum_new,
        S_top=S_top_final,
        S_phreatic=S_phreatic_new,
        snowfall_opts=snowfall_stack.detach(),
        rainfall_opts=rainfall_stack.detach(),
        snow_outflow_opts=snow_outflow_stack,   
        melt_opts=melt_stack.detach(),
        infiltration_opts=inf_stack,            
        surface_runoff=surface_runoff.detach(),
        evaporation_opts=ev_stack,              
        quickflow_opts=qf_stack,                
        baseflow_opts=bf_stack,                 
        Q_surface=surface_runoff.detach(),
        Q_quick=quickflow.detach(),
        Q_base=baseflow.detach(),
    )


class DiffBlendV1(nn.Module):
    """可微分公式混合水文模型 V1

    每个水文过程有多个公式选项，通过权重混合。
    3层土壤结构: TOPSOIL + PHREATIC + Groundwater(无限深)。
    """

    # 35个物理参数的边界 (x1 ~ x35)
    PARAM_BOUNDS = {
        # 入渗
        "x1_hmets_runoff_coeff": [0.0, 1.0],
        "x2_b_exp": [0.3, 3.0],
        "x3_hbv_beta": [0.5, 3.0],
        # 快速流
        "x4_log_k_quick": [-5.0, -1.0],
        "x5_q_max": [0.0, 100.0],
        "x6_n_quick": [0.5, 2.0],
        "x7_topmodel_lambda": [5.0, 10.0],
        # 蒸发
        "x8_pet_correction": [0.0, 3.0],
        "x9_sat_wilt": [0.0, 0.05],
        "x10_delta_fc": [0.0, 0.45],
        # 基流
        "x11_log_k_base": [-5.0, -2.0],
        "x12_n_base": [0.5, 2.0],
        # 雪平衡
        "x13_swi_min": [0.0, 0.1],
        "x14_delta_swi_max": [0.01, 0.3],
        "x15_swi_reduct": [0.005, 0.1],
        "x16_refreeze_temp": [-5.0, 2.0],
        "x17_refreeze_exp": [0.3, 1.0],
        "x18_refreeze_factor": [0.0, 5.0],
        "x19_snow_swi_hbv": [0.0, 0.4],
        # 汇流
        "x20_gamma_shape_surf": [0.3, 20.0],
        "x21_gamma_scale_surf": [0.01, 5.0],
        "x22_gamma_shape_delay": [0.5, 13.0],
        "x23_gamma_scale_delay": [0.15, 1.5],
        # 潜在融雪
        "x24_min_melt_factor": [1.5, 3.0],
        "x25_delta_melt_factor": [0.0, 5.0],
        "x26_dd_melt_temp": [-1.0, 1.0],
        "x27_dd_aggradation": [0.01, 0.2],
        # 渗漏与土壤
        "x28_perc_coeff_top": [0.00001, 0.02],
        "x29_thickness_top": [0.0, 0.5],
        "x30_thickness_phreatic": [0.0, 2.0],
        # 气象
        "x31_rainsnow_temp": [-3.0, 3.0],
        "x32_rainsnow_delta": [0.5, 4.0],
        "x33_rain_correction": [0.8, 1.2],
        "x34_snow_correction": [0.8, 1.2],
        "x35_perc_coeff_phreatic": [0.0, 0.02],
        "x36_soilevap_vic_gamma": [0.3, 3.0],
    }

    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}


    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.name = "DiffBlendV1"
        self.config = config or {}
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.activate = torch.sigmoid
        self.weight_method = "gumbel_softmax"  # softmax | gumbel_softmax | sparsemax | entmax15
        self.tau = 1.0
        self.balance_window = 90

        self.param_names = list(self.PARAM_BOUNDS.keys())
        self.routing_param_names = list(self.ROUTING_BOUNDS.keys())
        self.process_names = list(PROCESS_OPTIONS.keys())

        # 总参数: 物理参数*nmul + 路由参数 + 权重logits
        self.learnable_param_count = (
            len(self.param_names) * self.nmul
            + len(self.routing_param_names)
            + TOTAL_WEIGHT_LOGITS
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if config is not None:
            self._load_config(config)
        self._setup_compiled_kernels()

    def _load_config(self, config: Dict[str, Any]) -> None:
        for attr in ["warm_up", "warm_up_states", "variables", "nearzero", "nmul"]:
            if attr in config:
                setattr(self, attr, config[attr])
        if "weight_method" in config:
            self.weight_method = config["weight_method"]
        if "tau" in config:
            self.tau = config["tau"]
        if "balance_window" in config:
            self.balance_window = config["balance_window"]
        # 重新计算参数数量
        self.learnable_param_count = (
            len(self.param_names) * self.nmul
            + len(self.routing_param_names)
            + TOTAL_WEIGHT_LOGITS
        )

    def _setup_compiled_kernels(self) -> None:
        """使用 torch.compile 预编译关键单步函数。"""
        self.blend_step_compiled = torch.compile(diff_blend_step)

    # ---------------------------------------------------------------
    # 参数解包与反归一化
    # ---------------------------------------------------------------

    def unpack_parameters(
        self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """解包 NN 输出为物理参数、权重、路由参数。

        Args:
            parameters: (None, raw_tensor) where raw_tensor shape [B, learnable_param_count]

        Returns:
            phy_params: {name: [B, nmul]} 物理参数 (已反归一化)
            weights: {process_name: [B, n_options]} 归一化权重
            routing_params: {name: [B]} 路由参数 (已反归一化)
        """
        _, raw = parameters
        B = raw.shape[0]
        n_phy = len(self.param_names)
        n_rout = len(self.routing_param_names)

        # 切分
        idx = 0
        raw_phy = raw[:, idx: idx + n_phy * self.nmul]
        idx += n_phy * self.nmul
        raw_rout = raw[:, idx: idx + n_rout]
        idx += n_rout
        raw_weights = raw[:, idx: idx + TOTAL_WEIGHT_LOGITS]

        # 物理参数: sigmoid -> descale
        phy_activated = self.activate(raw_phy).view(B, n_phy, self.nmul)
        phy_params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.PARAM_BOUNDS[name]
            phy_params[name] = phy_activated[:, i, :] * (hi - lo) + lo

        # 路由参数
        rout_activated = self.activate(raw_rout)
        routing_params = {}
        for i, name in enumerate(self.routing_param_names):
            lo, hi = self.ROUTING_BOUNDS[name]
            routing_params[name] = rout_activated[:, i] * (hi - lo) + lo

        # 权重: 按过程切分 logits -> activate
        weights = {}
        w_idx = 0
        for proc_name in self.process_names:
            n_opt = PROCESS_OPTIONS[proc_name]
            logits = raw_weights[:, w_idx: w_idx + n_opt]  # [B, n_opt]
            weights[proc_name] = activate_weights(
                logits, self.weight_method, self.tau, self.training, dim=-1
            )
            w_idx += n_opt

        return phy_params, weights, routing_params


    # ---------------------------------------------------------------
    # 参数转换 (blend_model.md 中的关键规则)
    # ---------------------------------------------------------------

    def _transform_params(self, p: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用 blend_model.md 中的参数转换规则。"""
        t = dict(p)  # shallow copy
        # 1. 对数采样还原
        t["k_quick"] = 10.0 ** t["x4_log_k_quick"]          # BASEFLOW_COEFF TOPSOIL
        t["k_base"] = 10.0 ** t["x11_log_k_base"]            # BASEFLOW_COEFF PHREATIC
        # 2. 增量参数 -> 真实值
        t["field_capacity"] = t["x9_sat_wilt"] + t["x10_delta_fc"]
        t["swi_max"] = t["x13_swi_min"] + t["x14_delta_swi_max"]
        t["max_melt_factor"] = t["x24_min_melt_factor"] + t["x25_delta_melt_factor"]
        # 3. 土壤容量 (thickness * 1000 mm/m 简化为直接使用 mm)
        t["Smax_top"] = t["x29_thickness_top"] * 1000.0 + 1.0   # 避免零
        t["Smax_phreatic"] = t["x30_thickness_phreatic"] * 1000.0 + 1.0
        return t

    # ---------------------------------------------------------------
    # 滑动窗口累积
    # ---------------------------------------------------------------

    @staticmethod
    def _rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
        """对时间维做因果滑动累积和（全程可导，梯度可通过 F.conv1d 反传）。

        Args:
            x: [T, B, n_options]
            window: 窗口天数

        Returns:
            [T, B, n_options]，每个位置是以该点为终点的 window 步累积和
        """
        T, B, n_opt = x.shape
        # 重排为 [B*n_opt, 1, T] 以适配 F.conv1d
        x_perm = x.permute(1, 2, 0).reshape(B * n_opt, 1, T)
        kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
        # 仅左侧补零，保证因果性（输出[t] = sum(x[t-window+1:t+1])）
        x_padded = F.pad(x_perm, (window - 1, 0))
        out = F.conv1d(x_padded, kernel)  # [B*n_opt, 1, T]
        return out.reshape(B, n_opt, T).permute(2, 0, 1)

    # ---------------------------------------------------------------
    # 路由
    # ---------------------------------------------------------------

    def _apply_routing(
        self, Q: torch.Tensor, rout_a: torch.Tensor, rout_b: torch.Tensor
    ) -> torch.Tensor:
        """Gamma 单位线路由。Q: [T, B], rout_a/rout_b: [B]"""
        n_steps, n_grid = Q.shape
        UH = uh_gamma(
            rout_a.unsqueeze(0).expand(n_steps, -1).unsqueeze(-1),
            rout_b.unsqueeze(0).expand(n_steps, -1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])
        rf = Q.unsqueeze(-1).permute([1, 2, 0])
        return uh_conv(rf, UH).permute([2, 0, 1]).squeeze(-1)

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: 包含 "x_phy" [T, B, n_vars], "target" 等
            parameters: (None, raw_params [B, learnable_param_count])

        Returns:
            result dict: streamflow, 各过程权重, 各公式flux记录
        """
        x = x_dict["x_phy"]
        n_steps, n_grid, _ = x.shape

        # 1. 解包参数
        phy_params, weights, routing_params = self.unpack_parameters(parameters)

        # 2. 参数转换
        p = self._transform_params(phy_params)

        # 3. 准备驱动数据 [T, B] -> [T, B, nmul]
        P = x[:, :, self.variables.index("prcp")].unsqueeze(2).expand(-1, -1, self.nmul)
        T = x[:, :, self.variables.index("tmean")].unsqueeze(2).expand(-1, -1, self.nmul)
        PET = x[:, :, self.variables.index("pet")].unsqueeze(2).expand(-1, -1, self.nmul)

        # 4. 初始化状态 [B, nmul]
        step_state = BlendStepState(
            S_snow=torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            S_liq=torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            S_cum=torch.zeros(n_grid, self.nmul, device=self.device),
            S_top=torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            S_phreatic=torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
        )

        # 5. 预分配输出
        Q_sim = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)

        # 预分配 flux buffer，避免 append + stack 的额外内存复制
        flux_buffers: Dict[str, torch.Tensor] = {}
        for key in STEP_FLUX_KEYS:
            if key in FLUX_OPTION_DIMS:
                flux_buffers[key] = torch.zeros(
                    n_steps,
                    n_grid,
                    FLUX_OPTION_DIMS[key],
                    device=self.device,
                    dtype=x.dtype,
                )
            else:
                flux_buffers[key] = torch.zeros(
                    n_steps,
                    n_grid,
                    device=self.device,
                    dtype=x.dtype,
                )

        # 预分配 balance buffer 列表 (不 detach，保留梯度链用于 balance loss)
        _BALANCE_OPT_KEYS = {
            "infiltration": "infiltration_opts",
            "evaporation":  "evaporation_opts",
            "quickflow":    "quickflow_opts",
            "baseflow":     "baseflow_opts",
            "snow_outflow": "snow_outflow_opts",
        }
        balance_buf_lists: Dict[str, list] = {k: [] for k in _BALANCE_OPT_KEYS}

        step_params = BlendStepParams(
            x33_rain_correction=p["x33_rain_correction"],
            x34_snow_correction=p["x34_snow_correction"],
            x31_rainsnow_temp=p["x31_rainsnow_temp"],
            x32_rainsnow_delta=p["x32_rainsnow_delta"],
            x24_min_melt_factor=p["x24_min_melt_factor"],
            x26_dd_melt_temp=p["x26_dd_melt_temp"],
            max_melt_factor=p["max_melt_factor"],
            x18_refreeze_factor=p["x18_refreeze_factor"],
            x16_refreeze_temp=p["x16_refreeze_temp"],
            x19_snow_swi_hbv=p["x19_snow_swi_hbv"],
            x17_refreeze_exp=p["x17_refreeze_exp"],
            x13_swi_min=p["x13_swi_min"],
            swi_max=p["swi_max"],
            x15_swi_reduct=p["x15_swi_reduct"],
            Smax_top=p["Smax_top"],
            x1_hmets_runoff_coeff=p["x1_hmets_runoff_coeff"],
            x2_b_exp=p["x2_b_exp"],
            x3_hbv_beta=p["x3_hbv_beta"],
            x8_pet_correction=p["x8_pet_correction"],
            field_capacity=p["field_capacity"],
            x28_perc_coeff_top=p["x28_perc_coeff_top"],
            k_quick=p["k_quick"],
            x5_q_max=p["x5_q_max"],
            x6_n_quick=p["x6_n_quick"],
            x7_topmodel_lambda=p["x7_topmodel_lambda"],
            Smax_phreatic=p["Smax_phreatic"],
            x35_perc_coeff_phreatic=p["x35_perc_coeff_phreatic"],
            k_base=p["k_base"],
            x12_n_base=p["x12_n_base"],
            x27_dd_aggradation=p["x27_dd_aggradation"],
            soilevap_vic_gamma=p["x36_soilevap_vic_gamma"],
        )
        # 权重为流域级结构偏好，不随 nmul 扩展；unsqueeze(1) 在时间步内 broadcast 到 nmul 维
        step_weights = BlendStepWeights(
            rainsnow=weights["rainsnow"].unsqueeze(1),
            snowbal=weights["snowbal"].unsqueeze(1),
            infiltration=weights["infiltration"].unsqueeze(1),
            evaporation=weights["evaporation"].unsqueeze(1),
            quickflow=weights["quickflow"].unsqueeze(1),
            baseflow=weights["baseflow"].unsqueeze(1),
        )

        # 6. 时间循环
        for t in range(n_steps):
            step_out = self.blend_step_compiled(
                BlendStepForcing(P_t=P[t], T_t=T[t], PET_t=PET[t]),
                step_state,
                step_params,
                step_weights,
                self.nearzero,
            )
            Q_sim[t] = step_out.Q_total

            step_state = BlendStepState(
                S_snow=step_out.S_snow,
                S_liq=step_out.S_liq,
                S_cum=step_out.S_cum,
                S_top=step_out.S_top,
                S_phreatic=step_out.S_phreatic,
            )

            # 收集 flux 记录 (写入 flux_buffers 前 detach，仅用于可视化)
            for key in STEP_FLUX_KEYS:
                val = getattr(step_out, key)
                if val.dim() == 3:
                    # [B, nmul, n_options] -> [B, n_options]
                    val_reduced = val.mean(dim=1).detach()
                elif val.dim() == 2:
                    # [B, nmul] -> [B]
                    val_reduced = val.mean(dim=-1).detach()
                else:
                    val_reduced = val.detach()
                flux_buffers[key][t] = val_reduced

            # 收集 balance buffer (不 detach，保留梯度链用于 balance loss)
            for proc_name, buf_key in _BALANCE_OPT_KEYS.items():
                val = getattr(step_out, buf_key)  # [B, nmul, n_options]
                balance_buf_lists[proc_name].append(val.mean(dim=1))  # [B, n_options]

        # stack balance buf lists -> [T, B, n_options]，梯度链完整
        balance_stacked: Dict[str, torch.Tensor] = {
            proc_name: torch.stack(lst, dim=0)
            for proc_name, lst in balance_buf_lists.items()
        }

        # 7. 平均 nmul
        Q_mean = Q_sim.mean(-1)  # [T, B]

        # 8. 路由
        Qrouted = self._apply_routing(
            Q_mean, routing_params["rout_a"], routing_params["rout_b"]
        )

        # 9. 构造返回字典
        result: Dict[str, torch.Tensor] = {
            "streamflow": Qrouted,
        }

        # 保存权重 (扩展到时间维度)
        for proc_name in self.process_names:
            w = weights[proc_name]  # [B, n_options]
            n_opt = w.shape[-1]
            for i in range(n_opt):
                key = f"w_{proc_name}_{i}"
                result[key] = w[:, i].unsqueeze(0).expand(n_steps, -1)

        # 保存各过程 flux 记录
        for key, stacked in flux_buffers.items():
            if stacked.dim() == 3:
                # 多选项: 保存每个选项
                for i in range(stacked.shape[-1]):
                    result[f"{key}_{i}"] = stacked[:, :, i]
            else:
                result[key] = stacked

        # --- 公式累积平衡约束序列 (步骤 9 后，warmup 截断前) ---
        # 使用保留梯度的 balance_stacked，_rolling_sum 全程可导，梯度可传播至模型参数
        for proc_name, buf in balance_stacked.items():
            rolled = self._rolling_sum(buf, self.balance_window)  # [T, B, n_options]
            n_opt = buf.shape[-1]
            for j in range(1, n_opt):
                result[f"balance_{proc_name}_0_{j}"] = rolled[:, :, 0] - rolled[:, :, j]

        # 10. 截断 warmup
        if not self.warm_up_states:
            cutoff = self.warm_up if self.pred_cutoff == 0 else self.pred_cutoff
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][cutoff:]

        return result
