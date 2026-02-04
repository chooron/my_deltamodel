"""
水文模型核心计算模块 — 仅保留单步计算逻辑。

包含四个模型的单步函数（无时间循环）：
- HBV: hbv_step
- SHM: shm_step
- EXPHYDRO: exphydro_step
- HYMOD: hymod_step

Author: chooron
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


MODEL_STATES_NUM = {
    "HBV": 5,
    "SHM": 4,
    "EXPHYDRO": 2,
    "HYMOD": 5,
}

HBV_PARAMS_BOUNDS = {
    "tt": [-3.0, 5.0],
    "tti": [0.0, 17.0],
    "ttm": [-3.0, 3.0],
    "cfr": [0.0, 1.0],
    "cfmax": [0.0, 20.0],
    "whc": [0.0, 1.0],
    "cflux": [0.0, 4.0],
    "fc": [1.0, 2000.0],  # Field Capacity - 土壤最大含水容量 (S3)
    "lp": [0.05, 0.95],
    "beta": [0.0, 10.0],
    "k0": [0.0, 1.0],
    "alpha": [0.0, 4.0],
    "perc": [0.0, 20.0],
    "k1": [0.0, 1.0],
}

SHM_PARAMS_BOUNDS = {
    "f_thr": [0.0, 50.0],
    "sumax": [
        20.0,
        700.0,
    ],  # Soil Moisture Storage Capacity - 土壤最大含水容量 (su)
    "beta": [1.0, 6.0],
    "perc": [0.0, 1.0],
    "kf": [1.0, 20.0],
    "ki": [1.0, 100.0],
    "kb": [10.0, 1000.0],
}

EXPHYDRO_PARAMS_BOUNDS = {
    "f": [0.0, 2.0],
    "ddf": [0.0, 10.0],
    "smax": [
        10.0,
        800.0,
    ],  # Maximum Soil Storage - 土壤最大含水容量 (soil_storage)
    "qmax": [0.0, 20.0],
    "mint": [-5.0, 2.0],
    "maxt": [0.0, 6.0],
}
HYMOD_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],  # Maximum Catchment Storage - 流域最大含水容量 (S1)
    "b_exp": [0.0, 10.0],
    "a_split": [0.0, 1.0],
    "kf": [0.0, 1.0],
    "ks": [0.0, 1.0],
}


def hbv_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    tt: torch.Tensor,
    tti: torch.Tensor,
    ttm: torch.Tensor,
    cfr: torch.Tensor,
    cfmax: torch.Tensor,
    whc: torch.Tensor,
    cflux: torch.Tensor,
    fc: torch.Tensor,
    lp: torch.Tensor,
    beta: torch.Tensor,
    k0: torch.Tensor,
    alpha: torch.Tensor,
    perc: torch.Tensor,
    k1: torch.Tensor,
    S1: Optional[torch.Tensor] = None,
    S2: Optional[torch.Tensor] = None,
    S3: Optional[torch.Tensor] = None,
    S4: Optional[torch.Tensor] = None,
    S5: Optional[torch.Tensor] = None,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """HBV-96 单步计算逻辑，直接展开，无外部导入包装。"""
    device = P.device
    n_grid = P.shape[0] if P.dim() > 0 else 1
    nmul = P.shape[1] if P.dim() > 1 else 1

    if S1 is None:
        S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S2 is None:
        S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S3 is None:
        S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S4 is None:
        S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S5 is None:
        S5 = torch.zeros((n_grid, nmul), device=device) + nearzero

    # 1) Snow routine (S1 & S2)
    # Snowfall
    t_max = tt + 0.5 * tti
    snow_frac = torch.clamp((t_max - T) / (tti + nearzero), min=0.0, max=1.0)
    flux_sf = P * snow_frac

    # Rainfall
    t_min = tt - 0.5 * tti
    rain_frac = torch.clamp((T - t_min) / (tti + nearzero), min=0.0, max=1.0)
    flux_rf = P * rain_frac

    # Melt
    melt_potential = cfmax * (T - ttm)
    flux_melt = F.relu(torch.minimum(melt_potential, S1))

    # Refreeze
    refreeze_potential = cfr * cfmax * (ttm - T)
    flux_refr = F.relu(torch.minimum(refreeze_potential, S2))

    S1 = S1 + flux_sf + flux_refr - flux_melt
    S1 = torch.clamp(S1, min=nearzero)

    S2_potential = S2 + flux_rf + flux_melt - flux_refr
    capacity = whc * S1
    flux_se = torch.relu(S2_potential - capacity)
    S2 = torch.minimum(S2_potential, capacity)
    S2 = torch.clamp(S2, min=nearzero)

    # 2) Interaction (S3 <-> S4) - Capillary rise
    flux_cf_pot = cflux * (1.0 - S3 / (fc + nearzero))
    flux_cf_pot = torch.minimum(F.relu(flux_cf_pot), S4)
    flux_cf = torch.minimum(flux_cf_pot, S4)

    S4 = S4 - flux_cf
    S3 = S3 + flux_cf

    # 3) Soil routine (S3)
    S3 = S3 + flux_se

    # Evaporation
    flux_ea_pot = torch.minimum(
        torch.minimum(S3 / (lp * fc + nearzero) * PET, PET), S3
    )
    flux_ea = torch.minimum(flux_ea_pot, S3)
    S3 = S3 - flux_ea

    # Recharge
    ratio_r = torch.clamp(F.relu(S3) / (fc + nearzero), max=1.5)
    flux_r_pot = flux_se * (ratio_r + nearzero).pow(beta)
    flux_r = torch.minimum(flux_r_pot, S3)
    S3 = S3 - flux_r
    S3 = torch.clamp(S3, min=nearzero)

    # 4) Upper zone (S4)
    S4 = S4 + flux_r

    # Percolation
    flux_perc_pot = torch.minimum(perc, S4)
    flux_perc = torch.minimum(flux_perc_pot, S4)
    S4 = S4 - flux_perc

    # Interflow
    flux_q0_pot = k0 * (S4 + nearzero).pow(1.0 + alpha)
    flux_q0_pot = torch.minimum(flux_q0_pot, S4)
    flux_q0 = torch.minimum(flux_q0_pot, S4)
    S4 = S4 - flux_q0
    S4 = torch.clamp(S4, min=nearzero)

    # 5) Lower zone (S5)
    S5 = S5 + flux_perc

    # Baseflow
    flux_q1_pot = k1 * S5
    flux_q1 = torch.minimum(flux_q1_pot, S5)
    S5 = S5 - flux_q1
    S5 = torch.clamp(S5, min=nearzero)

    # 6) Aggregation
    Qsim = flux_q0 + flux_q1
    Ea = flux_ea

    return Qsim, Ea, S1, S2, S3, S4, S5


def shm_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # --- Params ---
    f_thr: torch.Tensor,
    sumax: torch.Tensor,
    beta: torch.Tensor,
    perc: torch.Tensor,
    kf: torch.Tensor,
    ki: torch.Tensor,
    kb: torch.Tensor,
    # --- Constants ---
    nearzero: float = 1e-6,
    # --- States (Removed ss) ---
    su: Optional[torch.Tensor] = None,
    sf: Optional[torch.Tensor] = None,
    si: Optional[torch.Tensor] = None,
    sb: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    SHM 单步计算（无融雪、无表面存储ss版本）。
    适配接口：4个状态 (su, sf, si, sb)。
    """
    device = P.device
    n_grid = P.shape[0] if P.dim() > 0 else 1
    nmul = P.shape[1] if P.dim() > 1 else 1

    # 初始化状态
    if su is None:
        su = torch.zeros((n_grid, nmul), device=device) + nearzero
    if sf is None:
        sf = torch.zeros((n_grid, nmul), device=device) + nearzero
    if si is None:
        si = torch.zeros((n_grid, nmul), device=device) + nearzero
    if sb is None:
        sb = torch.zeros((n_grid, nmul), device=device) + nearzero

    zero = torch.tensor(0.0, device=device)
    one = torch.tensor(1.0, device=device)
    klu = torch.tensor(0.90, device=device)  # Land use correction
    pwp_ratio = torch.tensor(0.8, device=device)

    # ----------------------------------------------------
    # 1. 降雨分流 (Split)
    # ----------------------------------------------------
    # f_thr 控制超渗/优先流
    # qf_in: 进入快流 (Fast)
    # qu_in: 进入土壤 (Soil)
    qf_in = torch.maximum(zero, P - f_thr)
    qu_in = torch.minimum(P, f_thr)

    # ----------------------------------------------------
    # 2. 快流过程 (Fast Flow) -> sf
    # ----------------------------------------------------
    sf = sf + qf_in
    qf_out = sf / kf
    sf = sf - qf_out

    # ----------------------------------------------------
    # 3. 土壤过程 (Soil Moisture) -> su
    # ----------------------------------------------------
    # 产流因子 psi = (su / sumax)^beta
    rel_soil = torch.clamp(su / (sumax + nearzero), max=1.0)
    psi = rel_soil.pow(beta)

    # 更新土壤水
    su_temp = su + qu_in * (one - psi)

    # 溢出处理 (Fill-and-Spill)
    excess = torch.maximum(zero, su_temp - sumax)
    su = torch.minimum(su_temp, sumax)

    # 土壤产流 = 直接穿透(psi) + 溢出(excess)
    qu_out = qu_in * psi + excess

    # ----------------------------------------------------
    # 4. 蒸发 (Evapotranspiration)
    # ----------------------------------------------------
    pwp = pwp_ratio * sumax
    # 修正逻辑：相对于 PWP 线性衰减，而非相对于 Sumax
    # 这样在土壤较干时也能保持一定的蒸发能力
    rel_evap = su / (pwp + nearzero)
    ktetha = torch.clamp(rel_evap, max=1.0)

    ret = PET * klu * ktetha
    # 限制蒸发不能超过现有水量
    ret = torch.minimum(ret, su)
    su = su - ret

    # ----------------------------------------------------
    # 5. 慢流分流 (Interflow & Baseflow)
    # ----------------------------------------------------
    # 壤中流 (Interflow) -> si
    qi_in = qu_out * perc
    si = si + qi_in
    qi_out = si / ki
    si = si - qi_out

    # 基流 (Baseflow) -> sb
    qb_in = qu_out * (one - perc)
    sb = sb + qb_in
    qb_out = sb / kb
    sb = sb - qb_out

    # ----------------------------------------------------
    # 6. 总输出
    # ----------------------------------------------------
    Qsim = qf_out + qi_out + qb_out

    return Qsim, ret, su, sf, si, sb


def exphydro_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    f: torch.Tensor,
    ddf: torch.Tensor,
    smax: torch.Tensor,
    qmax: torch.Tensor,
    mint: torch.Tensor,
    maxt: torch.Tensor,
    nearzero: float,
    soil_storage: Optional[torch.Tensor] = None,
    snow_storage: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """EXP-HYDRO 单步计算，返回产流、蒸发损失及最新状态。"""
    device = P.device
    n_grid = P.shape[0] if P.dim() > 0 else 1
    nmul = P.shape[1] if P.dim() > 1 else 1

    if soil_storage is None:
        soil_storage = (
            torch.zeros((n_grid, nmul), dtype=torch.float32, device=device)
            + nearzero
        )
    if snow_storage is None:
        snow_storage = (
            torch.zeros((n_grid, nmul), dtype=torch.float32, device=device)
            + nearzero
        )

    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    safe_smax = torch.clamp(smax, min=1.0)

    ps = torch.where(T < mint, P, zero)
    pr = P - ps

    potential_melt = torch.where(T > maxt, ddf * (T - maxt), zero)
    melt = torch.minimum(snow_storage, potential_melt)
    snow_storage = snow_storage + ps - melt

    water_in = pr + melt
    s = soil_storage

    safe_s_for_exp = torch.minimum(s, safe_smax)
    saturation_ratio = torch.clamp(s / safe_smax, max=1.0)
    et_potential = PET * saturation_ratio

    term_exp = torch.exp(-f * (safe_smax - safe_s_for_exp))
    qsub_potential = qmax * term_exp

    total_outflow_demand = et_potential + qsub_potential
    scaling_factor = torch.minimum(
        torch.tensor(1.0, device=device), s / (total_outflow_demand + nearzero)
    )

    et = et_potential * scaling_factor
    qsub = qsub_potential * scaling_factor

    qsurf = torch.relu(s - safe_smax)

    ds = water_in - et - qsub - qsurf
    soil_storage = soil_storage + ds
    soil_storage = torch.clamp(soil_storage, min=0.0)

    qsim = qsub + qsurf

    return qsim, et, soil_storage, snow_storage, melt


def hymod_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    smax: torch.Tensor,
    b_exp: torch.Tensor,
    a_split: torch.Tensor,
    kf: torch.Tensor,
    ks: torch.Tensor,
    S1: Optional[torch.Tensor] = None,
    S2: Optional[torch.Tensor] = None,
    S3: Optional[torch.Tensor] = None,
    S4: Optional[torch.Tensor] = None,
    S5: Optional[torch.Tensor] = None,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """HyMOD 单步计算逻辑，直接展开。"""
    device = P.device
    n_grid = P.shape[0] if P.dim() > 0 else 1
    nmul = P.shape[1] if P.dim() > 1 else 1

    if S1 is None:
        S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S2 is None:
        S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S3 is None:
        S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S4 is None:
        S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    if S5 is None:
        S5 = torch.zeros((n_grid, nmul), device=device) + nearzero

    # Saturation excess
    s_rel = S1 / (smax + nearzero)
    term = torch.clamp(1.0 - s_rel, min=0.0, max=1.0)
    out_frac = 1.0 - (term + nearzero).pow(b_exp)
    flux_pe = out_frac * P
    zeros = torch.zeros_like(flux_pe)
    flux_pe = torch.clamp(flux_pe, min=zeros, max=P)

    S1_tmp = S1 + P - flux_pe
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # Evaporation
    ratio_ea = torch.clamp(S1_tmp / smax, max=1.0)
    flux_ea = torch.minimum(ratio_ea * PET, S1_tmp)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = F.relu(flux_ea)

    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero)

    # Runoff splitting
    flux_pf = a_split * flux_pe
    flux_ps = F.relu(flux_pe - flux_pf)

    # Fast Tank 1 (S2)
    S2_tmp = S2 + flux_pf
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    flux_qf1 = kf * S2_tmp
    flux_qf1 = torch.minimum(flux_qf1, S2_tmp - nearzero)
    S2_new = S2_tmp - flux_qf1

    # Fast Tank 2 (S3)
    S3_tmp = S3 + flux_qf1
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    flux_qf2 = kf * S3_tmp
    flux_qf2 = torch.minimum(flux_qf2, S3_tmp - nearzero)
    S3_new = S3_tmp - flux_qf2

    # Fast Tank 3 (S4)
    S4_tmp = S4 + flux_qf2
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    flux_qf3 = kf * S4_tmp
    flux_qf3 = torch.minimum(flux_qf3, S4_tmp - nearzero)
    S4_new = S4_tmp - flux_qf3

    # Slow Tank (S5)
    S5_tmp = S5 + flux_ps
    S5_tmp = torch.clamp(S5_tmp, min=nearzero)
    flux_qs = ks * S5_tmp
    flux_qs = torch.minimum(flux_qs, S5_tmp - nearzero)
    S5_new = S5_tmp - flux_qs

    Qsim = flux_qf3 + flux_qs
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new
