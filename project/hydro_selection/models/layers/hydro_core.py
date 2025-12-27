"""
水文模型核心计算模块

包含 JIT 优化的水文模型时间步循环函数，供各模型变体调用。

支持的模型:
- HBV: hbv_timestep_loop
- SHM: shm_timestep_loop
- CFE: cfe_timestep_loop
- ABCD: abcd_timestep_loop

Author: chooron
"""

import torch
from typing import Optional

HBV_PARAMS_BOUNDS = {
    "parTT": [-2.5, 2.5],
    "parCFMAX": [0.5, 10],
    "parCFR": [0, 0.1],
    "parCWH": [0, 0.2],
    "parFC": [50, 1000],
    "parBETA": [1.0, 6.0],
    "parLP": [0.2, 1],
    "parBETAET": [0.3, 5],
    "parC": [0, 1],
    "parPERC": [0, 10],
    "parK0": [0.05, 0.9],
    "parK1": [0.01, 0.5],
    "parK2": [0.001, 0.2],
    "parUZL": [0, 100],
}

SHM_PARAMS_BOUNDS = {
    "dd": [0.0, 10.0],
    "f_thr": [10.0, 60.0],
    "sumax": [20.0, 700.0],
    "beta": [1.0, 6.0],
    "perc": [0.0, 1.0],
    "kf": [1.0, 20.0],
    "ki": [1.0, 100.0],
    "kb": [10.0, 1000.0],
}

EXPHYDRO_PARAMS_BOUNDS = {
    "f": [0.0, 2.0],
    "ddf": [0.0, 10.0],
    "smax": [10.0, 800.0],
    "qmax": [0.0, 20.0],
    "mint": [-5.0, 2.0],
    "maxt": [0.0, 6.0],
}
HYMOD_PARAMS_BOUNDS = {
    "Tth": [-2.0, 2.0],
    "Tb": [-2.0, 2.0],
    "DDF": [0.0, 5.0],
    "Huz": [1.0, 800.0],
    "Cpar": [1.0, 1200.0],
    "B": [0.0, 2.5],
    "Kv": [0.0, 1.0],
    "alpha": [0.0, 1.0],
    "Kq": [0.0, 1.0],
    "Ks": [0.0, 5.0],
}

# @torch.jit.script
def hbv_timestep_loop(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    parTT: torch.Tensor,
    parCFMAX: torch.Tensor,
    parCFR: torch.Tensor,
    parCWH: torch.Tensor,
    parFC: torch.Tensor,
    parBETA: torch.Tensor,
    parLP: torch.Tensor,
    parBETAET: torch.Tensor,
    parC: torch.Tensor,
    parPERC: torch.Tensor,
    parK0: torch.Tensor,
    parK1: torch.Tensor,
    parK2: torch.Tensor,
    parUZL: torch.Tensor,
    nearzero: float,
    SNOWPACK: Optional[torch.Tensor] = None,
    MELTWATER: Optional[torch.Tensor] = None,
    SM: Optional[torch.Tensor] = None,
    SUZ: Optional[torch.Tensor] = None,
    SLZ: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    HBV 模型时间步循环（JIT 优化版本）

    Parameters
    ----------
    P, T, PET : torch.Tensor
        驱动数据，形状: (T, B, E)
        T=时间步数, B=流域数, E=专家数(nmul)
    SNOWPACK, MELTWATER, SM, SUZ, SLZ : torch.Tensor or None
        初始状态，形状: (B, E)。如果为 None，则默认使用零张量。
    par* : torch.Tensor
        模型参数，静态参数形状: (B, E)，动态参数形状: (T, B, E)
    nearzero : float
        防止除零的小数

    Returns
    -------
    tuple
        包含以下输出（均为形状 (T, B, E)）:
        - Qsim_out: 总产流
        - Q0_out: 快速壤中流
        - Q1_out: 慢速壤中流
        - Q2_out: 地下水出流
        - AET_out: 实际蒸发
        - recharge_out: 补给量
        - excs_out: 超渗量
        - evapfactor_out: 蒸发系数
        - tosoil_out: 入渗量
        - PERC_out: 下渗量
        - SWE_out: 雪水当量
        - SM_out: 土壤含水量
        - capillary_out: 毛管上升量
        - soil_wetness_out: 土壤湿度

        以及最终状态（均为形状 (B, E)）:
        - SNOWPACK: 积雪
        - MELTWATER: 融雪水
        - SM: 土壤水
        - SUZ: 上层地下水
        - SLZ: 下层地下水
    """
    n_steps = P.shape[0]
    n_grid = P.shape[1]
    nmul = P.shape[2]
    device = P.device

    # default initial states if not provided (shape: (B, E))
    if SNOWPACK is None:
        SNOWPACK = torch.zeros(
            (n_grid, nmul), dtype=torch.float32, device=device
        ) + nearzero
    if MELTWATER is None:
        MELTWATER = torch.zeros(
            (n_grid, nmul), dtype=torch.float32, device=device
        ) + nearzero
    if SM is None:
        SM = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if SUZ is None:
        SUZ = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if SLZ is None:
        SLZ = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero

    # 初始化输出张量
    Qsim_out = torch.zeros(
        (n_steps, n_grid, nmul), dtype=torch.float32, device=device
    ) + nearzero
    Q0_out = torch.zeros_like(Qsim_out) + nearzero
    Q1_out = torch.zeros_like(Qsim_out) + nearzero
    Q2_out = torch.zeros_like(Qsim_out) + nearzero
    AET_out = torch.zeros_like(Qsim_out) + nearzero
    recharge_out = torch.zeros_like(Qsim_out) + nearzero
    excs_out = torch.zeros_like(Qsim_out) + nearzero
    evapfactor_out = torch.zeros_like(Qsim_out) + nearzero
    tosoil_out = torch.zeros_like(Qsim_out) + nearzero
    PERC_out = torch.zeros_like(Qsim_out) + nearzero
    SWE_out = torch.zeros_like(Qsim_out) + nearzero
    SM_out = torch.zeros_like(Qsim_out) + nearzero
    capillary_out = torch.zeros_like(Qsim_out) + nearzero
    soil_wetness_out = torch.zeros_like(Qsim_out) + nearzero

    # 判断参数是否为动态（通过维度判断）
    parTT_is_dynamic = parTT.dim() == 3
    parCFMAX_is_dynamic = parCFMAX.dim() == 3
    parCFR_is_dynamic = parCFR.dim() == 3
    parCWH_is_dynamic = parCWH.dim() == 3
    parFC_is_dynamic = parFC.dim() == 3
    parBETA_is_dynamic = parBETA.dim() == 3
    parLP_is_dynamic = parLP.dim() == 3
    parBETAET_is_dynamic = parBETAET.dim() == 3
    parC_is_dynamic = parC.dim() == 3
    parPERC_is_dynamic = parPERC.dim() == 3
    parK0_is_dynamic = parK0.dim() == 3
    parK1_is_dynamic = parK1.dim() == 3
    parK2_is_dynamic = parK2.dim() == 3
    parUZL_is_dynamic = parUZL.dim() == 3

    for t in range(n_steps):
        # 获取当前时间步参数
        TT = parTT[t] if parTT_is_dynamic else parTT
        CFMAX = parCFMAX[t] if parCFMAX_is_dynamic else parCFMAX
        CFR = parCFR[t] if parCFR_is_dynamic else parCFR
        CWH = parCWH[t] if parCWH_is_dynamic else parCWH
        FC = parFC[t] if parFC_is_dynamic else parFC
        BETA = parBETA[t] if parBETA_is_dynamic else parBETA
        LP = parLP[t] if parLP_is_dynamic else parLP
        BETAET = parBETAET[t] if parBETAET_is_dynamic else parBETAET
        C = parC[t] if parC_is_dynamic else parC
        PERC_par = parPERC[t] if parPERC_is_dynamic else parPERC
        K0 = parK0[t] if parK0_is_dynamic else parK0
        K1 = parK1[t] if parK1_is_dynamic else parK1
        K2 = parK2[t] if parK2_is_dynamic else parK2
        UZL = parUZL[t] if parUZL_is_dynamic else parUZL

        # 降水分离
        temp_diff = T[t] - TT
        RAIN = P[t] * (temp_diff >= 0).float()
        SNOW = P[t] * (temp_diff < 0).float()

        # 积雪模块
        SNOWPACK = SNOWPACK + SNOW
        melt = torch.clamp(CFMAX * temp_diff, min=0.0)
        melt = torch.min(melt, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt

        refreezing = torch.clamp(CFR * CFMAX * (-temp_diff), min=0.0)
        refreezing = torch.min(refreezing, MELTWATER)
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing

        tosoil = torch.clamp(MELTWATER - CWH * SNOWPACK, min=0.0)
        MELTWATER = MELTWATER - tosoil

        # 土壤模块
        soil_wetness = torch.clamp((SM / FC) ** BETA, min=0.0, max=1.0)
        recharge = (RAIN + tosoil) * soil_wetness
        SM = SM + RAIN + tosoil - recharge

        excess = torch.clamp(SM - FC, min=0.0)
        SM = SM - excess

        # 蒸发
        evapfactor = torch.clamp((SM / (LP * FC)) ** BETAET, min=0.0, max=1.0)
        ETact = torch.min(SM, PET[t] * evapfactor)
        SM = torch.clamp(SM - ETact, min=nearzero)

        # 毛管上升
        capillary = torch.min(
            SLZ, C * SLZ * (1.0 - torch.clamp(SM / FC, max=1.0))
        )
        SM = torch.clamp(SM + capillary, min=nearzero)
        SLZ = torch.clamp(SLZ - capillary, min=nearzero)

        # 地下水模块
        SUZ = SUZ + recharge + excess
        PERC = torch.min(SUZ, PERC_par)
        SUZ = SUZ - PERC
        Q0 = K0 * torch.clamp(SUZ - UZL, min=0.0)
        SUZ = SUZ - Q0
        Q1 = K1 * SUZ
        SUZ = SUZ - Q1
        SLZ = torch.clamp(SLZ + PERC, min=0.0)
        Q2 = K2 * SLZ
        SLZ = SLZ - Q2

        # 记录输出
        Qsim_out[t] = Q0 + Q1 + Q2
        Q0_out[t] = Q0
        Q1_out[t] = Q1
        Q2_out[t] = Q2
        AET_out[t] = ETact
        SWE_out[t] = SNOWPACK
        SM_out[t] = SM
        capillary_out[t] = capillary
        recharge_out[t] = recharge
        excs_out[t] = excess
        evapfactor_out[t] = evapfactor
        tosoil_out[t] = tosoil + RAIN
        PERC_out[t] = PERC
        soil_wetness_out[t] = soil_wetness

    return (
        Qsim_out,
        Q0_out,
        Q1_out,
        Q2_out,
        AET_out,
        recharge_out,
        excs_out,
        evapfactor_out,
        tosoil_out,
        PERC_out,
        SWE_out,
        SM_out,
        capillary_out,
        soil_wetness_out,
        SNOWPACK,
        MELTWATER,
        SM,
        SUZ,
        SLZ,
    )


# @torch.jit.script
def shm_timestep_loop(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    dd: torch.Tensor,
    f_thr: torch.Tensor,
    sumax: torch.Tensor,
    beta: torch.Tensor,
    perc: torch.Tensor,
    kf: torch.Tensor,
    ki: torch.Tensor,
    kb: torch.Tensor,
    nearzero: float,
    ss: Optional[torch.Tensor] = None,
    sf: Optional[torch.Tensor] = None,
    su: Optional[torch.Tensor] = None,
    si: Optional[torch.Tensor] = None,
    sb: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    SHM 模型时间步循环（JIT 优化版本）

    Parameters
    ----------
    P : torch.Tensor
        降水，形状: (T, B, E)
    T : torch.Tensor
        温度，形状: (T, B, E)
    PET : torch.Tensor
        潜在蒸发，形状: (T, B, E)
    ss, sf, su, si, sb : torch.Tensor or None
        初始状态，形状: (B, E)。如果为 None，则默认使用零张量。
        ss: 积雪, sf: 快速流蓄水, su: 非饱和带, si: 壤中流蓄水, sb: 基流蓄水
    dd : torch.Tensor
        度日因子，形状: (B, E) 或 (T, B, E)
    f_thr : torch.Tensor
        快速流阈值，形状: (B, E) 或 (T, B, E)
    sumax : torch.Tensor
        非饱和带最大蓄水量，形状: (B, E) 或 (T, B, E)
    beta : torch.Tensor
        非饱和带形状参数，形状: (B, E) 或 (T, B, E)
    perc : torch.Tensor
        下渗比例，形状: (B, E) 或 (T, B, E)
    kf : torch.Tensor
        快速流退水系数，形状: (B, E) 或 (T, B, E)
    ki : torch.Tensor
        壤中流退水系数，形状: (B, E) 或 (T, B, E)
    kb : torch.Tensor
        基流退水系数，形状: (B, E) 或 (T, B, E)
    nearzero : float
        防止除零的小数

    Returns
    -------
    tuple
        包含以下输出:
        - Qsim_out: 总流量 (T, B, E)
        - Qf_out: 快速流 (T, B, E)
        - Qi_out: 壤中流 (T, B, E)
        - Qb_out: 基流 (T, B, E)
        - AET_out: 实际蒸发 (T, B, E)
        - ss_out: 积雪状态 (T, B, E)
        - sf_out: 快速流蓄水状态 (T, B, E)
        - su_out: 非饱和带状态 (T, B, E)
        - si_out: 壤中流蓄水状态 (T, B, E)
        - sb_out: 基流蓄水状态 (T, B, E)
        以及最终状态:
        - ss, sf, su, si, sb: (B, E)
    """
    n_steps = P.shape[0]
    n_grid = P.shape[1]
    nmul = P.shape[2]
    device = P.device

    # default initial states if not provided (shape: (B, E))
    if ss is None:
        ss = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero  
    if sf is None:
        sf = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if su is None:
        su = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if si is None:
        si = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if sb is None:
        sb = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero

    # 常量
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    klu = torch.tensor(
        0.90, dtype=torch.float32, device=device
    )  # 土地利用校正因子
    pwp_ratio = torch.tensor(
        0.8, dtype=torch.float32, device=device
    )  # 永久凋萎点比例

    # 初始化输出张量
    Qsim_out = torch.zeros(
        (n_steps, n_grid, nmul), dtype=torch.float32, device=device
    ) + nearzero
    Qf_out = torch.zeros_like(Qsim_out) + nearzero
    Qi_out = torch.zeros_like(Qsim_out) + nearzero
    Qb_out = torch.zeros_like(Qsim_out) + nearzero
    AET_out = torch.zeros_like(Qsim_out) + nearzero
    ss_out = torch.zeros_like(Qsim_out) + nearzero
    sf_out = torch.zeros_like(Qsim_out) + nearzero
    su_out = torch.zeros_like(Qsim_out) + nearzero
    si_out = torch.zeros_like(Qsim_out) + nearzero
    sb_out = torch.zeros_like(Qsim_out) + nearzero

    # 判断参数是否为动态（通过维度判断）
    dd_is_dynamic = dd.dim() == 3
    f_thr_is_dynamic = f_thr.dim() == 3
    sumax_is_dynamic = sumax.dim() == 3
    beta_is_dynamic = beta.dim() == 3
    perc_is_dynamic = perc.dim() == 3
    kf_is_dynamic = kf.dim() == 3
    ki_is_dynamic = ki.dim() == 3
    kb_is_dynamic = kb.dim() == 3

    for t in range(n_steps):
        # 获取当前时间步参数
        DD = dd[t] if dd_is_dynamic else dd
        F_THR = f_thr[t] if f_thr_is_dynamic else f_thr
        SUMAX = sumax[t] if sumax_is_dynamic else sumax
        BETA = beta[t] if beta_is_dynamic else beta
        PERC = perc[t] if perc_is_dynamic else perc
        KF = kf[t] if kf_is_dynamic else kf
        KI = ki[t] if ki_is_dynamic else ki
        KB = kb[t] if kb_is_dynamic else kb

        # 永久凋萎点
        pwp = pwp_ratio * SUMAX

        # ===== 积雪模块 =====
        # 融雪量（温度 >= 0 时融雪）
        snow_melt = DD * T[t]
        snow_melt = torch.where(T[t] < zero, zero, snow_melt)
        qs_out = torch.minimum(ss, snow_melt)
        ss = ss - qs_out

        # 降雪/降雨分离
        snow_in = torch.where(T[t] < zero, P[t], zero)
        rain_in = torch.where(T[t] >= zero, P[t], zero)
        ss = ss + snow_in

        # 融雪 + 液态降水
        qsp_out = qs_out + rain_in

        # ===== 快速流/非饱和带分离 =====
        qf_in = torch.maximum(zero, qsp_out - F_THR)
        qu_in = torch.minimum(qsp_out, F_THR)

        # ===== 快速流模块 =====
        sf = sf + qf_in
        qf_out = sf / KF
        sf = sf - qf_out

        # ===== 非饱和带模块 =====
        psi = (su / SUMAX) ** BETA
        psi = torch.clamp(psi, min=zero, max=one)
        su_temp = su + qu_in * (one - psi)
        su = torch.minimum(su_temp, SUMAX)
        qu_out = qu_in * psi + torch.maximum(zero, su_temp - SUMAX)

        # ===== 蒸发模块 =====
        ktetha = su / SUMAX
        ktetha = torch.where(su <= pwp, ktetha, one)
        ret = PET[t] * klu * ktetha
        su = torch.maximum(zero, su - ret)

        # ===== 壤中流模块 =====
        qi_in = qu_out * PERC
        si = si + qi_in
        qi_out = si / KI
        si = si - qi_out

        # ===== 基流模块 =====
        qb_in = qu_out * (one - PERC)
        sb = sb + qb_in
        qb_out = sb / KB
        sb = sb - qb_out

        # 记录输出
        Qsim_out[t] = qf_out + qi_out + qb_out
        Qf_out[t] = qf_out
        Qi_out[t] = qi_out
        Qb_out[t] = qb_out
        AET_out[t] = ret
        ss_out[t] = ss
        sf_out[t] = sf
        su_out[t] = su
        si_out[t] = si
        sb_out[t] = sb

    return (
        Qsim_out,
        Qf_out,
        Qi_out,
        Qb_out,
        AET_out,
        ss_out,
        sf_out,
        su_out,
        si_out,
        sb_out,
        ss,
        sf,
        su,
        si,
        sb,
    )


# @torch.jit.script
def exphydro_timestep_loop(
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    数值稳定的 EXP-HYDRO 实现
    """
    n_steps = P.shape[0]
    n_grid = P.shape[1]
    nmul = P.shape[2]
    device = P.device

    # 初始化状态
    if soil_storage is None:
        soil_storage = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if snow_storage is None:
        snow_storage = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero

    # 预分配输出
    qsim_out = torch.zeros((n_steps, n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    et_out = torch.zeros_like(qsim_out) + nearzero
    melt_out = torch.zeros_like(qsim_out) + nearzero
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    # 保证 smax 不会过小导致除法爆炸，虽然有 bounds，为了数值安全再做一次保护
    safe_smax = torch.clamp(smax, min=1.0) # 设为 1.0 或比 nearzero 大一点的值

    for t in range(n_steps):
        p = P[t]
        te = T[t]
        pet = PET[t]

        # 1. 降雨/降雪 分离 (Rain/Snow Partition)
        # 使用 sigmoid 替代硬阈值可以获得更好的梯度（可选，此处保留原逻辑但确保数值安全）
        ps = torch.where(te < mint, p, zero)
        pr = p - ps

        # 2. 积雪融化 (Snow Bucket)
        # melt = min(snow, potential_melt)
        potential_melt = torch.where(te > maxt, ddf * (te - maxt), zero)
        melt = torch.minimum(snow_storage, potential_melt)
        
        # 更新积雪 (无负数风险)
        snow_storage = snow_storage + ps - melt
        
        # 3. 土壤水 (Soil Bucket)
        # 输入水量
        water_in = pr + melt
        
        # --- 关键修改 A: 计算饱和超渗 (Surface Runoff) ---
        # 如果当前水量 + 输入水量 > smax，超出部分直接作为地表径流
        # 注意：这里我们基于 (s_old + water_in) 来判断，比原代码基于 s_old 判断更符合物理过程
        # 原代码逻辑是：如果 s > smax，则 qsurf = s - smax。
        # 这里为了保持原逻辑的风格，但增加数值稳定性：
        
        s = soil_storage
        
        # 为了防止 exp 计算溢出，我们需要对 s 进行逻辑上的截断
        # 只有当 s < smax 时，exp 公式才有效。
        # 我们构造一个 safe_s_for_exp，它永远 <= smax，这样 (smax - s) 永远 >= 0
        # -f * (smax - s) 永远 <= 0，exp 永远在 [0, 1] 之间，杜绝了 Inf
        safe_s_for_exp = torch.minimum(s, safe_smax)
        
        # ET 计算
        # 同样使用 safe_smax 防止除以 0
        # 且限制 s/smax 不超过 1 (虽然 s 可能 > smax，但在 ET 公式里通常意味着最大蒸发)
        saturation_ratio = torch.clamp(s / safe_smax, max=1.0) 
        et_potential = pet * saturation_ratio
        
        # Baseflow (Qsub) 计算
        # 使用 safe_s_for_exp 确保指数项安全
        # 只有 s < smax 时才用 exp，否则 qsub = qmax
        # 通过构造 safe_term，我们避免了 torch.where 的梯度问题
        term_exp = torch.exp(-f * (safe_smax - safe_s_for_exp))
        qsub_potential = qmax * term_exp
        
        # 处理 s > smax 的情况：
        # 如果 s > smax，qsub 应该是 qmax，et 应该是 pet
        # 上面的公式中，如果 s > smax:
        # saturation_ratio 被 clamp 为 1.0 -> et = pet (正确)
        # safe_s_for_exp 被 min 为 smax -> exp(0) = 1 -> qsub = qmax (正确)
        # 这种写法避免了 if/else 和 where 的梯度双算问题
        
        # --- 关键修改 B: 水量平衡限制 (Flux Limiting) ---
        # 防止流出量大于现有水量导致负值
        # 可用水量 = 当前土壤水 + 本时刻入流 (显式格式通常用 s 计算通量，但也可用 s + in)
        # 原代码逻辑是用 s 计算通量。我们这里加上“防止透支”的缩放。
        
        total_outflow_demand = et_potential + qsub_potential
        
        # 真正的流出不能超过 s (假设本时刻入流 water_in 在通量计算后加入，或者同时也参与流出)
        # 为了数值最稳定，通常允许从 (s + water_in) 中流出，或者严格限制只能从 s 流出。
        # 这里采用严格限制：流出 <= s
        scaling_factor = torch.minimum(
            torch.tensor(1.0, device=device),
            s / (total_outflow_demand + nearzero)
        )
        
        et = et_potential * scaling_factor
        qsub = qsub_potential * scaling_factor
        
        # 计算 Qsurf (饱和超渗)
        # 在扣除 ET 和 Qsub 之前还是之后计算？
        # 原代码逻辑：qsurf = s - smax (如果 s > smax)。
        # 这意味着在时间步开始时，超过 smax 的水瞬间变成径流。
        qsurf = torch.relu(s - safe_smax) 
        
        # 更新状态
        # s_new = s + in - out
        # 注意：这里需要仔细处理顺序。如果 qsurf 移除了 s-smax，那么参与 et/qsub 的 s 实际上是 smax。
        # 但为了简化且保持梯度平滑，我们使用上面计算的受限 flux。
        
        # 修正后的状态更新：
        ds = water_in - et - qsub - qsurf
        soil_storage = soil_storage + ds
        
        # 最后的保险：防止数值误差带来的微小负数
        soil_storage = torch.clamp(soil_storage, min=0.0)

        # 记录输出
        qsim_out[t] = qsub + qsurf
        et_out[t] = et
        melt_out[t] = melt

    return qsim_out, et_out, melt_out, snow_storage, soil_storage


# @torch.jit.script
def hymod_timestep_loop(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    Tth: torch.Tensor,
    Tb: torch.Tensor,
    DDF: torch.Tensor,
    Huz: torch.Tensor,
    Cpar: torch.Tensor,
    B: torch.Tensor,
    Kv: torch.Tensor,
    alpha: torch.Tensor,
    Kq: torch.Tensor,
    Ks: torch.Tensor,
    nearzero: float,
    snow_store: Optional[torch.Tensor] = None,
    XHuz: Optional[torch.Tensor] = None,
    XCuz: Optional[torch.Tensor] = None,
    Xs: Optional[torch.Tensor] = None,
    Xq: Optional[torch.Tensor] = None,
):
    """
    数值稳定的 HyMod 实现
    """
    n_steps = P.shape[0]
    n_grid = P.shape[1]
    nmul = P.shape[2]
    device = P.device

    # 1. 初始化状态
    if snow_store is None:
        snow_store = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if XHuz is None:
        XHuz = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if XCuz is None: # 实际上 XCuz 是 XHuz 的派生变量，通常只需要维护一个，这里为了兼容保持
        XCuz = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if Xs is None:
        Xs = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero
    if Xq is None:
        Xq = torch.zeros((n_grid, nmul), dtype=torch.float32, device=device) + nearzero

    # 2. 预分配输出
    # ... (保持原样)
    snow_out = torch.zeros_like(P)
    melt_out = torch.zeros_like(P)
    effPrecip_out = torch.zeros_like(P)
    PE_out = torch.zeros_like(P)
    OV_out = torch.zeros_like(P)
    AE_out = torch.zeros_like(P)
    OV1_out = torch.zeros_like(P)
    OV2_out = torch.zeros_like(P)
    Qq_out = torch.zeros_like(P)
    Qs_out = torch.zeros_like(P)
    Q_out = torch.zeros_like(P)

    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    epsilon = 1e-5  # 防止幂运算底数为0导致的梯度爆炸

    # 预处理参数，防止除零
    safe_Huz = torch.clamp(Huz, min=1.0)
    safe_Cpar = torch.clamp(Cpar, min=1.0)
    
    # 修正 Ks 的使用：Ks 是滞留时间。如果 Ks -> 0，流速 -> 无穷大。
    # 限制 Ks 最小为 0.1 (或其他合理的时间步长单位)，防止除以极小值
    safe_Ks = torch.clamp(Ks, min=0.1) 

    for t in range(n_steps):
        # --- Snow Module ---
        # 使用 sigmoid 使得温度阈值平滑化是更好的选择，但为了保持逻辑一致，先保留硬阈值
        is_snow = (T[t] < Tth).float() 
        snow_in = P[t] * is_snow
        rain_in = P[t] * (1.0 - is_snow)

        snow_store = snow_store + snow_in

        # 计算潜在融雪，并限制不能超过现有积雪
        potential_melt = torch.where(
            T[t] > Tb, 
            DDF * (T[t] - Tb), 
            zero
        )
        melt = torch.minimum(snow_store, torch.relu(potential_melt)) # relu防止负数
        snow_store = snow_store - melt
        
        # 有效降水
        Qout = rain_in + melt

        # --- PDM Soil Moisture Module ---
        
        # 计算当前的相对饱和度
        # frac = XHuz / safe_Huz
        # 关键修正 A: 限制 frac 在 [0, 1-eps] 之间，防止 pow(0) 导致反向传播梯度 NaN
        # 同时也防止 XHuz > Huz 的情况
        frac = torch.clamp(XHuz / safe_Huz, min=0.0, max=1.0 - epsilon)
        
        # 计算 Cbeg (Capacity at beginning)
        # 幂运算安全化：确保底数非负且不为0
        base_beg = torch.clamp(one - frac, min=epsilon)
        Cbeg = safe_Cpar * (one - torch.pow(base_beg, one + B))

        # 计算饱和超渗 OV2
        # 注意：这里逻辑上假设瞬间饱和。
        OV2 = torch.relu(Qout + XHuz - safe_Huz)
        
        # 实际入渗量
        PPinf = Qout - OV2
        
        # 更新后的深度 Hint
        # Hint = min(Huz, XHuz + PPinf)
        # 由于上面已经减去了 OV2，理论上 XHuz + PPinf <= Huz
        Hint = XHuz + PPinf
        
        # 计算对应的 Cint
        frac_int = torch.clamp(Hint / safe_Huz, min=0.0, max=1.0 - epsilon)
        base_int = torch.clamp(one - frac_int, min=epsilon)
        Cint = safe_Cpar * (one - torch.pow(base_int, one + B))

        # 计算入渗超渗 OV1
        OV1 = torch.relu(PPinf + Cbeg - Cint)
        OV = OV1 + OV2

        # --- Evapotranspiration ---
        # AE = min(Cint, potential_ET)
        # Kv 应当是 [0, 1] 之间的效率因子
        et_efficiency = torch.clamp(Cint / safe_Cpar, min=0.0, max=1.0)
        potential_evap = et_efficiency * PET[t] * Kv
        AE = torch.minimum(Cint, potential_evap)

        # --- Update Soil State ---
        # 更新 XCuz (Capacity unit)
        # 先限制下界为 0 (使用 relu 或者 maximum(x, zero))
        XCuz_temp = torch.relu(Cint - AE)
        # 再限制上界为 safe_Cpar (使用 minimum 处理张量边界)
        XCuz_new = torch.minimum(XCuz_temp, safe_Cpar)
        
        # 关键修正 B: 逆运算求 XHuz (Depth unit)
        # 公式: XHuz = Huz * (1 - (1 - XCuz/Cpar)^(1/(1+B)))
        # 这里的指数 1/(1+B) < 1。如果底数 (1 - XCuz/Cpar) 接近 0，梯度爆炸。
        # 因此，需要限制 (1 - XCuz/Cpar) 不小于 epsilon
        
        ratio_c = XCuz_new / safe_Cpar
        base_inv = torch.clamp(one - ratio_c, min=epsilon, max=1.0)
        
        exponent = one / (one + B)
        XHuz = safe_Huz * (one - torch.pow(base_inv, exponent))

        # --- Routing Module ---
        
        # 分流
        q_in = alpha * OV
        s_in = (one - alpha) * OV

        # Slow Flow (Linear Reservoir)
        # 原代码: Qs = Xs / Ks. 
        # 问题: 显式欧拉法如果 Ks < 1，会导致流出 > 存量。且 Ks=0 会崩溃。
        # 修正: 使用解析解衰减因子，或者限制流出量。
        # 解析解: outflow = S * (1 - exp(-1/K))
        
        # 确保 Ks 不为0 (上面已定义 safe_Ks)
        k_rate = one / safe_Ks
        decay_factor = torch.exp(-k_rate) # 这一步流出后的剩余比例
        
        # 先加入输入
        Xs = Xs + s_in
        
        # 计算流出 (Xs * (1 - decay))
        Qs = Xs * (one - decay_factor)
        
        # 更新状态
        Xs = Xs - Qs 
        # 再次clamp防止浮点误差导致负数
        Xs = torch.clamp(Xs, min=0.0)

        # Quick Flow (Linear Reservoir)
        # Kq 是速率常数 [0, 1] 还是滞留时间？
        # 通常 HyMod 中快流也是滞留时间。但参数范围 Kq [0, 1] 暗示它可能是速率 k (Q = kS)。
        # 如果 Kq 是速率 (Q = Kq * Xq)，必须保证 Kq <= 1。
        # 如果 Kq 是时间 (Q = Xq / Kq)，同慢流处理。
        # 根据原代码 `Qq = Kq * Xq`，假设它是速率系数 (Recession constant)。
        # 为防止透支，Qq 不能超过 Xq。
        
        Xq = Xq + q_in
        
        # 限制流出系数不超过 1.0
        safe_Kq = torch.clamp(Kq, max=1.0)
        Qq = safe_Kq * Xq
        
        Xq = Xq - Qq
        Xq = torch.clamp(Xq, min=0.0)

        Q = Qq + Qs

        # Record
        snow_out[t] = snow_in
        melt_out[t] = melt
        effPrecip_out[t] = Qout
        PE_out[t] = PET[t]
        OV_out[t] = OV
        AE_out[t] = AE
        OV1_out[t] = OV1
        OV2_out[t] = OV2
        Qq_out[t] = Qq
        Qs_out[t] = Qs
        Q_out[t] = Q

    return (
        Q_out, snow_out, melt_out, effPrecip_out, PE_out,
        OV_out, AE_out, OV1_out, OV2_out, Qq_out, Qs_out,
        snow_store, XHuz, XCuz, Xs, Xq,
    )
