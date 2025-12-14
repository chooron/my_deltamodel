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


@torch.jit.script
def hbv_timestep_loop(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    SNOWPACK: torch.Tensor,
    MELTWATER: torch.Tensor,
    SM: torch.Tensor,
    SUZ: torch.Tensor,
    SLZ: torch.Tensor,
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    HBV 模型时间步循环（JIT 优化版本）
    
    Parameters
    ----------
    P, T, PET : torch.Tensor
        驱动数据，形状: (T, B, E)
        T=时间步数, B=流域数, E=专家数(nmul)
    SNOWPACK, MELTWATER, SM, SUZ, SLZ : torch.Tensor
        初始状态，形状: (B, E)
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
    
    # 初始化输出张量
    Qsim_out = torch.zeros((n_steps, n_grid, nmul), dtype=torch.float32, device=device)
    Q0_out = torch.zeros_like(Qsim_out)
    Q1_out = torch.zeros_like(Qsim_out)
    Q2_out = torch.zeros_like(Qsim_out)
    AET_out = torch.zeros_like(Qsim_out)
    recharge_out = torch.zeros_like(Qsim_out)
    excs_out = torch.zeros_like(Qsim_out)
    evapfactor_out = torch.zeros_like(Qsim_out)
    tosoil_out = torch.zeros_like(Qsim_out)
    PERC_out = torch.zeros_like(Qsim_out)
    SWE_out = torch.zeros_like(Qsim_out)
    SM_out = torch.zeros_like(Qsim_out)
    capillary_out = torch.zeros_like(Qsim_out)
    soil_wetness_out = torch.zeros_like(Qsim_out)
    
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
        evapfactor = torch.clamp(
            (SM / (LP * FC)) ** BETAET, min=0.0, max=1.0
        )
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
        Qsim_out, Q0_out, Q1_out, Q2_out, AET_out, recharge_out, excs_out,
        evapfactor_out, tosoil_out, PERC_out, SWE_out, SM_out, capillary_out,
        soil_wetness_out, SNOWPACK, MELTWATER, SM, SUZ, SLZ
    )


@torch.jit.script
def shm_timestep_loop(
    P: torch.Tensor,
    T_min: torch.Tensor,
    T_max: torch.Tensor,
    PET: torch.Tensor,
    ss: torch.Tensor,
    sf: torch.Tensor,
    su: torch.Tensor,
    si: torch.Tensor,
    sb: torch.Tensor,
    dd: torch.Tensor,
    f_thr: torch.Tensor,
    sumax: torch.Tensor,
    beta: torch.Tensor,
    perc: torch.Tensor,
    kf: torch.Tensor,
    ki: torch.Tensor,
    kb: torch.Tensor,
    nearzero: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, 
]:
    """
    SHM 模型时间步循环（JIT 优化版本）
    
    Parameters
    ----------
    P : torch.Tensor
        降水，形状: (T, B, E)
    T_min, T_max : torch.Tensor
        最低/最高温度，形状: (T, B, E)
    PET : torch.Tensor
        潜在蒸发，形状: (T, B, E)
    ss, sf, su, si, sb : torch.Tensor
        初始状态，形状: (B, E)
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
    
    # 常量
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    klu = torch.tensor(0.90, dtype=torch.float32, device=device)  # 土地利用校正因子
    pwp_ratio = torch.tensor(0.8, dtype=torch.float32, device=device)  # 永久凋萎点比例
    
    # 初始化输出张量
    Qsim_out = torch.zeros((n_steps, n_grid, nmul), dtype=torch.float32, device=device)
    Qf_out = torch.zeros_like(Qsim_out)
    Qi_out = torch.zeros_like(Qsim_out)
    Qb_out = torch.zeros_like(Qsim_out)
    AET_out = torch.zeros_like(Qsim_out)
    ss_out = torch.zeros_like(Qsim_out)
    sf_out = torch.zeros_like(Qsim_out)
    su_out = torch.zeros_like(Qsim_out)
    si_out = torch.zeros_like(Qsim_out)
    sb_out = torch.zeros_like(Qsim_out)
    
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
        
        # 计算平均温度
        t_mean = (T_min[t] + T_max[t]) / 2.0
        
        # 永久凋萎点
        pwp = pwp_ratio * SUMAX
        
        # ===== 积雪模块 =====
        # 融雪量（温度 >= 0 时融雪）
        snow_melt = DD * t_mean
        snow_melt = torch.where(t_mean < zero, zero, snow_melt)
        qs_out = torch.minimum(ss, snow_melt)
        ss = ss - qs_out
        
        # 降雪/降雨分离
        snow_in = torch.where(t_mean < zero, P[t], zero)
        rain_in = torch.where(t_mean >= zero, P[t], zero)
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
        Qsim_out, Qf_out, Qi_out, Qb_out, AET_out,
        ss_out, sf_out, su_out, si_out, sb_out,
        ss, sf, su, si, sb
    )


# @torch.jit.script
# def cfe_timestep_loop(
#     P: torch.Tensor,
#     PET: torch.Tensor,
#     soil_storage: torch.Tensor,
#     gw_storage: torch.Tensor,
#     # Schaake partitioning parameters
#     Schaake_adjusted_magic_constant_by_soil_type: torch.Tensor,
#     # Soil reservoir parameters
#     smcmax: torch.Tensor,
#     soil_depth: torch.Tensor,
#     wltsmc: torch.Tensor,
#     satpsi: torch.Tensor,
#     bb: torch.Tensor,
#     mult: torch.Tensor,
#     satdk: torch.Tensor,
#     slop: torch.Tensor,
#     # Groundwater parameters
#     max_gw_storage: torch.Tensor,
#     Cgw: torch.Tensor,
#     expon: torch.Tensor,
#     # Nash cascade parameters
#     K_nash: torch.Tensor,
#     nash_storage: torch.Tensor,
#     num_nash_reservoirs: int,
#     nearzero: float,
#     timestep_d: float,
# ) -> tuple[
#     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
#     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
#     torch.Tensor, torch.Tensor,
# ]:
#     """
#     CFE 模型时间步循环（JIT 优化版本，Classic 模式）
    
#     Parameters
#     ----------
#     P : torch.Tensor
#         降水，形状: (T, B, E)
#     PET : torch.Tensor
#         潜在蒸发，形状: (T, B, E)
#     soil_storage : torch.Tensor
#         土壤蓄水初始状态，形状: (B, E)
#     gw_storage : torch.Tensor
#         地下水蓄水初始状态，形状: (B, E)
#     Schaake_adjusted_magic_constant_by_soil_type : torch.Tensor
#         Schaake 参数，形状: (B, E) 或 (T, B, E)
#     smcmax : torch.Tensor
#         最大土壤含水量（孔隙度），形状: (B, E) 或 (T, B, E)
#     soil_depth : torch.Tensor
#         土壤深度 [m]，形状: (B, E) 或 (T, B, E)
#     wltsmc : torch.Tensor
#         凋萎点含水量，形状: (B, E) 或 (T, B, E)
#     satpsi : torch.Tensor
#         饱和水势，形状: (B, E) 或 (T, B, E)
#     bb : torch.Tensor
#         土壤水力特性参数 b，形状: (B, E) 或 (T, B, E)
#     mult : torch.Tensor
#         渗透乘数，形状: (B, E) 或 (T, B, E)
#     satdk : torch.Tensor
#         饱和导水率，形状: (B, E) 或 (T, B, E)
#     slop : torch.Tensor
#         坡度，形状: (B, E) 或 (T, B, E)
#     max_gw_storage : torch.Tensor
#         最大地下水蓄水量，形状: (B, E) 或 (T, B, E)
#     Cgw : torch.Tensor
#         地下水出流系数，形状: (B, E) 或 (T, B, E)
#     expon : torch.Tensor
#         地下水出流指数，形状: (B, E) 或 (T, B, E)
#     K_nash : torch.Tensor
#         Nash 退水系数，形状: (B, E) 或 (T, B, E)
#     nash_storage : torch.Tensor
#         Nash 级联蓄水初始状态，形状: (B, E, num_nash)
#     num_nash_reservoirs : int
#         Nash 级联水库数量
#     nearzero : float
#         防止除零的小数
#     timestep_d : float
#         时间步长（天）
        
#     Returns
#     -------
#     tuple
#         包含各输出变量和最终状态
#     """
#     n_steps = P.shape[0]
#     n_grid = P.shape[1]
#     nmul = P.shape[2]
#     device = P.device
    
#     # 常量
#     zero = torch.tensor(0.0, dtype=torch.float32, device=device)
#     one = torch.tensor(1.0, dtype=torch.float32, device=device)
    
#     # 初始化输出张量
#     Qsim_out = torch.zeros((n_steps, n_grid, nmul), dtype=torch.float32, device=device)
#     Qsurf_out = torch.zeros_like(Qsim_out)
#     Qlat_out = torch.zeros_like(Qsim_out)
#     Qgw_out = torch.zeros_like(Qsim_out)
#     AET_out = torch.zeros_like(Qsim_out)
#     soil_storage_out = torch.zeros_like(Qsim_out)
#     gw_storage_out = torch.zeros_like(Qsim_out)
#     infiltration_out = torch.zeros_like(Qsim_out)
#     percolation_out = torch.zeros_like(Qsim_out)
#     runoff_out = torch.zeros_like(Qsim_out)
    
#     # 判断参数是否为动态（通过维度判断）
#     schaake_is_dynamic = Schaake_adjusted_magic_constant_by_soil_type.dim() == 3
#     smcmax_is_dynamic = smcmax.dim() == 3
#     soil_depth_is_dynamic = soil_depth.dim() == 3
#     wltsmc_is_dynamic = wltsmc.dim() == 3
#     satpsi_is_dynamic = satpsi.dim() == 3
#     bb_is_dynamic = bb.dim() == 3
#     mult_is_dynamic = mult.dim() == 3
#     satdk_is_dynamic = satdk.dim() == 3
#     slop_is_dynamic = slop.dim() == 3
#     max_gw_storage_is_dynamic = max_gw_storage.dim() == 3
#     Cgw_is_dynamic = Cgw.dim() == 3
#     expon_is_dynamic = expon.dim() == 3
#     K_nash_is_dynamic = K_nash.dim() == 3
    
#     for t in range(n_steps):
#         # 获取当前时间步参数
#         SCHAAKE = Schaake_adjusted_magic_constant_by_soil_type[t] if schaake_is_dynamic else Schaake_adjusted_magic_constant_by_soil_type
#         SMCMAX = smcmax[t] if smcmax_is_dynamic else smcmax
#         SOIL_DEPTH = soil_depth[t] if soil_depth_is_dynamic else soil_depth
#         WLTSMC = wltsmc[t] if wltsmc_is_dynamic else wltsmc
#         SATPSI = satpsi[t] if satpsi_is_dynamic else satpsi
#         BB = bb[t] if bb_is_dynamic else bb
#         MULT = mult[t] if mult_is_dynamic else mult
#         SATDK = satdk[t] if satdk_is_dynamic else satdk
#         SLOP = slop[t] if slop_is_dynamic else slop
#         MAX_GW = max_gw_storage[t] if max_gw_storage_is_dynamic else max_gw_storage
#         CGW = Cgw[t] if Cgw_is_dynamic else Cgw
#         EXPON = expon[t] if expon_is_dynamic else expon
#         K_NASH = K_nash[t] if K_nash_is_dynamic else K_nash
        
#         # 计算土壤参数
#         storage_max_m = SMCMAX * SOIL_DEPTH  # 最大蓄水量
#         storage_threshold_primary_m = (SMCMAX - WLTSMC) * SOIL_DEPTH  # 田间持水量阈值
#         wilting_point_m = WLTSMC * SOIL_DEPTH  # 凋萎点
        
#         # 计算渗透和侧向流系数
#         # coeff_primary = Kperc (percolation)
#         # coeff_secondary = Klf (lateral flow)
#         coeff_primary = SATDK * MULT  # 渗透系数
#         coeff_secondary = SATDK * SLOP  # 侧向流系数
        
#         # ===== ET from rainfall =====
#         rainfall = P[t].clone()
#         pet = PET[t].clone()
        
#         # 如果有降雨，先从降雨中满足 ET
#         actual_et_from_rain = torch.where(
#             rainfall > pet,
#             pet,  # 降雨 > PET，AET = PET
#             rainfall  # 降雨 < PET，AET = 降雨
#         )
#         rainfall = rainfall - actual_et_from_rain
#         reduced_pet = pet - actual_et_from_rain
        
#         # ===== Soil moisture deficit =====
#         soil_deficit = storage_max_m - soil_storage
        
#         # ===== Schaake partitioning =====
#         # 计算入渗能力
#         exp_term = torch.exp(-SCHAAKE * timestep_d)
#         Ic = soil_deficit * (one - exp_term)
#         infilt = torch.where(
#             rainfall > zero,
#             rainfall * Ic / (rainfall + Ic + nearzero),
#             zero
#         )
#         surface_runoff = torch.maximum(zero, rainfall - infilt)
        
#         # ===== Adjust runoff and infiltration for saturation excess =====
#         excess_infil = torch.maximum(zero, infilt - soil_deficit)
#         surface_runoff = surface_runoff + excess_infil
#         infilt = infilt - excess_infil
        
#         # ===== Add infiltration to soil storage =====
#         soil_storage = soil_storage + infilt
        
#         # ===== ET from soil (Budyko type) =====
#         # 当土壤水 > 凋萎点时，可以蒸发
#         storage_above_wp = soil_storage - wilting_point_m
#         storage_ratio_et = torch.clamp(
#             storage_above_wp / (storage_threshold_primary_m - wilting_point_m + nearzero),
#             min=zero, max=one
#         )
#         actual_et_from_soil = torch.where(
#             soil_storage > wilting_point_m,
#             torch.minimum(reduced_pet * storage_ratio_et, storage_above_wp),
#             zero
#         )
#         soil_storage = soil_storage - actual_et_from_soil
        
#         # ===== Soil reservoir flux (percolation + lateral flow) =====
#         storage_above_threshold = soil_storage - storage_threshold_primary_m
#         storage_diff = storage_max_m - storage_threshold_primary_m + nearzero
#         storage_ratio = torch.clamp(storage_above_threshold / storage_diff, min=zero, max=one)
        
#         # 渗透通量
#         perc_flux = torch.where(
#             storage_above_threshold > zero,
#             torch.minimum(coeff_primary * storage_ratio, storage_above_threshold),
#             zero
#         )
#         soil_storage = soil_storage - perc_flux
        
#         # 侧向流通量
#         storage_above_threshold = soil_storage - storage_threshold_primary_m
#         lat_flux = torch.where(
#             storage_above_threshold > zero,
#             torch.minimum(coeff_secondary * storage_ratio, storage_above_threshold),
#             zero
#         )
#         soil_storage = soil_storage - lat_flux
        
#         # ===== Groundwater reservoir =====
#         gw_deficit = MAX_GW - gw_storage
        
#         # 渗透进入地下水（考虑容量限制）
#         perc_to_gw = torch.minimum(perc_flux, gw_deficit)
#         overflow_to_runoff = perc_flux - perc_to_gw
#         surface_runoff = surface_runoff + overflow_to_runoff
#         gw_storage = gw_storage + perc_to_gw
        
#         # 地下水出流（指数形式）
#         gw_flux = torch.minimum(
#             CGW * (torch.exp(EXPON * gw_storage / MAX_GW) - one),
#             gw_storage
#         )
#         gw_storage = gw_storage - gw_flux
        
#         # ===== Nash cascade for lateral flow =====
#         # 简化的 Nash 级联
#         nash_out = zero
#         prev_q_out = zero  # 初始化前一水库出流
#         for n in range(num_nash_reservoirs):
#             # 从当前水库出流
#             q_out = K_NASH * nash_storage[:, :, n]
#             nash_storage[:, :, n] = nash_storage[:, :, n] - q_out
            
#             if n == 0:
#                 # 第一个水库接收侧向流
#                 nash_storage[:, :, n] = nash_storage[:, :, n] + lat_flux
#             else:
#                 # 后续水库接收上一个水库的出流
#                 nash_storage[:, :, n] = nash_storage[:, :, n] + prev_q_out
            
#             prev_q_out = q_out
        
#         nash_out = q_out  # 最后一个水库的出流
        
#         # ===== Total discharge =====
#         total_q = surface_runoff + nash_out + gw_flux
#         total_et = actual_et_from_rain + actual_et_from_soil
        
#         # 记录输出
#         Qsim_out[t] = total_q
#         Qsurf_out[t] = surface_runoff
#         Qlat_out[t] = nash_out
#         Qgw_out[t] = gw_flux
#         AET_out[t] = total_et
#         soil_storage_out[t] = soil_storage
#         gw_storage_out[t] = gw_storage
#         infiltration_out[t] = infilt
#         percolation_out[t] = perc_flux
#         runoff_out[t] = surface_runoff
    
#     return (
#         Qsim_out, Qsurf_out, Qlat_out, Qgw_out, AET_out,
#         soil_storage_out, gw_storage_out, infiltration_out, percolation_out, runoff_out,
#         soil_storage, gw_storage
#     )


@torch.jit.script
def abcd_timestep_loop(
    P: torch.Tensor,
    PET: torch.Tensor,
    S: torch.Tensor,
    G: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    nearzero: float,
):
    """
    ABCD 模型时间步循环（JIT 优化版本）
    
    ABCD 模型是一个简洁的四参数概念式水文模型（Thomas, 1981）。
    
    模型结构:
    - 土壤水分模块: 基于参数 a, b 计算有效水分和蒸发
    - 地下水模块: 基于参数 c, d 计算补给和基流
    
    Parameters
    ----------
    P : torch.Tensor
        降水，形状: (T, B, E)
    PET : torch.Tensor
        潜在蒸散发，形状: (T, B, E)
    S : torch.Tensor
        初始土壤蓄水，形状: (B, E)
    G : torch.Tensor
        初始地下水蓄水，形状: (B, E)
    a : torch.Tensor
        参数 a - 控制径流产生的倾向，形状: (B, E) 或 (T, B, E)
    b : torch.Tensor
        参数 b - 土壤蓄水容量上限，形状: (B, E) 或 (T, B, E)
    c : torch.Tensor
        参数 c - 地下水补给比例，形状: (B, E) 或 (T, B, E)
    d : torch.Tensor
        参数 d - 地下水退水系数，形状: (B, E) 或 (T, B, E)
    nearzero : float
        防止除零的极小值
        
    Returns
    -------
    tuple
        (Qsim_out, Qsurf_out, Qgw_out, AET_out, 
         S_out, G_out, W_out, Y_out,
         S_final, G_final)
    """
    n_steps = P.shape[0]
    n_grid = P.shape[1]
    nmul = P.shape[2]
    
    # 判断参数是否为动态参数
    a_dynamic = a.dim() == 3
    b_dynamic = b.dim() == 3
    c_dynamic = c.dim() == 3
    d_dynamic = d.dim() == 3
    
    # 初始化输出张量
    Qsim_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    Qsurf_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    Qgw_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    AET_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    S_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    G_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    W_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    Y_out = torch.zeros((n_steps, n_grid, nmul), dtype=P.dtype, device=P.device)
    
    one = torch.ones_like(S)
    
    for t in range(n_steps):
        # 获取当前时间步的驱动数据
        p_t = P[t]   # 降水
        pet_t = PET[t]  # 潜在蒸散发
        
        # 获取当前时间步的参数
        a_t = a[t] if a_dynamic else a
        b_t = b[t] if b_dynamic else b
        c_t = c[t] if c_dynamic else c
        d_t = d[t] if d_dynamic else d
        
        # ===== 土壤水分模块 =====
        # W = 可用水分 = 降水 + 前期土壤蓄水
        W = p_t + S
        
        # Y = 蒸发机会 (evapotranspiration opportunity)
        # Y = (W + b) / 2a - sqrt[((W + b) / 2a)^2 - Wb/a]
        # 简化形式，避免数值问题
        Wb_a = W * b_t / (a_t + nearzero)
        half_W_b_a = (W + b_t) / (2.0 * a_t + nearzero)
        
        # 确保根号内非负
        sqrt_term = half_W_b_a * half_W_b_a - Wb_a
        sqrt_term = torch.clamp(sqrt_term, min=0.0)
        
        Y = half_W_b_a - torch.sqrt(sqrt_term)
        Y = torch.clamp(Y, min=0.0, max=W)  # Y 不能超过 W
        
        # ===== 蒸散发计算 =====
        # 实际蒸散发 = Y * (1 - exp(-PET/b))
        exp_term = torch.exp(-pet_t / (b_t + nearzero))
        AET = Y * (one - exp_term)
        AET = torch.clamp(AET, min=0.0, max=pet_t)  # 实际蒸发不超过潜在蒸发
        
        # ===== 土壤蓄水更新 =====
        # S_new = Y - AET
        S_new = Y - AET
        S_new = torch.clamp(S_new, min=nearzero)
        
        # ===== 径流生成 =====
        # 总径流机会 = W - Y
        runoff_opportunity = W - Y
        runoff_opportunity = torch.clamp(runoff_opportunity, min=0.0)
        
        # 地下水补给 = c * (W - Y)
        recharge = c_t * runoff_opportunity
        
        # 直接径流 = (1 - c) * (W - Y)
        direct_runoff = (one - c_t) * runoff_opportunity
        
        # ===== 地下水模块 =====
        # G_new = G + recharge
        G_new = G + recharge
        
        # 基流 = d * G_new
        baseflow = d_t * G_new
        
        # 更新地下水蓄水
        G_new = G_new - baseflow
        G_new = torch.clamp(G_new, min=nearzero)
        
        # ===== 总径流 =====
        total_runoff = direct_runoff + baseflow
        
        # 更新状态
        S = S_new
        G = G_new
        
        # 记录输出
        Qsim_out[t] = total_runoff
        Qsurf_out[t] = direct_runoff
        Qgw_out[t] = baseflow
        AET_out[t] = AET
        S_out[t] = S
        G_out[t] = G
        W_out[t] = W
        Y_out[t] = Y
    
    return (
        Qsim_out, Qsurf_out, Qgw_out, AET_out,
        S_out, G_out, W_out, Y_out,
        S, G
    )
