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