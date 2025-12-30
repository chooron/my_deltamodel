import torch
from typing import Optional, Tuple

# 假设核心组件已经存在，后续再补充具体实现
from ..marrmot.evap import evap_1, evap_2
from ..marrmot.interception import interception_1
from ..marrmot.infiltration import infiltration_1, infiltration_2
from ..marrmot.interflow import interflow_1
from ..marrmot.recharge import recharge_1
from ..marrmot.saturation import saturation_1
from ..marrmot.depression import depression_1
from ..marrmot.exchange import exchange_1, exchange_3
from ..marrmot.baseflow import baseflow_1

# 参数取值范围字典 (匹配 MATLAB m_36_modhydrolog_15p_5s)
MODHYDROLOG_PARAMS_BOUNDS = {
    "insc": [0.0, 5.0],           # Maximum interception capacity [mm]
    "coeff": [0.0, 600.0],        # Maximum infiltration loss parameter [mm]
    "sq": [0.0, 15.0],            # Infiltration loss exponent [-]
    "smsc": [1.0, 2000.0],        # Maximum soil moisture capacity [mm]
    "sub": [0.0, 1.0],            # Proportionality constant [-]
    "crak": [0.0, 1.0],           # Proportionality constant [-]
    "em": [0.0, 20.0],            # Maximum plant-controlled evap rate [mm/d]
    "dsc": [0.0, 50.0],           # Maximum depression capacity [mm]
    "ads": [0.0, 1.0],            # Land fraction functioning as depression storage [-]
    "md": [0.99, 1.0],            # Depression storage parameter [-]
    "vcond": [0.0, 0.5],          # Leakage coefficient [mm/d]
    "dlev": [-10.0, 10.0],        # Datum around which groundwater fluctuates relative to river bed [mm]
    "k1": [0.0, 1.0],             # Flow exchange parameter [d-1]
    "k2": [0.0, 1.0],             # Flow exchange parameter [d-1]
    "k3": [0.0, 100.0],           # Flow exchange parameter [d-1]
}

# 参数物理描述
MODHYDROLOG_PARAMS_DESC = {
    "insc": "Maximum interception capacity [mm]",
    "coeff": "Maximum infiltration loss parameter [mm]",
    "sq": "Infiltration loss exponent [-]",
    "smsc": "Maximum soil moisture capacity [mm]",
    "sub": "Proportionality constant [-]",
    "crak": "Proportionality constant [-]",
    "em": "Maximum plant-controlled evap rate [mm/d]",
    "dsc": "Maximum depression capacity [mm]",
    "ads": "Land fraction functioning as depression storage [-]",
    "md": "Depression storage parameter [-]",
    "vcond": "Leakage coefficient [mm/d]",
    "dlev": "Datum around which groundwater fluctuates relative to river bed [mm]",
    "k1": "Flow exchange parameter [d-1]",
    "k2": "Flow exchange parameter [d-1]",
    "k3": "Flow exchange parameter [d-1]",
}

def create_initial_state(
    n_grid: int, 
    nmul: int, 
    device: torch.device, 
    nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for MODHYDROLOG model.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5

def modhydrolog_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序必须与字典键顺序一致
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    em: torch.Tensor,
    dsc: torch.Tensor,
    ads: torch.Tensor,
    md: torch.Tensor,
    vcond: torch.Tensor,
    dlev: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MODHYDROLOG 模型单步计算.
    """
    # --- 1. 拦截层 (S1) ---
    S1 = S1 + P                                         # Step 1: Inflow
    flux_EXC = interception_1(P, S1, insc)              # Step 2: Fast (Overflow)
    S1 = torch.clamp(S1 - flux_EXC, min=nearzero)
    flux_Ei = evap_1(S1, PET)                           # Step 3: Evap
    S1 = torch.clamp(S1 - flux_Ei, min=nearzero)

    # --- 2. 产流与入渗分离 ---
    # 基于 EXC 计算潜在入渗和地表径流
    flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    
    # 壤中流和补给计算 (基于 S2 的中间量)
    flux_INT = interflow_1(sub, S2, smsc, flux_INF)
    flux_REC = recharge_1(crak, S2, smsc, flux_INF - flux_INT)
    flux_SMF = flux_INF - flux_INT - flux_REC           # 真正进入 S2 的水

    # --- 3. 洼地储蓄 (S3) ---
    flux_TRAP = depression_1(ads, md, S3, dsc, flux_RUN)
    S3 = S3 + flux_TRAP
    flux_Ed = evap_1(S3, ads * PET)                     # 洼地蒸发
    S3 = torch.clamp(S3 - flux_Ed, min=nearzero)
    flux_DINF = ads * infiltration_2(coeff, sq, S2, smsc, flux_SMF, S3)
    S3 = torch.clamp(S3 - flux_DINF, min=nearzero)
    
    flux_SRUN = flux_RUN - flux_TRAP                    # 绕过洼地的地表径流

    # --- 4. 土壤水分层 (S2) ---
    S2 = S2 + flux_SMF + flux_DINF                      # Inflow
    flux_GWF = saturation_1(flux_SMF + flux_DINF, S2, smsc) # Step 2: Fast (Saturation overflow)
    S2 = torch.clamp(S2 - flux_GWF, min=nearzero)
    flux_Et = evap_2(em, S2, smsc, PET)                 # Step 3: Evap
    S2 = torch.clamp(S2 - flux_Et, min=nearzero)

    # --- 5. 地下水层 (S4) ---
    S4 = S4 + flux_REC + flux_GWF                       # Inflow (Slow + Saturation)
    flux_FLOW = exchange_1(k1, k2, k3, S4, flux_SRUN)   # 与河道的交换
    S4 = S4 - flux_FLOW                                 # 注意：S4 在 MATLAB 中可以为负
    flux_SEEP = exchange_3(vcond, S4, dlev)             # 地下水排泄/补给
    S4 = S4 - flux_SEEP

    # --- 6. 出流层 (S5) ---
    S5 = S5 + flux_SRUN + flux_INT + flux_FLOW          # Step 1: Inflow
    flux_Q = baseflow_1(torch.ones_like(S5), S5)        # Step 4: Slow (Discharge)
    S5 = torch.clamp(S5 - flux_Q, min=nearzero)

    # --- 7. 输出聚合 ---
    Q = flux_Q
    Ea = flux_Ei + flux_Et + flux_Ed

    return Q, Ea, S1, S2, S3, S4, S5
