import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# 假设核心组件已经存在，后续再补充具体实现
from .flux.evap import evap_1 # , evap_2
# from .flux.interception import interception_1
# from .flux.infiltration import infiltration_1, infiltration_2
from .flux.interflow import interflow_1
from .flux.recharge import recharge_1
from .flux.saturation import saturation_1
# from .flux.depression import depression_1
from .flux.exchange import exchange_3 # exchange_1, 
from .flux.baseflow import baseflow_1

# 参数取值范围字典 (匹配 MATLAB m_36_modhydrolog_15p_5s)
MODHYDROLOG_PARAMS_BOUNDS = {
    "insc": [0.1, 5.0],           # Maximum interception capacity [mm]
    "coeff": [0.0, 600.0],        # Maximum infiltration loss parameter [mm]
    "sq": [0.0, 15.0],            # Infiltration loss exponent [-]
    "smsc": [1.0, 2000.0],        # Maximum soil moisture capacity [mm]
    "sub": [0.0, 1.0],            # Proportionality constant [-]
    "crak": [0.0, 1.0],           # Proportionality constant [-]
    "em": [0.0, 20.0],            # Maximum plant-controlled evap rate [mm/d]
    "dsc": [1.0, 50.0],           # Maximum depression capacity [mm]
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
# ==============================================================================
# 1. 核心 Flux 函数 (Hard Logic / Numerical Safe Versions)
# ==============================================================================

def depression_1(p1, p2, S, Smax, incoming_flux, nearzero=1e-6):
    """[Hard Logic] Linear damping to avoid singularity."""
    capacity = F.relu(Smax - S)
    Smax_safe = torch.clamp(Smax, min=1.0)
    ratio_filled = torch.clamp(S / Smax_safe, max=1.0)
    damping = 1.0 - ratio_filled
    potential = p1 * incoming_flux * damping
    return torch.minimum(potential, capacity)

def exchange_1(p1, p2, p3, S, fmax, nearzero=1e-6):
    """[Safe] No sign(), safe exp."""
    s_abs = torch.abs(S)
    linear_part = p1 * S
    arg = torch.clamp(-p3 * s_abs, min=-30.0, max=0.0)
    exp_term = 1.0 - torch.exp(arg)
    nonlinear_part = p2 * exp_term * (S / (s_abs + nearzero))
    flow = linear_part + nonlinear_part
    return torch.maximum(flow, -torch.abs(fmax))

def infiltration_1(p1, p2, S, Smax, fin, nearzero=1e-6):
    """[Safe] Denominator lock."""
    Smax_safe = torch.clamp(Smax, min=1.0)
    arg = torch.clamp(-p2 * S / Smax_safe, min=-30.0, max=0.0)
    rate = p1 * torch.exp(arg)
    return torch.minimum(rate, fin)

def infiltration_2(p1, p2, S1, S1max, flux, S2, nearzero=1e-6):
    """[Safe] Denominator lock."""
    S1max_safe = torch.clamp(S1max, min=1.0)
    arg = torch.clamp(-p2 * S1 / S1max_safe, min=-30.0, max=0.0)
    rate = p1 * torch.exp(arg)
    net_inf = F.relu(rate - flux)
    return torch.minimum(net_inf, S2)

def evap_2(p1, S, Smax, Ep, nearzero=1e-6):
    Smax_safe = torch.clamp(Smax, min=1.0)
    ratio = torch.clamp(S / Smax_safe, max=1.0)
    potential = p1 * ratio
    return torch.minimum(torch.minimum(potential, Ep), S)

def interception_1(incoming_flux, S, Smax, nearzero=1e-6):
    "影响比较大"
    excess = F.relu(S - Smax)
    return torch.minimum(excess, S)

def create_initial_state(n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5

# ==============================================================================
# 2. 主 Step 函数 (包含入口清洗逻辑)
# ==============================================================================

def modhydrolog_step(
    P, T, PET, 
    insc, coeff, sq, smsc, sub, crak, em, dsc, ads, md, vcond, dlev, k1, k2, k3,
    S1, S2, S3, S4, S5, nearzero=1e-6,
):
    """
    MODHYDROLOG Step Function (Sanitized & Hard Logic).
    """
    # ----------------------------------------------------------------
    # B. 强制参数安全锁 (Parameter Safety Barrier)
    # ----------------------------------------------------------------
    # 你的报错显示 insc 进来就是 NaN。这里使用 nan_to_num 强制给它一个物理初值。
    # 这样即使网络炸了，物理过程也能跑下去，让优化器有机会自我修正。
    
    # 1. 填补 NaN (使用合理的中值或下限)
    insc  = torch.nan_to_num(insc, nan=1.0)
    coeff = torch.nan_to_num(coeff, nan=50.0)
    sq    = torch.nan_to_num(sq, nan=1.0)
    smsc  = torch.nan_to_num(smsc, nan=500.0)
    sub   = torch.nan_to_num(sub, nan=0.0)
    crak  = torch.nan_to_num(crak, nan=0.0)
    em    = torch.nan_to_num(em, nan=5.0)
    dsc   = torch.nan_to_num(dsc, nan=5.0)
    ads   = torch.nan_to_num(ads, nan=0.01)
    md    = torch.nan_to_num(md, nan=0.5)
    vcond = torch.nan_to_num(vcond, nan=0.01)
    dlev  = torch.nan_to_num(dlev, nan=0.0)
    k1    = torch.nan_to_num(k1, nan=0.01)
    k2    = torch.nan_to_num(k2, nan=0.01)
    k3    = torch.nan_to_num(k3, nan=1.0)

    # 2. 物理范围截断 (Safe Bounds)
    # 确保没有 0.0 (除 dlev 和部分比例参数外)
    # insc = torch.clamp(insc, min=0.1, max=5.0)
    # coeff = torch.clamp(coeff, min=0.01, max=600.0)
    # sq = torch.clamp(sq, min=0.1, max=15.0)
    # smsc = torch.clamp(smsc, min=1.0, max=2000.0)
    # sub = torch.clamp(sub, min=0.0, max=1.0)
    # crak = torch.clamp(crak, min=0.0, max=1.0)
    # em = torch.clamp(em, min=0.1, max=20.0)
    # dsc = torch.clamp(dsc, min=1.0, max=50.0)
    # ads = torch.clamp(ads, min=0.0, max=1.0)
    # md = torch.clamp(md, min=0.1, max=3.0) # 放宽范围
    # vcond = torch.clamp(vcond, min=0.001, max=0.5)
    # dlev = torch.clamp(dlev, min=-10.0, max=10.0)
    # k1 = torch.clamp(k1, min=0.01, max=1.0)
    # k2 = torch.clamp(k2, min=0.01, max=1.0)
    # k3 = torch.clamp(k3, min=0.01, max=100.0)

    # ----------------------------------------------------------------
    # C. 物理过程计算 (Physics Step)
    # ----------------------------------------------------------------

    # --- 1. 拦截层 (S1) ---
    S1 = S1 + P
    flux_EXC = interception_1(P, S1, insc)
    S1 = torch.clamp(S1 - flux_EXC, min=nearzero)
    flux_Ei = evap_1(S1, PET)
    S1 = torch.clamp(S1 - flux_Ei, min=nearzero)

    # --- 2. 产流与入渗分离 ---
    # flux_INF 计算: S2 是状态，smsc 是容量
    flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    
    flux_INT = interflow_1(sub, S2, smsc, flux_INF)
    flux_REC = recharge_1(crak, S2, smsc, flux_INF - flux_INT)
    flux_SMF = flux_INF - flux_INT - flux_REC

    # --- 3. 洼地储蓄 (S3) ---
    flux_TRAP = depression_1(ads, md, S3, dsc, flux_RUN)
    S3 = S3 + flux_TRAP
    
    # 洼地蒸发
    flux_Ed = evap_1(S3, ads * PET)
    S3 = torch.clamp(S3 - flux_Ed, min=nearzero)
    
    # 洼地再入渗 (S3 -> S2)
    # 参数含义: infiltration_2(coeff, sq, S_soil, S_soil_cap, flux_already_in, S_source_limit)
    flux_DINF = ads * infiltration_2(coeff, sq, S2, smsc, flux_SMF, S3)
    S3 = torch.clamp(S3 - flux_DINF, min=nearzero)
    
    flux_SRUN = flux_RUN - flux_TRAP

    # --- 4. 土壤水分层 (S2) ---
    S2 = S2 + flux_SMF + flux_DINF
    
    # Saturation excess (S2 满了溢出)
    flux_GWF = saturation_1(flux_SMF + flux_DINF, S2, smsc)
    S2 = torch.clamp(S2 - flux_GWF, min=nearzero)
    
    flux_Et = evap_2(em, S2, smsc, PET)
    S2 = torch.clamp(S2 - flux_Et, min=nearzero)

    # --- 5. 地下水层 (S4) ---
    S4 = S4 + flux_REC + flux_GWF
    
    # 与河道交换
    flux_FLOW = exchange_1(k1, k2, k3, S4, flux_SRUN)
    S4 = S4 - flux_FLOW
    
    # 地下水基流/渗漏
    flux_SEEP = exchange_3(vcond, S4, dlev)
    S4 = S4 - flux_SEEP

    # --- 6. 出流层 (S5) ---
    S5 = S5 + flux_SRUN + flux_INT + flux_FLOW
    
    # 线性水库出流 (k=1.0)
    flux_Q = baseflow_1(torch.ones_like(S5), S5)
    S5 = torch.clamp(S5 - flux_Q, min=nearzero)

    # --- 7. 输出聚合 ---
    Q = flux_Q
    Ea = flux_Ei + flux_Et + flux_Ed

    return Q, Ea, S1, S2, S3, S4, S5