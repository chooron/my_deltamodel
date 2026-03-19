import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# 假设核心组件已经存在，后续再补充具体实现
from ..flux.evap import evap_1  # , evap_2

# from ..flux.interception import interception_1
# from ..flux.infiltration import infiltration_1, infiltration_2
from ..flux.interflow import interflow_1
from ..flux.recharge import recharge_1
from ..flux.saturation import saturation_1

# from ..flux.depression import depression_1
from ..flux.exchange import exchange_3  # exchange_1,
from ..flux.baseflow import baseflow_1

# 参数取值范围字典 (匹配 MATLAB m_36_modhydrolog_15p_5s)
MODHYDROLOG_PARAMS_BOUNDS = {
    "insc": [0.1, 5.0],  # Maximum interception capacity [mm]
    "coeff": [0.0, 600.0],  # Maximum infiltration loss parameter [mm]
    "sq": [0.0, 15.0],  # Infiltration loss exponent [-]
    "smsc": [1.0, 2000.0],  # Maximum soil moisture capacity [mm]
    "sub": [0.0, 1.0],  # Proportionality constant [-]
    "crak": [0.0, 1.0],  # Proportionality constant [-]
    "em": [0.0, 20.0],  # Maximum plant-controlled evap rate [mm/d]
    "dsc": [1.0, 50.0],  # Maximum depression capacity [mm]
    "ads": [0.0, 1.0],  # Land fraction functioning as depression storage [-]
    "md": [0.99, 1.0],  # Depression storage parameter [-]
    "vcond": [0.0, 0.5],  # Leakage coefficient [mm/d]
    "dlev": [
        -10.0,
        10.0,
    ],  # Datum around which groundwater fluctuates relative to river bed [mm]
    "k1": [0.0, 1.0],  # Flow exchange parameter [d-1]
    "k2": [0.0, 1.0],  # Flow exchange parameter [d-1]
    "k3": [0.0, 100.0],  # Flow exchange parameter [d-1]
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


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


# ==============================================================================
# 2. 主 Step 函数 (包含入口清洗逻辑)
# ==============================================================================
def depression_1(ads, md, S, Smax, incoming_flux, nearzero=1e-6):
    """
    [Fixed] Uses MD parameter based on MODHYDROLOG Eq (Source 108).
    TRAP = (DSC - ADS * S_accumulated) * exp(-MD * DSC / RUN)
    """
    # 1. 计算剩余容量 (Capacity)
    # S 是当前洼地蓄水量，Smax 是 DSC (Depression Store Capacity)
    # 论文中 Source 108: (DSC - ADS * ARGD)
    # 这里简化模型假设 S 已经是归一化后的洼地蓄水
    capacity = F.relu(Smax - S)

    # 2. 构建指数衰减因子 (The filling efficiency)
    # 原文逻辑：RUN (incoming_flux) 越大，截留越容易充满
    # 我们使用稳定的软阈值来模拟 RUN 对截留率的影响

    valid_run = torch.clamp(incoming_flux, min=nearzero)
    Smax_safe = torch.clamp(Smax, min=1.0)

    # 构造指数项：exp(-MD * DSC / RUN)
    # 注意：如果 RUN 很大，exponent -> 0, exp -> 1.0 (全截留)
    # 如果 RUN 很小，exponent -> -inf, exp -> 0.0 (不截留)
    # 为了数值稳定性，限制 exponent 的下限
    exponent = -md * (Smax_safe / valid_run)
    exponent = torch.clamp(exponent, min=-20.0, max=0.0)  # 防止梯度爆炸

    efficiency_factor = torch.exp(exponent)

    # 3. 计算潜在截留量
    # ads 是洼地面积比例 (Fraction of area)
    potential_trap = ads * incoming_flux * efficiency_factor

    # 4. 物理约束：不能超过剩余容量，也不能超过流入量
    return torch.minimum(torch.minimum(potential_trap, capacity), incoming_flux)


# ==============================================================================
# 修复的 Step Function
# ==============================================================================

def modhydrolog_step(
    P, T, PET,
    insc, coeff, sq, smsc, sub, crak, em, dsc, ads, md, vcond, dlev, k1, k2, k3,
    S1, S2, S3, S4, S5,
    nearzero=1e-6,
):
    # 保存初始状态用于后续可能的 Check (可选)
    # S1_old, S2_old, ... = S1.clone(), S2.clone(), ...

    # ==========================================================================
    # 1. 拦截层 (S1) - Interception
    # ==========================================================================
    # 1.1 蒸发 (Evaporation)
    flux_Ei_pot = evap_1(S1, PET)
    flux_Ei = torch.minimum(flux_Ei_pot, S1)
    S1 = S1 - flux_Ei

    # 1.2 截留与溢出 (Throughfall)
    # 你的 interception_1 是基于 (S-Smax) 计算的，所以必须先把 P 加进去
    S1 = S1 + P 
    # 注意：这里的 P 是这一步新加的水，S1 暂时可能 > Smax
    flux_EXC = interception_1(P, S1, insc) 
    # 物理限制：溢出不能超过 S1 当前持有的水
    flux_EXC = torch.minimum(flux_EXC, S1)
    S1 = S1 - flux_EXC 
    
    # ==========================================================================
    # 2. 产流分离 (Flux Partitioning)
    # ==========================================================================
    # 2.1 总入渗 (Infiltration)
    flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC)
    flux_INF = torch.minimum(flux_INF, flux_EXC)
    
    # 剩余的变成地表径流
    flux_RUN = flux_EXC - flux_INF

    # 2.2 壤中流 (Interflow)
    flux_INT_pot = interflow_1(sub, S2, smsc, flux_INF)
    flux_INT = torch.minimum(flux_INT_pot, flux_INF)
    
    # 2.3 补给 (Recharge)
    remain_after_int = flux_INF - flux_INT
    flux_REC_pot = recharge_1(crak, S2, smsc, remain_after_int)
    flux_REC = torch.minimum(flux_REC_pot, remain_after_int)
    
    # 2.4 土壤补给 (Soil Moisture Feed)
    flux_SMF = remain_after_int - flux_REC
    
    # ==========================================================================
    # 3. 洼地层 (S3) - Depression Storage
    # ==========================================================================
    # 3.1 洼地截留
    flux_TRAP = depression_1(ads, md, S3, dsc, flux_RUN, nearzero)
    S3 = S3 + flux_TRAP
    
    # 计算最终地表径流 (Surface Runoff)
    flux_SRUN = flux_RUN - flux_TRAP

    # 3.2 洼地蒸发
    flux_Ed_pot = evap_1(S3, ads * PET)
    flux_Ed = torch.minimum(flux_Ed_pot, S3)
    S3 = S3 - flux_Ed 

    # 3.3 滞后入渗 (Delayed Infiltration S3 -> S2)
    flux_DINF_pot = infiltration_2(coeff, sq, S2, smsc, flux_SMF, S3)
    flux_DINF = flux_DINF_pot * ads 
    flux_DINF = torch.minimum(flux_DINF, S3) 
    S3 = S3 - flux_DINF
    
    S3 = torch.clamp(S3, min=nearzero)

    # ==========================================================================
    # 4. 土壤水分层 (S2)
    # ==========================================================================
    # 4.1 接收水分
    S2 = S2 + flux_SMF + flux_DINF
    
    # 4.2 蒸发
    flux_Et_pot = evap_2(em, S2, smsc, PET)
    flux_Et = torch.minimum(flux_Et_pot, S2)
    S2 = S2 - flux_Et
    
    # 4.3 饱和溢出 (GWF)
    # PyTorch 顺序逻辑：S2 此时已经包含了输入水，如果超过 smsc，多余的溢出
    # MATLAB 逻辑是 flux_GWF = saturation_1(flux_SMF, S2, smsc)，通常也是指 S2+SMF-Smax
    excess_s2 = torch.relu(S2 - smsc)
    flux_GWF = excess_s2 
    S2 = S2 - flux_GWF
    
    S2 = torch.clamp(S2, min=nearzero)

    # ==========================================================================
    # 5. 地下水与河道交互 (S4 <-> S5)
    # ==========================================================================
    # 5.1 S4 接收补给
    S4 = S4 + flux_REC + flux_GWF
    
    # 5.2 S5 接收径流 (此时 S5 是交互前的状态)
    S5 = S5 + flux_SRUN + flux_INT
    
    # 5.3 深层渗漏 (Seepage from S4) [关键修正点]
    flux_SEEP_pot = exchange_3(vcond, S4, dlev)
    flux_SEEP = torch.minimum(flux_SEEP_pot, S4)
    S4 = S4 - flux_SEEP # 这里扣除了水，必须在输出中体现
    
    # 5.4 交互流 (Flow between S4 and S5)
    pot_flow = exchange_1(k1, k2, k3, S4, flux_SRUN)
    
    flow_out = F.relu(pot_flow)      # S4 -> S5
    flow_in  = F.relu(-pot_flow)     # S5 -> S4
    
    real_flow_out = torch.minimum(flow_out, S4)
    real_flow_in = torch.minimum(flow_in, S5)
    
    flux_FLOW = real_flow_out - real_flow_in
    
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    
    S4 = torch.clamp(S4, min=nearzero)

    # ==========================================================================
    # 6. 河道汇流 (S5)
    # ==========================================================================
    flux_Q_pot = baseflow_1(torch.ones_like(S5), S5)
    flux_Q = torch.minimum(flux_Q_pot, S5)
    S5 = S5 - flux_Q
    
    S5 = torch.clamp(S5, min=nearzero)

    # ==========================================================================
    # 7. 输出聚合 [关键修正]
    # ==========================================================================
    Q = flux_Q
    Ea = flux_Ei + flux_Et + flux_Ed + flux_SEEP # 其他损失
    # 必须把 SEEP 返回，因为这是从系统边界流出的水
    # Loss = flux_SEEP 
    
    return Q, Ea, S1, S2, S3, S4, S5