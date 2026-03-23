import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# 注意：这里假设你已经创建了对应的核心组件
from ..flux.saturation import saturation_14, saturation_2
from ..flux.interflow import interflow_5
from ..flux.baseflow import baseflow_1
from ..flux.split import split_1, split_2
from ..flux.evap import evap_21

# 参数取值范围字典 (基于 flux m_28_xinanjiang_12p_4s)
XINANJIANG_PARAMS_BOUNDS = {
    "aim": [0.0, 1.0],
    "par_a": [-0.49, 0.49],
    "par_b": [0.0, 10.0],
    "stot": [1.0, 2000.0],
    "fwm": [0.01, 0.99],
    "flm": [0.01, 0.99],
    "par_c": [0.01, 0.99],
    "ex": [0.0, 10.0],
    "ki": [0.0, 1.0],
    "kg": [0.0, 1.0],
    "ci": [0.0, 1.0],
    "cg": [0.0, 1.0],
}

# 参数描述字典
XINANJIANG_PARAMS_DESC = {
    "aim": "Fraction impervious area [-]",
    "par_a": "Tension water distribution inflection parameter [-]",
    "par_b": "Tension water distribution shape parameter [-]",
    "stot": "Total soil moisture storage (W+S) [mm]",
    "fwm": "Fraction of Stot that is Wmax [-]",
    "flm": "Fraction of wmax that is LM [-]",
    "par_c": "Fraction of LM for second evaporation change [-]",
    "ex": "Free water distribution shape parameter [-]",
    "ki": "Free water interflow parameter [d-1]",
    "kg": "Free water groundwater parameter [d-1]",
    "ci": "Interflow time coefficient [d-1]",
    "cg": "Baseflow time coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Xinanjiang model.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def xinanjiang_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数
    aim: torch.Tensor,
    par_a: torch.Tensor,
    par_b: torch.Tensor,
    stot: torch.Tensor,
    fwm: torch.Tensor,
    flm: torch.Tensor,
    par_c: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    # 状态
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    新安江模型单步计算 (顺序更新版 - Sequential Update)
    [Mass Conservative] By strictly limiting fluxes to residual storage at each step.
    """
    # 辅助参数
    wmax = fwm * stot
    smax = (1.0 - fwm) * stot
    lm = flm * wmax

    # --------------------------------------------------------------------------
    # 1. 降水分离
    # --------------------------------------------------------------------------
    flux_rb = split_1(aim, P, nearzero)
    flux_pi = split_2(aim, P, nearzero) # P - rb

    # --------------------------------------------------------------------------
    # 2. 张性水 (S1) - 产流 R
    # --------------------------------------------------------------------------
    # 产流计算
    flux_r = saturation_14(par_a, par_b, S1, wmax, flux_pi, nearzero)
    flux_r = torch.minimum(flux_r, flux_pi) # 物理限制

    # 蒸发计算 (基于当前 S1)
    # 计算扣除产流后的净输入
    s1_net_in = flux_pi - flux_r
    
    # S1 暂时拥有的总水量 (用于计算蒸发)
    s1_avail = S1 + s1_net_in
    s1_avail = F.relu(s1_avail) # 防负数
    
    flux_e_pot = evap_21(lm, par_c, S1, PET, nearzero)
    flux_e = torch.minimum(flux_e_pot, s1_avail) # 限制蒸发

    # 更新 S1
    S1_new = s1_avail - flux_e
    S1_new = torch.clamp(S1_new, min=0.0)

    # --------------------------------------------------------------------------
    # 3. 自由水 (S2) - 顺序显式步进
    # --------------------------------------------------------------------------
    # flux_rs: 饱和超渗，状态用原始 S2（flux_r 是本步输入，尚未加入）
    flux_rs = saturation_2(S2, smax, ex, flux_r, nearzero)
    S2_curr = F.relu(S2 + flux_r - flux_rs)

    # flux_ri: 壤中流，基于更新后的 S2_curr
    flux_ri = saturation_2(S2_curr, smax, ex, S2_curr * ki, nearzero)
    flux_ri = torch.minimum(flux_ri, S2_curr)
    S2_curr = S2_curr - flux_ri

    # flux_rg: 地下径流，基于再次更新后的 S2_curr
    flux_rg = saturation_2(S2_curr, smax, ex, S2_curr * kg, nearzero)
    flux_rg = torch.minimum(flux_rg, S2_curr)

    S2_new = torch.clamp(S2_curr - flux_rg, min=0.0)

    # --------------------------------------------------------------------------
    # 4. 汇流 (S3, S4)
    # --------------------------------------------------------------------------
    # S3: 壤中流汇流
    S3_tmp = S3 + flux_ri
    flux_qi = interflow_5(ci, S3_tmp, nearzero)
    flux_qi = torch.minimum(flux_qi, S3_tmp) # 防抽干
    S3_new = S3_tmp - flux_qi
    S3_new = torch.clamp(S3_new, min=0.0)

    # S4: 地下水汇流
    S4_tmp = S4 + flux_rg
    flux_qg = baseflow_1(cg, S4_tmp, nearzero)
    flux_qg = torch.minimum(flux_qg, S4_tmp) # 防抽干
    S4_new = S4_tmp - flux_qg
    S4_new = torch.clamp(S4_new, min=0.0)

    # --------------------------------------------------------------------------
    # 5. 输出
    # --------------------------------------------------------------------------
    Qsim = flux_rb + flux_rs + flux_qi + flux_qg
    Ea = flux_e

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new