import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_6, evap_5
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_9
from ..flux.baseflow import baseflow_1

# 参数取值范围字典 (基于 MARRMoT m_04_newzealand1_6p_1s)
NEWZEALAND1_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "sfc_frac": [
        0.05,
        0.95,
    ],  # Field capacity as fraction of maximum soil moisture [-]
    "m": [0.05, 0.95],  # Fraction forest [-]
    "a": [0.0, 1.0],  # Subsurface runoff coefficient [d-1]
    "b": [1.0, 5.0],  # Runoff non-linearity [-]
    "tcbf": [0.0, 1.0],  # Baseflow runoff coefficient [d-1]
}

# 参数描述字典
NEWZEALAND1_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "sfc_frac": "Field capacity as fraction of maximum soil moisture [-]",
    "m": "Fraction forest [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "b": "Runoff non-linearity [-]",
    "tcbf": "Baseflow runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    创建 New Zealand v1 模型的初始状态.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def newzealand1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 NEWZEALAND1_PARAMS_BOUNDS 的键顺序一致
    s1max: torch.Tensor,
    sfc_frac: torch.Tensor,
    m: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tcbf: torch.Tensor,
    # 状态变量
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    New Zealand model v1 单步计算函数.

    模型引用:
    Atkinson, S. E., Woods, R. A., & Sivapalan, M. (2002). Climate and
    landscape controls on water balance model complexity over changing
    timescales. Water Resources Research, 38(12), 17-50.
    """

    # 1. 产流计算 (饱和产流)
    # flux_qse = saturation_1(P, S1, s1max)
    flux_qse = saturation_1(P, S1, s1max, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=P)

    # 2. 状态预更新 (用于计算蒸发)
    S1_tmp = S1 + P - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # 3. 蒸发计算
    # flux_veg (植被蒸腾): evap_6(m, sfc, S1, s1max, PET)
    # flux_ebs (裸地蒸发): evap_5(m, S1, s1max, PET)
    flux_veg = evap_6(m, sfc_frac, S1_tmp, s1max, PET, nearzero=nearzero)
    flux_ebs = evap_5(m, S1_tmp, s1max, PET, nearzero=nearzero)

    # 限制总蒸发量以确保质量守恒
    flux_ea_total = flux_veg + flux_ebs
    flux_ea_total = torch.minimum(flux_ea_total, S1_tmp - nearzero)
    flux_ea_total = torch.minimum(flux_ea_total, PET)
    flux_ea_total = F.relu(flux_ea_total)

    # 4. 慢速产流计算 (地下径流与底流)
    S1_tmp2 = S1_tmp - flux_ea_total
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # flux_qss (壤中流): interflow_9(S1, a, sfc*s1max, b)
    sfc_threshold = sfc_frac * s1max
    flux_qss = interflow_9(S1_tmp2, a, sfc_threshold, b, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S1_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # 状态在计算 qbf 前再次修正
    S1_tmp3 = S1_tmp2 - flux_qss
    S1_tmp3 = torch.clamp(S1_tmp3, min=nearzero)

    # flux_qbf (底流): baseflow_1(tcbf, S1)
    flux_qbf = baseflow_1(tcbf, S1_tmp3, nearzero=nearzero)
    flux_qbf = torch.minimum(flux_qbf, S1_tmp3 - nearzero)
    flux_qbf = F.relu(flux_qbf)

    # 5. 更新最终状态
    S1_new = S1_tmp3 - flux_qbf
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 6. 变量聚合与返回
    # Qsim = qse (地表) + qss (壤中) + qbf (底流)
    Qsim = flux_qse + flux_qss + flux_qbf
    Ea = flux_ea_total

    return Qsim, Ea, S1_new
