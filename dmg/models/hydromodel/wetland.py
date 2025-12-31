import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.interception import interception_2
from ..marrmot.evap import evap_1
from ..marrmot.saturation import saturation_2
from ..marrmot.baseflow import baseflow_1

# 参数取值范围字典 (基于 MARRMoT m_02_wetland_4p_1s)
WETLAND_PARAMS_BOUNDS = {
    "dw": [0.0, 5.0],  # Interception capacity [mm]
    "betaw": [0.0, 10.0],  # Soil moisture distribution parameter [-]
    "swmax": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "kw": [0.0, 1.0],  # Base flow time parameter [d-1]
}

# 参数描述字典
WETLAND_PARAMS_DESC = {
    "dw": "Interception capacity [mm]",
    "betaw": "Soil moisture distribution parameter [-]",
    "swmax": "Maximum soil moisture depth [mm]",
    "kw": "Base flow time parameter [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    创建 Wetland 模型的初始状态.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def wetland_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 WETLAND_PARAMS_BOUNDS 的键顺序一致
    dw: torch.Tensor,
    betaw: torch.Tensor,
    swmax: torch.Tensor,
    kw: torch.Tensor,
    # 状态变量
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wetland model (FLEX-Topo) 单步计算函数.

    模型引用:
    Savenije, H. H. G. (2010). "Topography driven conceptual modelling
    (FLEX-Topo)." Hydrology and Earth System Sciences, 14(12), 2681-2692.
    """

    # 1. 降雨拦截 (Interception)
    # flux_pe = interception_2(P, dw)
    # flux_ei = P - flux_pe
    flux_pe = interception_2(P, dw, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pe)

    # 2. 产流计算 (Saturation excess)
    # 基于当前状态计算饱和超渗产流
    # flux_qwsof = saturation_2(S1, swmax, betaw, flux_pe)
    flux_qwsof = saturation_2(S1, swmax, betaw, flux_pe, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qwsof)
    flux_qwsof = torch.clamp(flux_qwsof, min=zeros, max=flux_pe)

    # 3. 状态预更新 (用于蒸发和底流计算)
    S1_tmp = S1 + flux_pe - flux_qwsof
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # 4. 蒸发计算 (Soil evaporation)
    # flux_ew = evap_1(S1, PET)
    flux_ew = evap_1(S1_tmp, PET, nearzero=nearzero)

    # 限制蒸发量以确保质量守恒
    flux_ew = torch.minimum(flux_ew, S1_tmp - nearzero)
    flux_ew = torch.minimum(flux_ew, PET)
    flux_ew = F.relu(flux_ew)

    # 5. 底流计算 (Baseflow)
    # flux_qwgw = baseflow_1(kw, S1)
    S1_tmp2 = S1_tmp - flux_ew
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    flux_qwgw = baseflow_1(kw, S1_tmp2, nearzero=nearzero)
    flux_qwgw = torch.minimum(flux_qwgw, S1_tmp2 - nearzero)
    flux_qwgw = F.relu(flux_qwgw)

    # 6. 更新最终状态
    S1_new = S1_tmp2 - flux_qwgw
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 7. 变量聚合与返回
    # Ea = ei (拦截蒸发) + ew (土壤蒸发)
    # Qsim = qwsof (地表产流) + qwgw (地底产流/底流)
    Ea = flux_ei + flux_ew
    Qsim = flux_qwsof + flux_qwgw

    return Qsim, Ea, S1_new
