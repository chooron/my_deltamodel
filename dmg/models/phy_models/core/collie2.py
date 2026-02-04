import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7, evap_3
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_8

# 参数取值范围字典 (基于 MARRMoT m_03_collie2_4p_1s)
COLLIE2_PARAMS_BOUNDS = {
    "Smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "Sfc_frac": [0.05, 0.95],  # Field capacity as fraction of Smax [-]
    "a": [0.0, 1.0],  # Subsurface runoff coefficient [d-1]
    "M": [0.05, 0.95],  # Fraction forest cover [-]
}

# 参数描述字典
COLLIE2_PARAMS_DESC = {
    "Smax": "Maximum soil moisture storage [mm]",
    "Sfc_frac": "Field capacity as fraction of Smax [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "M": "Fraction forest cover [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    创建 Collie2 模型的初始状态.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def collie2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 COLLIE2_PARAMS_BOUNDS 的键顺序一致
    Smax: torch.Tensor,
    Sfc_frac: torch.Tensor,
    a: torch.Tensor,
    M: torch.Tensor,
    # 状态变量
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collie River v2 单步计算函数.

    模型引用:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward
    approach to hydrological model development. Journal of Hydrology, 254,
    174 198. doi: 10.1016/S0022-1694(01)00496-6.
    """

    # 1. 产流计算 (饱和产流)
    # flux_qse = saturation_1(P, S1, Smax)
    flux_qse = saturation_1(P, S1, Smax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=P)

    # 2. 状态预更新 (用于计算蒸发)
    S1_tmp = S1 + P - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # 3. 蒸发计算
    # flux_eb (来自裸地的蒸发): evap_7(S1, Smax, (1-M)*PET)
    # flux_ev (来自植被的蒸腾): evap_3(Sfc_frac, S1, Smax, M*PET)
    pet_bare = (1.0 - M) * PET
    pet_veg = M * PET

    flux_eb = evap_7(S1_tmp, Smax, pet_bare, nearzero=nearzero)
    flux_ev = evap_3(Sfc_frac, S1_tmp, Smax, pet_veg, nearzero=nearzero)

    # 限制总蒸发量以确保质量守恒
    flux_ea_total = flux_eb + flux_ev
    flux_ea_total = torch.minimum(flux_ea_total, S1_tmp - nearzero)
    flux_ea_total = torch.minimum(flux_ea_total, PET)
    flux_ea_total = F.relu(flux_ea_total)

    # 4. 壤中流计算 (Slow Process)
    S1_tmp2 = S1_tmp - flux_ea_total
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # flux_qss = interflow_8(S1, a, Sfc_frac * Smax)
    Sfc = Sfc_frac * Smax
    flux_qss = interflow_8(S1_tmp2, a, Sfc, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S1_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # 5. 更新最终状态
    S1_new = S1_tmp2 - flux_qss
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 6. 变量聚合与返回
    Qsim = flux_qse + flux_qss
    Ea = flux_ea_total

    return Qsim, Ea, S1_new
