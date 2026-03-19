import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7
from ..flux.saturation import saturation_1

# 参数取值范围字典 (基于 MARRMoT m_01_collie1_1p_1s)
COLLIE1_PARAMS_BOUNDS = {
    "Smax": [1.0, 2000.0],
}

# 参数描述字典
COLLIE1_PARAMS_DESC = {
    "Smax": "Maximum soil moisture storage [mm]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    创建 Collie1 模型的初始状态.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def collie1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 COLLIE1_PARAMS_BOUNDS 的键顺序完全一致
    Smax: torch.Tensor,
    # 状态变量
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collie River 1 (传统水桶模型) 单步计算函数.
    
    模型引用:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development. Journal of Hydrology, 254,
    174-198. doi: 10.1016/S0022-1694(01)497-6.
    """

    # 1. 产流计算 (饱和产流)
    # flux_qse = P * (S1 / Smax)
    flux_qse = saturation_1(P, S1, Smax, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=0.0)

    # 2. 更新瞬时状态 (用于计算蒸发)
    # S1_tmp = S1 + P - flux_qse
    S1_tmp = S1 + P - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # 3. 蒸发计算
    # flux_ea = PET * (S1 / Smax)
    flux_ea = evap_7(S1_tmp, Smax, PET, nearzero=nearzero)
    
    # 限制蒸发量以确保质量守恒 (Rule 5.3)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = torch.minimum(flux_ea, PET)
    flux_ea = F.relu(flux_ea)

    # 4. 更新最终状态
    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 5. 变量聚合与返回
    Qsim = flux_qse
    Ea = flux_ea

    return Qsim, Ea, S1_new