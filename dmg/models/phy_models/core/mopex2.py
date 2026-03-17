from typing import Tuple
import torch
import torch.nn.functional as F

# 假设您已经在同级目录下的 mopex1.py 中使用了上一轮修复好的 evap_7 
from .mopex1 import (
    MOPEX1_PARAMS_BOUNDS,
    baseflow_1,    # 虽然导入了，但在下方已改为解析解计算以防梯度截断
    recharge_3,
    evap_7,
    saturation_1,
)

# MOPEX 2 Parameter Bounds
MOPEX2_PARAMS_BOUNDS: dict = MOPEX1_PARAMS_BOUNDS.copy()
MOPEX2_PARAMS_BOUNDS.update(
    {
        "ddf": [0.0, 20.0],  # mm/day/C (Expanded range for large samples)
        "tr": [-2.0, 3.0],   # Critical temperature [C]
    }
)

def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (S1, S2, Sc1, Sc2, Sn)."""
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero
    )

def melt_1(
    T: torch.Tensor, 
    Sn: torch.Tensor, 
    ddf: torch.Tensor, 
    T_crit: float = 0.0, 
    dt: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Degree-Day Snowmelt Module.
    Returns:
        is_rain: Fraction of liquid precipitation
        Qn: Snowmelt
    """
    # ✅ 修复：使用 Sigmoid 替代 (T > T_crit).float()
    # 提供平滑的雨雪混合过渡区，使得反向传播时能拥有连续的梯度
    is_rain = torch.sigmoid(T - T_crit)
    
    # Potential Melt (F.relu 天然可导)
    melt_pot = F.relu(T - T_crit) * ddf * dt
    
    # Actual Melt 
    Qn = torch.minimum(melt_pot, Sn)
    
    return is_rain, Qn


def mopex2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters (Inherits MOPEX1 + Snow)
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tr: torch.Tensor,
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,  # New State: Snowpack
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # --- 0. Guards ---
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn  = F.relu(Sn)

    # --- 1. Snow Module ---
    # ✅ 修复：将硬阈值 (T > tr).float() 替换为 Sigmoid。
    # 这一步对于神经网络学习 tr 参数至关重要，否则 tr 将无法从降水分配中获得梯度！
    is_rain = torch.sigmoid(T - tr)
    flux_qn = torch.minimum(is_rain* F.softplus(T - tr) * ddf * delta_t, Sn)

    Ps = P * (1.0 - is_rain)  # Snowfall
    Pr = P * is_rain          # Rainfall

    # Update Snowpack
    # (Sn - flux_qn >= 0 且 Ps >= 0，所以天然 >= 0，无需再用 clamp 截断梯度)
    Sn_new = Sn + Ps - flux_qn

    # Effective Precipitation entering Soil
    P_eff = Pr + flux_qn

    # --- 2. Surface Soil (S1): overflow -> ET -> infiltration ---
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    # evap_7 已在 mopex1 中修复了不超 PET 的问题
    flux_et1_pot = evap_7(S1, Sb1, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1 = S1 - flux_et1

    # ✅ 修复：将前向欧拉替换为指数解析解，保证时间参数 tw 全局可导
    flux_qw = S1 * (1.0 - torch.exp(-delta_t / (tw + nearzero)))
    S1_new  = S1 - flux_qw

    # --- 3. Subsurface (S2) ---
    S2 = S2 + flux_qw
    
    # ✅ 修复：将基流的前向欧拉替换为指数解析解
    flux_q2u = S2 * (1.0 - torch.exp(-delta_t / (tu + nearzero)))
    S2 = S2 - flux_q2u

    remaining_pet = F.relu(PET - flux_et1)
    flux_et2_pot = evap_7(S2, Se, remaining_pet, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = S2 - flux_et2

    # --- 4. Routing ---
    Sc1 = Sc1 + flux_q1f
    # ✅ 修复：将汇流的前向欧拉替换为指数解析解
    flux_qf = Sc1 * (1.0 - torch.exp(-delta_t / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    # ✅ 修复：将汇流的前向欧拉替换为指数解析解
    flux_qs = Sc2 * (1.0 - torch.exp(-delta_t / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new