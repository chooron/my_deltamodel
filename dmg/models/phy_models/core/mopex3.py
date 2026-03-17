import torch
import torch.nn.functional as F
from typing import Tuple

# 假设同级目录下已有修复好的 mopex1 和 mopex2
from .mopex1 import (
    baseflow_1,
    recharge_3,
    evap_7,
    saturation_1,
)
from .mopex2 import MOPEX2_PARAMS_BOUNDS

# MOPEX 3 Parameter Bounds
MOPEX3_PARAMS_BOUNDS = MOPEX2_PARAMS_BOUNDS.copy()
MOPEX3_PARAMS_BOUNDS.update(
    {
        "sb2": [1.0, 2000.0],  # Subsurface overflow threshold [mm] [cite: 1]
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

def mopex3_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tr: torch.Tensor,
    Sb2: torch.Tensor,  # New Parameter
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
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

    # --- 1. Snow Module (Inherited from MOPEX 2) ---
    # ✅ 修复：将硬阈值 (T > tr).float() 替换为 Sigmoid，打通温度参数的梯度传递路径
    is_rain = torch.sigmoid(T - tr)
    flux_qn = torch.minimum(is_rain * F.softplus(T - tr) * ddf * delta_t, Sn)
    Ps = P * (1.0 - is_rain)
    Pr = P * is_rain
    
    # ✅ 修复：去掉非必要的 clamp，因为 Ps>=0 且 flux_qn<=Sn，Sn 天然非负
    Sn_new = Sn + Ps - flux_qn
    P_eff  = Pr + flux_qn

    # --- 2. Surface Soil (S1): overflow -> ET -> infiltration ---
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    flux_et1_pot = evap_7(S1, Sb1, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1 = S1 - flux_et1

    # ✅ 修复：替换为指数解析解，保证下渗的平滑梯度
    flux_qw = S1 * (1.0 - torch.exp(-delta_t / (tw + nearzero)))
    S1_new  = S1 - flux_qw

    # --- 3. Subsurface (S2) with Overflow (Q2f) ---
    S2 = S2 + flux_qw
    
    # ✅ 优化：MOPEX 3 新增机制。原本需要 torch.zeros_like(S2) 来匹配参数，
    # 但本质就是溢流 F.relu(S2 - Sb2)，这样写能避免重复分配全零内存，计算更高效。
    flux_q2f = F.relu(S2 - Sb2)
    S2 = S2 - flux_q2f

    # ✅ 修复：替换为指数解析解计算基流
    flux_q2u = S2 * (1.0 - torch.exp(-delta_t / (tu + nearzero)))
    S2 = S2 - flux_q2u

    remaining_pet = F.relu(PET - flux_et1)
    flux_et2_pot = evap_7(S2, Se, remaining_pet, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = S2 - flux_et2

    # --- 4. Routing ---
    # [cite_start]Fast Routing (Sc1) receives Q1f AND Q2f [cite: 1]
    Sc1 = Sc1 + flux_q1f + flux_q2f
    
    # ✅ 修复：地表/地下溢流的汇流替换为解析解
    flux_qf = Sc1 * (1.0 - torch.exp(-delta_t / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    
    # ✅ 修复：慢速基流的汇流替换为解析解
    flux_qs = Sc2 * (1.0 - torch.exp(-delta_t / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new