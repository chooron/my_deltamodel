import torch
import torch.nn.functional as F
from typing import Tuple

# 假设同级目录下已有修复好的 mopex1
from .mopex1 import (
    baseflow_1,
    recharge_3,
    evap_7,
    saturation_1,
)

# MARRMoT-style parameter bounds (seasonal interception, no external LAI)
MOPEX4_PARAMS_BOUNDS = {
    "Sb1": [0.01, 50.0],
    "tw": [0.01, 5.0],
    "tu": [1.0, 2000.0],
    "Se": [1.0, 1000.0],
    "tc": [0.1, 30.0],
    "ddf": [0.0, 20.0],
    "tcrit": [-3.0, 3.0],
    "Sb2": [1.0, 1500.0],
    "alpha": [0.0, 1.0],
    "is_time": [0.0, 365.0],
}

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

def interception_seasonal(
    P: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Seasonal interception using a cosine-based seasonality factor that peaks at day `is_time`.
    Replaces external LAI forcing with a calibrated sinusoid (MARRMoT convention).
    """
    # ✅ 优化：使用 PyTorch 内置的 torch.pi 提升精度和规范性
    rad = 2.0 * torch.pi * (doy - is_time) / 365.0
    season_factor = 0.5 * (torch.cos(rad) + 1.0)

    flux_potential = alpha * P * season_factor
    flux_interception = torch.minimum(flux_potential, P)

    return flux_interception


def interception_1(
    P: torch.Tensor,
    alpha: torch.Tensor,
    LAI: torch.Tensor,
    LAI_max: float = 5.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Vegetation Interception.
    I = alpha * P * (LAI / LAI_max)
    """
    # Normalized LAI
    lai_ratio = torch.clamp(LAI / (LAI_max + nearzero), max=1.0)
    I_pot = alpha * P * lai_ratio
    I = torch.minimum(I_pot, P)
    return I


def mopex4_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # Parameters
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
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
    # --- Guards ---
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn  = F.relu(Sn)

    # --- 0. Seasonal Interception ---
    flux_i_pot = interception_seasonal(P, doy, alpha, is_time, nearzero)
    
    # ✅ 修复（致命物理漏洞）：截留的水最终要蒸发，蒸发量不能超过当天的总 PET。
    # 这一步确保在极端暴雨下，拦截的蒸发水不会引发能量不守恒。
    flux_i = torch.minimum(flux_i_pot, PET)
    P_through = P - flux_i
    
    # ✅ 修复：扣除被树冠截留消耗的 PET，剩下的能量再交给地表土壤蒸发
    pet_for_soil = F.relu(PET - flux_i)

    # --- 1. Snow Module (Uses P_through) ---
    # ✅ 修复：将硬阈值替换为 Sigmoid，打通 tcrit 参数的梯度回传
    is_rain = torch.sigmoid(T - tcrit)
    flux_qn = torch.minimum(is_rain * F.softplus(T - tcrit) * ddf * delta_t, Sn)
    Ps = P_through * (1.0 - is_rain)
    Pr = P_through * is_rain
    
    # 取消冗余 clamp
    Sn_new = Sn + Ps - flux_qn
    P_eff = Pr + flux_qn

    # --- 2. Soil: overflow -> ET -> infiltration ---
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    # ✅ 修复：传入扣除了冠层截留后的剩余 PET (pet_for_soil)
    flux_et1_pot = evap_7(S1, Sb1, pet_for_soil, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1 = S1 - flux_et1

    # ✅ 修复：指数衰减解析解替代欧拉近似
    flux_qw = S1 * (1.0 - torch.exp(-delta_t / tw))
    S1_new  = S1 - flux_qw

    # --- 3. Subsurface ---
    S2 = S2 + flux_qw
    
    # ✅ 修复：去掉占用显存的 torch.zeros_like 算子，简化为纯净的 ReLU 操作
    flux_q2f = F.relu(S2 - Sb2)
    S2 = S2 - flux_q2f

    # ✅ 修复：指数衰减解析解替代欧拉近似
    flux_q2u = S2 * (1.0 - torch.exp(-delta_t / tu))
    S2 = S2 - flux_q2u

    # 计算 S2 能用的最终 PET
    remaining_pet = F.relu(pet_for_soil - flux_et1)
    flux_et2_pot = evap_7(S2, Se, remaining_pet, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = S2 - flux_et2

    # --- 4. Routing (Same as Mopex 3) ---
    Sc1 = Sc1 + flux_q1f + flux_q2f
    # ✅ 修复：汇流使用指数解析解
    flux_qf = Sc1 * (1.0 - torch.exp(-delta_t / tc))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    # ✅ 修复：汇流使用指数解析解
    flux_qs = Sc2 * (1.0 - torch.exp(-delta_t / tc))
    Sc2_new = Sc2 - flux_qs

    # Total ET includes Interception
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total  = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new