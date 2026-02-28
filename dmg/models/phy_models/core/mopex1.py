import torch
import torch.nn.functional as F
from typing import Tuple

# ================================================================
# 1. Parameter Configuration
# Adapted for large-sample hydrology (559 catchments)
# ================================================================

MOPEX1_PARAMS_BOUNDS = {
    # Surface bucket capacity [mm] (Ye et al. 2012: max ~1.0mm)
    # Expanded for large samples, but kept small to maintain saturation excess mechanism
    "s1max": [0.01, 50.0],

    # Infiltration time constant [days] (Ye et al. 2012: mean ~0.19 days)
    # Must be > 0.01 to avoid division instability
    "tw": [0.01, 5.0],

    # Subsurface flow recession constant [days] (Ye et al. 2012: max ~1300 days)
    "tu": [1.0, 2000.0],

    # Root zone storage capacity [mm] (Ye et al. 2012: max ~340mm)
    "se": [1.0, 1000.0],

    # Routing time constant [days]
    "tc": [0.1, 30.0],
}

MOPEX1_PARAMS_DESC = {
    "s1max": "Surface/Depression storage capacity (Threshold for Q1f) [mm]",
    "tw": "Infiltration time constant (Surface -> RootZone) [days]",
    "tu": "Subsurface flow recession constant [days]",
    "se": "Root zone storage capacity (Controls ET2) [mm]",
    "tc": "Streamflow routing time constant [days]",
}

def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (S1, S2, Sc1, Sc2)."""
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero
    )

# ================================================================
# 2. Flux Functions (Modular Physics Operators)
# ================================================================

def saturation_1(P: torch.Tensor, S: torch.Tensor, Smax: torch.Tensor) -> torch.Tensor:
    """Calculate saturation excess flow (Overflow)."""
    return F.relu((S + P) - Smax)

def evap_7(S: torch.Tensor, Smax: torch.Tensor, Ep: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Calculate potential evaporation based on relative storage."""
    ratio = S / (Smax + 1e-6)
    return Ep * ratio * dt

def recharge_3(k: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Calculate potential linear recharge/infiltration (k is time constant)."""
    return S / (k + 1e-6)

def baseflow_1(k: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Calculate potential linear baseflow/routing release."""
    return S / (k + 1e-6)

# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    Sb1: torch.Tensor, # s1max
    tw: torch.Tensor,  # tw
    tu: torch.Tensor,  # tu
    Se: torch.Tensor,  # se
    tc: torch.Tensor,  # tc
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # --- Safety Guards ---
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    
    # ==========================================
    # 1. Bucket 1 (Surface Soil)
    # 逻辑：参考 HBV，一步一更新
    # ==========================================
    
    # [Step 1] 加雨 & 算溢流 (Saturation Excess)
    # HBV逻辑: excess = SM - FC
    flux_q1f = saturation_1(P, S1, Sb1)
    
    # 立即更新 S1：先把雨加进去，把溢流减掉
    # 此时 S1 含有了 P 中没流走的部分
    S1 = S1 + P - flux_q1f
    
    # [Step 2] 算下渗 (Recharge)
    # 这里的 S1 已经是扣除溢流后的状态
    flux_qw_pot = recharge_3(tw, S1)
    # 强制约束：不能超过当前水量 (参考 HBV: min)
    flux_qw = torch.minimum(flux_qw_pot, S1)
    
    # 立即更新 S1
    S1 = S1 - flux_qw
    
    # [Step 3] 算蒸发 (Evaporation)
    # 这里的 S1 已经是扣除溢流和下渗后的状态 (吃剩下的)
    flux_et1_pot = evap_7(S1, Sb1, PET, delta_t)
    # 强制约束
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    
    # 立即更新 S1 (这是最终的 S1_new)
    S1_new = torch.clamp(S1 - flux_et1, min=0.0)

    # ==========================================
    # 2. Bucket 2 (Subsurface)
    # 逻辑：一步一更新
    # ==========================================
    
    # [Step 1] 接收水分
    S2 = S2 + flux_qw
    
    # [Step 2] 算基流 (Baseflow)
    flux_q2u_pot = baseflow_1(tu, S2)
    flux_q2u = torch.minimum(flux_q2u_pot, S2)
    
    # 立即更新 S2
    S2 = S2 - flux_q2u
    
    # [Step 3] 算蒸发 (Evaporation from S2)
    flux_et2_pot = evap_7(S2, Se, PET, delta_t)
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    
    # 立即更新 S2
    S2_new = torch.clamp(S2 - flux_et2, min=0.0)

    # ==========================================
    # 3. Routing (Fast & Slow)
    # 逻辑：一步一更新
    # ==========================================
    
    # --- Fast Flow Routing ---
    Sc1 = Sc1 + flux_q1f
    
    flux_qf_pot = baseflow_1(tc, Sc1)
    flux_qf = torch.minimum(flux_qf_pot, Sc1)
    
    Sc1_new = torch.clamp(Sc1 - flux_qf, min=0.0)
    
    # --- Slow Flow Routing ---
    Sc2 = Sc2 + flux_q2u
    
    flux_qs_pot = baseflow_1(tc, Sc2)
    flux_qs = torch.minimum(flux_qs_pot, Sc2)
    
    Sc2_new = torch.clamp(Sc2 - flux_qs, min=0.0)

    # ==========================================
    # 4. Returns
    # ==========================================
    
    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2
    
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new