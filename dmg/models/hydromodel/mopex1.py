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
    # Parameters
    Sb1: torch.Tensor, # s1max
    tw: torch.Tensor,  # tw
    tu: torch.Tensor,  # tu
    Se: torch.Tensor,  # se
    tc: torch.Tensor,  # tc
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor, # Fast routing store
    Sc2: torch.Tensor, # Slow routing store
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-1 (Base Model) Single Step Calculation.
    Features numerical stability guards and mass balance allocation.
    """
    
    # --- Safety Guards ---
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    
    # ==========================================
    # 1. Bucket 1 (Surface Soil)
    # ==========================================
    
    # [Flux 1] Saturation Excess (Overflow)
    flux_q1f = saturation_1(P, S1, Sb1)
    
    # Available water after overflow
    S1_avail = S1 + P - flux_q1f
    
    # [Flux 2 & 3] Potential Recharge & Evaporation
    flux_qw_pot  = recharge_3(tw, S1_avail)
    flux_et1_pot = evap_7(S1_avail, Sb1, PET, delta_t)
    
    # [Allocation] Ensure outflows do not exceed S1_avail
    sum_flux_pot = flux_qw_pot + flux_et1_pot
    sum_flux_actual = torch.minimum(sum_flux_pot, S1_avail)
    
    alloc_ratio = sum_flux_actual / (sum_flux_pot + nearzero)
    
    flux_qw  = flux_qw_pot * alloc_ratio
    flux_et1 = flux_et1_pot * alloc_ratio
    
    S1_new = torch.clamp(S1_avail - flux_qw - flux_et1, min=0.0)

    # ==========================================
    # 2. Bucket 2 (Subsurface)
    # ==========================================
    
    S2_in = S2 + flux_qw
    
    # [Flux 4 & 5] Potential Baseflow & Evaporation
    flux_q2u_pot = baseflow_1(tu, S2_in)
    flux_et2_pot = evap_7(S2_in, Se, PET, delta_t)
    
    # [Allocation]
    sum_flux_pot_2 = flux_q2u_pot + flux_et2_pot
    sum_flux_actual_2 = torch.minimum(sum_flux_pot_2, S2_in)
    
    alloc_ratio_2 = sum_flux_actual_2 / (sum_flux_pot_2 + nearzero)
    
    flux_q2u = flux_q2u_pot * alloc_ratio_2
    flux_et2 = flux_et2_pot * alloc_ratio_2
    
    S2_new = torch.clamp(S2_in - flux_q2u - flux_et2, min=0.0)

    # ==========================================
    # 3. Routing (Fast & Slow)
    # ==========================================
    
    # --- Fast Flow Routing ---
    Sc1_in = Sc1 + flux_q1f
    flux_qf_pot = baseflow_1(tc, Sc1_in)
    flux_qf = torch.minimum(flux_qf_pot, Sc1_in)
    Sc1_new = torch.clamp(Sc1_in - flux_qf, min=0.0)
    
    # --- Slow Flow Routing ---
    Sc2_in = Sc2 + flux_q2u
    flux_qs_pot = baseflow_1(tc, Sc2_in)
    flux_qs = torch.minimum(flux_qs_pot, Sc2_in)
    Sc2_new = torch.clamp(Sc2_in - flux_qs, min=0.0)

    # ==========================================
    # 4. Returns
    # ==========================================
    
    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2
    
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new