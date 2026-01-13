import torch
import torch.nn.functional as F
from typing import Tuple
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
    """Initialize state variables (S1, S2, Sc1, Sc2)."""
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
    # ... (Guards same as Mopex 2, add Sb2 clamp) ...
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)
    # --- 1. Snow Module (Same as Mopex 2) ---
    is_rain = (T > tr).float()
    flux_qn = torch.minimum(F.relu(T - tr) * ddf * delta_t, Sn)
    Ps = P * (1 - is_rain)
    Pr = P * is_rain
    Sn_new = torch.clamp(Sn + Ps - flux_qn, min=0.0)
    P_eff = Pr + flux_qn

    # --- 2. Surface Soil (S1) ---
    S1 = S1 + P_eff
    flux_q1f = saturation_1(torch.zeros_like(S1), S1, Sb1)
    S1 = S1 - flux_q1f

    flux_qw_pot = recharge_3(tw, S1)
    flux_qw = torch.minimum(flux_qw_pot, S1)
    S1 = S1 - flux_qw

    flux_et1_pot = evap_7(S1, Sb1, PET, delta_t)
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1_new = torch.clamp(S1 - flux_et1, min=0.0)

    # --- 3. Subsurface (S2) with Overflow (Q2f) ---
    S2 = S2 + flux_qw
    flux_q2f = saturation_1(torch.zeros_like(S2), S2, Sb2)
    S2 = S2 - flux_q2f

    flux_q2u_pot = baseflow_1(tu, S2)
    flux_q2u = torch.minimum(flux_q2u_pot, S2)
    S2 = S2 - flux_q2u

    flux_et2_pot = evap_7(S2, Se, PET, delta_t)
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = torch.clamp(S2 - flux_et2, min=0.0)

    # --- 4. Routing ---
    # [cite_start]Fast Routing (Sc1) receives Q1f AND Q2f [cite: 1]
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = torch.minimum(baseflow_1(tc, Sc1), Sc1)
    Sc1_new = torch.clamp(Sc1 - flux_qf, min=0.0)

    Sc2 = Sc2 + flux_q2u
    flux_qs = torch.minimum(baseflow_1(tc, Sc2), Sc2)
    Sc2_new = torch.clamp(Sc2 - flux_qs, min=0.0)

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
