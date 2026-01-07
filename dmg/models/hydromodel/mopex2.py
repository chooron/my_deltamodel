from typing import Tuple
import torch
import torch.nn.functional as F
from .mopex1 import (
    MOPEX1_PARAMS_BOUNDS,
    baseflow_1,
    recharge_3,
    evap_7,
    saturation_1,
)

# MOPEX 2 Parameter Bounds
MOPEX2_PARAMS_BOUNDS: dict = MOPEX1_PARAMS_BOUNDS.copy()
MOPEX2_PARAMS_BOUNDS.update(
    {
        "ddf": [0.0, 20.0],  # mm/day/C (Expanded range for large samples)
        "tr": [-2.0, 3.0],  # Critical temperature [C]
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
        Ps: Snowfall (Solid P)
        Pr: Rainfall (Liquid P)
        Qn: Snowmelt
    """
    # 1. Split Precipitation (Masking for stability)
    # T <= T_crit: Snow; T > T_crit: Rain
    is_rain = (T > T_crit).float()
    
    # Note: P is not input here, splitting logic usually happens with P.
    # But to keep modular, we just return the split fraction or handle P outside.
    # Here we assume P is handled outside or passed in. 
    # Let's handle Melt logic solely on T and Sn here.
    
    # 2. Potential Melt
    # Melt = ddf * (T - T_crit) * dt
    melt_pot = F.relu(T - T_crit) * ddf * dt
    
    # 3. Actual Melt (limited by Snow Storage)
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
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    # --- 1. Snow Module ---
    # Determine rain/snow fraction
    is_rain, flux_qn = melt_1(
        T, Sn, ddf, T_crit=0.0, dt=delta_t
    )  # Assuming tr is used as offset or T_crit
    # Or using the parameter 'tr' directly:
    is_rain = (T > tr).float()
    flux_qn = torch.minimum(F.relu(T - tr) * ddf * delta_t, Sn)

    Ps = P * (1 - is_rain)  # Snowfall
    Pr = P * is_rain  # Rainfall

    # Update Snowpack
    Sn_new = torch.clamp(Sn + Ps - flux_qn, min=0.0)

    # Effective Precipitation entering Soil
    P_eff = Pr + flux_qn

    # --- 2. Surface Soil (S1) ---
    # Same as Mopex 1 but input is P_eff
    flux_q1f = saturation_1(P_eff, S1, Sb1)
    S1_avail = S1 + P_eff - flux_q1f

    flux_qw_pot = recharge_3(tw, S1_avail)
    flux_et1_pot = evap_7(S1_avail, Sb1, PET, delta_t)

    sum_flux_pot = flux_qw_pot + flux_et1_pot
    sum_flux_actual = torch.minimum(sum_flux_pot, S1_avail)
    alloc_ratio = sum_flux_actual / (sum_flux_pot + nearzero)

    flux_qw = flux_qw_pot * alloc_ratio
    flux_et1 = flux_et1_pot * alloc_ratio
    S1_new = torch.clamp(S1_avail - flux_qw - flux_et1, min=0.0)

    # --- 3. Subsurface (S2) ---
    S2_in = S2 + flux_qw
    flux_q2u_pot = baseflow_1(tu, S2_in)
    flux_et2_pot = evap_7(S2_in, Se, PET, delta_t)

    sum_flux_pot_2 = flux_q2u_pot + flux_et2_pot
    sum_flux_actual_2 = torch.minimum(sum_flux_pot_2, S2_in)
    alloc_ratio_2 = sum_flux_actual_2 / (sum_flux_pot_2 + nearzero)

    flux_q2u = flux_q2u_pot * alloc_ratio_2
    flux_et2 = flux_et2_pot * alloc_ratio_2
    S2_new = torch.clamp(S2_in - flux_q2u - flux_et2, min=0.0)

    # --- 4. Routing ---
    Sc1_in = Sc1 + flux_q1f
    flux_qf = torch.minimum(baseflow_1(tc, Sc1_in), Sc1_in)
    Sc1_new = torch.clamp(Sc1_in - flux_qf, min=0.0)

    Sc2_in = Sc2 + flux_q2u
    flux_qs = torch.minimum(baseflow_1(tc, Sc2_in), Sc2_in)
    Sc2_new = torch.clamp(Sc2_in - flux_qs, min=0.0)

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
