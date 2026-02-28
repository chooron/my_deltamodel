import torch
import torch.nn.functional as F
from typing import Tuple
from .mopex1 import (
    baseflow_1,
    recharge_3,
    evap_7,
    saturation_1,
)
from .mopex4 import interception_seasonal

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

def gsi_1(T: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
    """
    Growing Season Index (phenology) as a calibrated linear ramp between tmin and tmax.
    """

    t_range = torch.clamp(tmax - tmin, min=0.1)
    gsi = (T - tmin) / t_range
    return torch.clamp(gsi, 0.0, 1.0)


# MARRMoT-style bounds for MOPEX 5 (12 parameters)
MOPEX5_PARAMS_BOUNDS = {
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
    "tmin": [-10.0, 5.0],
    "tmax": [5.0, 30.0],
}


def mopex5_step(
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
    tmin: torch.Tensor,
    tmax: torch.Tensor,
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
    # ... Guards ...
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    Sb1 = torch.clamp(Sb1, min=nearzero)
    tw = torch.clamp(tw, min=nearzero)
    tu = torch.clamp(tu, min=nearzero)
    Se = torch.clamp(Se, min=nearzero)
    tc = torch.clamp(tc, min=nearzero)
    ddf = torch.clamp(ddf, min=nearzero)
    Sb2 = torch.clamp(Sb2, min=nearzero)
    alpha = torch.clamp(alpha, min=nearzero)

    # --- 0. Phenology (GSI) Module ---
    flux_gsi = gsi_1(T, tmin, tmax)
    PET_effective = PET * flux_gsi

    # --- 1. Interception (Seasonal, no LAI forcing) ---
    flux_i = interception_seasonal(P, doy, alpha, is_time, nearzero)
    P_through = P - flux_i

    # --- 2. Snow Module (Same as Mopex 4) ---
    is_rain = (T > tcrit).float()
    flux_qn = torch.minimum(F.relu(T - tcrit) * ddf * delta_t, Sn)
    Ps = P_through * (1 - is_rain)
    Pr = P_through * is_rain
    Sn_new = torch.clamp(Sn + Ps - flux_qn, min=0.0)
    P_eff = Pr + flux_qn

    # --- 3. Soil (Uses PET_effective) ---
    S1 = S1 + P_eff
    flux_q1f = saturation_1(torch.zeros_like(S1), S1, Sb1)
    S1 = S1 - flux_q1f

    flux_qw_pot = recharge_3(tw, S1)
    flux_qw = torch.minimum(flux_qw_pot, S1)
    S1 = S1 - flux_qw

    flux_et1_pot = evap_7(S1, Sb1, PET_effective, delta_t)
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1_new = torch.clamp(S1 - flux_et1, min=0.0)

    # --- 4. Subsurface (Uses PET_effective) ---
    S2 = S2 + flux_qw
    flux_q2f = saturation_1(torch.zeros_like(S2), S2, Sb2)
    S2 = S2 - flux_q2f

    flux_q2u_pot = baseflow_1(tu, S2)
    flux_q2u = torch.minimum(flux_q2u_pot, S2)
    S2 = S2 - flux_q2u

    flux_et2_pot = evap_7(S2, Se, PET_effective, delta_t)
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = torch.clamp(S2 - flux_et2, min=0.0)

    # --- 5. Routing ---
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = torch.minimum(baseflow_1(tc, Sc1), Sc1)
    Sc1_new = torch.clamp(Sc1 - flux_qf, min=0.0)

    Sc2 = Sc2 + flux_q2u
    flux_qs = torch.minimum(baseflow_1(tc, Sc2), Sc2)
    Sc2_new = torch.clamp(Sc2 - flux_qs, min=0.0)

    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
