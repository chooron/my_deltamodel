import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_6, evap_5
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_7
from ..flux.baseflow import baseflow_1, baseflow_2

# Parameter range dictionary (based on MARRMoT m_09_susannah1_6p_2s)
SUSANNAH1_PARAMS_BOUNDS = {
    "sb": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "sfc_frac": [0.05, 0.95],  # Wilting point as fraction of sb [-]
    "m": [0.05, 0.95],  # Fraction forest [-]
    "a": [1.0, 50.0],  # Runoff coefficient [d]
    "b": [0.2, 1.0],  # Runoff coefficient [-]
    "r": [0.0, 1.0],  # Runoff coefficient [d-1]
}

# Parameter description dictionary
SUSANNAH1_PARAMS_DESC = {
    "sb": "Maximum soil moisture storage [mm]",
    "sfc_frac": "Wilting point as fraction of sb [-]",
    "m": "Fraction forest [-]",
    "a": "Runoff coefficient [d]",
    "b": "Runoff coefficient [-]",
    "r": "Runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Susannah Brook v1 model.
    S1: Soil moisture storage
    S2: Groundwater storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def susannah1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching SUSANNAH1_PARAMS_BOUNDS keys
    sb: torch.Tensor,
    sfc_frac: torch.Tensor,
    m: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Susannah Brook v1 single-step calculation.

    Model reference:
    Son, K., & Sivapalan, M. (2007). Improving model structure and reducing
    parameter uncertainty in conceptual water balance models through the use
    of auxiliary data. Water Resources Research, 43(1).
    """

    # --- 1. S1 Process (Soil Moisture) ---

    # Step 1: Inflow + Fast Runoff
    # flux_qse: Saturation excess runoff
    flux_qse = saturation_1(P, S1, sb, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=P)

    # Update state for evaporation
    S1_tmp = S1 + P - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # Step 2: Evapotranspiration
    # flux_ebs: Bare soil evaporation
    # flux_eveg: Vegetated transpiration
    flux_ebs = evap_5(m, S1_tmp, sb, PET, nearzero=nearzero)
    flux_eveg = evap_6(m, sfc_frac, S1_tmp, sb, PET, nearzero=nearzero)

    # Limit total evaporation from S1
    flux_ea_s1 = flux_ebs + flux_eveg
    flux_ea_s1 = torch.minimum(flux_ea_s1, S1_tmp - nearzero)
    flux_ea_s1 = torch.minimum(flux_ea_s1, PET)
    flux_ea_s1 = F.relu(flux_ea_s1)

    # Update state for interflow
    S1_tmp2 = S1_tmp - flux_ea_s1
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # Step 3: Interflow and Split
    # flux_qss: Interflow from S1
    # interflow_7(S, Smax, p1, p2, p3)
    flux_qss = interflow_7(S1_tmp2, sb, sfc_frac, a, b, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S1_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # Split interflow into S2 (recharge) and streamflow
    # flux_qr = r * flux_qss (recharge to GW)
    flux_qr = baseflow_1(r, flux_qss, nearzero=nearzero)
    flux_qr = torch.minimum(flux_qr, flux_qss)

    flux_qss_direct = F.relu(flux_qss - flux_qr)

    # Update S1 final
    S1_new = S1_tmp2 - flux_qss
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. S2 Process (Groundwater) ---

    # Step 4: S2 Update and Baseflow
    # Inflow is flux_qr
    S2_tmp = S2 + flux_qr

    # flux_qb: Baseflow from saturated storage
    # baseflow_2(S, p1, p2)
    flux_qb = baseflow_2(S2_tmp, a, b, nearzero=nearzero)
    flux_qb = torch.minimum(flux_qb, S2_tmp - nearzero)
    flux_qb = F.relu(flux_qb)

    # Update S2 final
    S2_new = S2_tmp - flux_qb
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Output Aggregation ---
    # Qsim = qse (saturation) + qss_direct (interflow) + qb (baseflow)
    Qsim = flux_qse + flux_qss_direct + flux_qb
    Ea = flux_ea_s1

    return Qsim, Ea, S1_new, S2_new
