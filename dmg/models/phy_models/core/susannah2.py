import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7
from ..flux.saturation import saturation_1
from ..flux.excess import excess_1
from ..flux.interflow import interflow_3

# Parameter range dictionary (based on MARRMoT m_10_susannah2_6p_2s)
SUSANNAH2_PARAMS_BOUNDS = {
    "sb": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "phi": [0.05, 0.95],  # Porosity [-]
    "fc": [0.05, 0.95],  # Field capacity as fraction of sb [-]
    "r": [0.0, 1.0],  # Fraction of recharge coefficient [-]
    "c": [0.0, 1.0],  # Subsurface flow constant [1/d]
    "d": [1.0, 5.0],  # Subsurface flow constant [-]
}

# Parameter description dictionary
SUSANNAH2_PARAMS_DESC = {
    "sb": "Maximum soil moisture storage [mm]",
    "phi": "Porosity [-]",
    "fc": "Field capacity as fraction of sb [-]",
    "r": "Fraction of recharge coefficient [-]",
    "c": "Subsurface flow constant [1/d]",
    "d": "Subsurface flow constant [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Susannah Brook v2 model.
    S1: Unsaturated storage
    S2: Saturated storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def susannah2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching SUSANNAH2_PARAMS_BOUNDS keys
    sb: torch.Tensor,
    phi: torch.Tensor,
    fc: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Susannah Brook model v2 single-step calculation.

    Model reference:
    Son, K., & Sivapalan, M. (2007). Improving model structure and reducing
    parameter uncertainty in conceptual water balance models through the use
    of auxiliary data. Water Resources Research, 43(1).
    """

    # --- 1. S1 Process (Unsaturated Store) ---

    # Inflow and Capacity to recharge S2
    # Capacity is (sb - S2) * fc / phi
    cap_s1_to_s2 = F.relu(sb - S2) * fc / (phi + nearzero)

    # recharge rg (saturation-based): fraction of P that infiltrates to S2
    flux_rg = saturation_1(P, S1, cap_s1_to_s2, nearzero=nearzero)
    zeros = torch.zeros_like(flux_rg)
    flux_rg = torch.clamp(flux_rg, min=zeros, max=P)

    # excess se (overflow when S1 already exceeds capacity)
    # Based on pre-inflow S1 to avoid double-counting with rg
    flux_se = excess_1(S1, cap_s1_to_s2, nearzero=nearzero)
    flux_se = F.relu(flux_se)

    # Total outflow from S1 to S2 capped at P + available S1 water
    # Prevents rg + se from exceeding what S1 can physically supply
    rg_se_max = P + F.relu(S1 - cap_s1_to_s2)
    total_rg_se = flux_rg + flux_se
    scale_rg_se = torch.where(
        total_rg_se > rg_se_max,
        rg_se_max / (total_rg_se + nearzero),
        torch.ones_like(total_rg_se),
    )
    flux_rg = flux_rg * scale_rg_se
    flux_se = flux_se * scale_rg_se

    # Temporary update for evaporation
    S1_tmp2 = torch.clamp(S1 + P - flux_rg - flux_se, min=nearzero)

    # Evaporation from unsaturated store
    flux_eus = evap_7(S1_tmp2, sb, PET, nearzero=nearzero)
    flux_eus = torch.minimum(flux_eus, S1_tmp2 - nearzero)
    flux_eus = torch.minimum(flux_eus, PET)
    flux_eus = F.relu(flux_eus)

    # Update S1 final
    S1_new = torch.clamp(S1_tmp2 - flux_eus, min=nearzero)

    # --- 2. S2 Process (Saturated Store) ---

    # Inflow to S2 is recharge and excess from S1
    S2_in = flux_rg + flux_se

    # Runoff from S2 (saturation excess)
    flux_qse = saturation_1(S2_in, S2, sb, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=S2_in)

    # Temporary update for evaporation
    S2_tmp = S2 + S2_in - flux_qse
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Evaporation from saturated store
    flux_esat = evap_7(S2_tmp, sb, PET, nearzero=nearzero)

    # Constraint to prevent negative storage
    flux_esat = torch.minimum(flux_esat, S2_tmp - nearzero)
    flux_esat = torch.minimum(flux_esat, PET)
    flux_esat = F.relu(flux_esat)

    # Update for slow processes
    S2_tmp2 = S2_tmp - flux_esat
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # Subsurface flow (qss) and Groundwater recharge (qr)
    # interflow_3(c_par, d_par, S)
    flux_qss = interflow_3((1.0 - r) * c, d, S2_tmp2, nearzero=nearzero)
    flux_qr = interflow_3(r * c, d, S2_tmp2, nearzero=nearzero)

    # Proportional scaling to prevent negative storage
    out_s2_total = flux_qss + flux_qr
    scaling_s2 = torch.where(
        out_s2_total > (S2_tmp2 - nearzero),
        (S2_tmp2 - nearzero) / (out_s2_total + nearzero),
        torch.ones_like(out_s2_total),
    )
    flux_qss = flux_qss * scaling_s2
    flux_qr = flux_qr * scaling_s2

    # Update S2 final
    S2_new = S2_tmp2 - flux_qss - flux_qr
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Output Aggregation ---
    # Qsim = qse (saturation excess) + qss (subsurface)
    # Ea = eus (unsaturated) + esat (saturated)
    # qr is a groundwater sink (GWsink), not part of Ea
    Qsim = flux_qse + flux_qss
    Ea = flux_eus + flux_esat

    return Qsim, Ea, S1_new, S2_new
