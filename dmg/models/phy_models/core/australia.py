import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7
from ..flux.saturation import saturation_1
from ..flux.excess import excess_1
from ..flux.interflow import interflow_3
from ..flux.recharge import recharge_3

# Parameter range dictionary (based on MARRMoT m_19_australia_8p_3s)
AUSTRALIA_PARAMS_BOUNDS = {
    "sb": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "phi": [0.05, 0.95],  # Porosity [-]
    "fc_frac": [0.01, 1.00],  # Wilting point as fraction of sb [-]
    "alpha_ss": [0.0, 1.0],  # Subsurface flow constant [1/d]
    "beta_ss": [1.0, 5.0],  # Subsurface non-linearity constant [-]
    "k_deep": [0.0, 1.0],  # Groundwater recharge constant [d-1]
    "alpha_bf": [0.0, 1.0],  # Groundwater flow constant [d-1]
    "beta_bf": [1.0, 5.0],  # Groundwater non-linearity constant [-]
}

# Parameter description dictionary
AUSTRALIA_PARAMS_DESC = {
    "sb": "Maximum soil moisture storage [mm]",
    "phi": "Porosity [-]",
    "fc_frac": "Wilting point as fraction of sb [-]",
    "alpha_ss": "Subsurface flow constant [1/d]",
    "beta_ss": "Subsurface non-linearity constant [-]",
    "k_deep": "Groundwater recharge constant [d-1]",
    "alpha_bf": "Groundwater flow constant [d-1]",
    "beta_bf": "Groundwater non-linearity constant [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Australia model.
    S1: Unsaturated storage
    S2: Saturated storage
    S3: Groundwater storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def australia_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching AUSTRALIA_PARAMS_BOUNDS keys
    sb: torch.Tensor,
    phi: torch.Tensor,
    fc_frac: torch.Tensor,
    alpha_ss: torch.Tensor,
    beta_ss: torch.Tensor,
    k_deep: torch.Tensor,
    alpha_bf: torch.Tensor,
    beta_bf: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Australia model single-step calculation.

    Model reference:
    Farmer, D., Sivapalan, M., & Jothityangkoon, C. (2003). Climate, soil,
    and vegetation controls upon the variability of water balance in
    temperate and semiarid landscapes: Downward approach to water balance
    analysis. Water Resources Research, 39(2).
    """

    # --- 1. S1 Process (Unsaturated Store) ---

    # Capacity available in S2 to receive recharge from S1
    # cap_s1_to_s2 = (sb - S2) * fc / phi
    cap_s1_to_s2 = F.relu(sb - S2) * fc_frac / (phi + nearzero)

    # flux_rg: Recharge flux from S1 to S2 (Saturation-based)
    flux_rg = saturation_1(P, S1, cap_s1_to_s2, nearzero=nearzero)
    zeros = torch.zeros_like(flux_rg)
    flux_rg = torch.clamp(flux_rg, min=zeros, max=P)

    # flux_se: Excess recharge (Overflow-based)
    S1_tmp_in = S1 + P - flux_rg
    flux_se = excess_1(S1_tmp_in, cap_s1_to_s2, nearzero=nearzero)
    flux_se = F.relu(flux_se)

    # Temporary update for evaporation
    S1_tmp_evap = S1_tmp_in - flux_se
    S1_tmp_evap = torch.clamp(S1_tmp_evap, min=nearzero)

    # flux_eus: Evaporation from unsaturated store
    flux_eus = evap_7(S1_tmp_evap, sb, PET, nearzero=nearzero)
    flux_eus = torch.minimum(flux_eus, S1_tmp_evap - nearzero)
    flux_eus = torch.minimum(flux_eus, PET)
    flux_eus = F.relu(flux_eus)

    # Update S1 final
    S1_new = S1_tmp_evap - flux_eus
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. S2 Process (Saturated Store) ---

    # Inflow to S2 is recharge and excess from S1
    S2_in = flux_rg + flux_se

    # flux_qse: Saturation excess runoff from S2
    flux_qse = saturation_1(S2_in, S2, sb, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=S2_in)

    # Update S2 for evaporation
    S2_tmp_evap = S2 + S2_in - flux_qse
    S2_tmp_evap = torch.clamp(S2_tmp_evap, min=nearzero)

    # flux_esat: Evaporation from saturated store
    flux_esat = evap_7(S2_tmp_evap, sb, PET, nearzero=nearzero)
    flux_esat = torch.minimum(flux_esat, S2_tmp_evap - nearzero)
    flux_esat = torch.minimum(flux_esat, PET)
    flux_esat = F.relu(flux_esat)

    # Update S2 for slow releases
    S2_tmp_release = S2_tmp_evap - flux_esat
    S2_tmp_release = torch.clamp(S2_tmp_release, min=nearzero)

    # Subsurface flow (qss) and Deep groundwater recharge (qr)
    # qss: interflow_3(alpha, beta, S)
    flux_qss = interflow_3(alpha_ss, beta_ss, S2_tmp_release, nearzero=nearzero)
    # qr: recharge_3(k_deep, S)
    flux_qr = recharge_3(k_deep, S2_tmp_release, nearzero=nearzero)

    # Constraint for combined slow outflows from S2
    total_out_s2 = flux_qss + flux_qr
    scale_s2 = torch.where(
        total_out_s2 > (S2_tmp_release - nearzero),
        (S2_tmp_release - nearzero) / (total_out_s2 + nearzero),
        torch.ones_like(total_out_s2),
    )
    flux_qss = flux_qss * scale_s2
    flux_qr = flux_qr * scale_s2

    # Update S2 final
    S2_new = S2_tmp_release - flux_qss - flux_qr
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. S3 Process (Groundwater Store) ---

    # Inflow to S3 is qr
    S3_tmp_in = S3 + flux_qr
    S3_tmp_in = torch.clamp(S3_tmp_in, min=nearzero)

    # flux_qbf: Baseflow from groundwater
    # qbf: interflow_3(alpha_bf, beta_bf, S3)
    flux_qbf = interflow_3(alpha_bf, beta_bf, S3_tmp_in, nearzero=nearzero)
    flux_qbf = torch.minimum(flux_qbf, S3_tmp_in - nearzero)
    flux_qbf = F.relu(flux_qbf)

    # Update S3 final
    S3_new = S3_tmp_in - flux_qbf
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = qse (saturation) + qss (subsurface) + qbf (baseflow)
    # Ea = eus (unsaturated ET) + esat (saturated ET)
    Qsim = flux_qse + flux_qss + flux_qbf
    Ea = flux_eus + flux_esat

    return Qsim, Ea, S1_new, S2_new, S3_new
