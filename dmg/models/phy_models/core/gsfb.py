import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_20
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_11
from ..flux.baseflow import baseflow_1, baseflow_9
from ..flux.recharge import recharge_5


# Parameter range dictionary (based on MARRMoT m_20_gsfb_8p_3s)
GSFB_PARAMS_BOUNDS = {
    "c": [0.0, 1.0],  # Recharge time coefficient [d-1]
    "ndc": [0.05, 0.95],  # Threshold fraction of Smax [-]
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "emax": [0.0, 20.0],  # Maximum evaporation flux [mm/d]
    "frate": [0.0, 200.0],  # Maximum infiltration rate [mm/d]
    "b": [0.0, 1.0],  # Fraction of subsurface flow that is baseflow [-]
    "dpf": [0.0, 1.0],  # Baseflow time coefficient [d-1]
    "sdrmax": [1.0, 300.0],  # Threshold before baseflow can occur [mm]
}

# Parameter description dictionary
GSFB_PARAMS_DESC = {
    "c": "Recharge time coefficient [d-1]",
    "ndc": "Threshold fraction of Smax [-]",
    "smax": "Maximum soil moisture storage [mm]",
    "emax": "Maximum evaporation flux [mm/d]",
    "frate": "Maximum infiltration rate [mm/d]",
    "b": "Fraction of subsurface flow that is baseflow [-]",
    "dpf": "Baseflow time coefficient [d-1]",
    "sdrmax": "Threshold before baseflow can occur [mm]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for GSFB model.
    S1: Soil moisture store
    S2: Intermediate store
    S3: Saturated zone store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def gsfb_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching GSFB_PARAMS_BOUNDS keys
    c: torch.Tensor,
    ndc: torch.Tensor,
    smax: torch.Tensor,
    emax: torch.Tensor,
    frate: torch.Tensor,
    b: torch.Tensor,
    dpf: torch.Tensor,
    sdrmax: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    GSFB model single-step calculation.

    Model reference:
    Nathan, R. J., & McMahon, T. A. (1990). SFB model part l . Validation of
    fixed model parameters. Civil Eng. Trans., 157-161.
    """

    # --- 1. Saturated zone to Soil moisture Recharge (S3 -> S1) ---
    # Threshold for recharge (ndc * smax)
    threshold_s1 = ndc * smax

    # flux_qdr: Recharge from S3 to S1
    # recharge_5(c, threshold, S_source, S_receiver)
    flux_qdr = recharge_5(c, threshold_s1, S3, S1, nearzero=nearzero)
    flux_qdr = torch.minimum(flux_qdr, S3 - nearzero)
    flux_qdr = F.relu(flux_qdr)

    # Interim update for S3
    S3_tmp = S3 - flux_qdr
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    # --- 2. Soil Moisture Store Processes (S1) ---
    # Potential inflow to S1
    S1_in = P + flux_qdr

    # flux_qs: Saturation excess runoff (Fast process)
    flux_qs = saturation_1(S1_in, S1, smax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qs)
    flux_qs = torch.clamp(flux_qs, min=zeros, max=S1_in)

    # Update S1 for evaporation
    S1_tmp = S1 + S1_in - flux_qs
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_ea: Evaporation from S1
    # evap_20(emax, ndc, S, Smax, Ep)
    flux_ea = evap_20(emax, ndc, S1_tmp, smax, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = torch.minimum(flux_ea, PET)
    flux_ea = F.relu(flux_ea)

    # Update S1 for infiltration to S2
    S1_tmp2 = S1_tmp - flux_ea
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # flux_f: Infiltration from S1 to S2
    # interflow_11(frate, threshold, S)
    flux_f = interflow_11(frate, threshold_s1, S1_tmp2, nearzero=nearzero)
    flux_f = torch.minimum(flux_f, S1_tmp2 - nearzero)
    flux_f = F.relu(flux_f)

    # Final S1 update
    S1_new = S1_tmp2 - flux_f
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 3. Intermediate Store Processes (S2) ---
    # Inflow to S2 is infiltration flux_f
    S2_tmp_in = S2 + flux_f
    S2_tmp_in = torch.clamp(S2_tmp_in, min=nearzero)

    # flux_qb: Baseflow from S2 (Slow process 1)
    # baseflow_9(coeff, threshold, S)
    flux_qb = baseflow_9(b * dpf, sdrmax, S2_tmp_in, nearzero=nearzero)
    flux_qb = torch.minimum(flux_qb, S2_tmp_in - nearzero)
    flux_qb = F.relu(flux_qb)

    # Update S2 for percolation to S3
    S2_tmp_perc = S2_tmp_in - flux_qb
    S2_tmp_perc = torch.clamp(S2_tmp_perc, min=nearzero)

    # flux_dp: Percolation from S2 to S3 (Slow process 2)
    # baseflow_1(coeff, S)
    flux_dp = baseflow_1((1.0 - b) * dpf, S2_tmp_perc, nearzero=nearzero)
    flux_dp = torch.minimum(flux_dp, S2_tmp_perc - nearzero)
    flux_dp = F.relu(flux_dp)

    # Final S2 update
    S2_new = S2_tmp_perc - flux_dp
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 4. Saturated Zone Store Update (S3) ---
    # S3 receives percolation flux_dp
    S3_new = S3_tmp + flux_dp
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 5. Output Aggregation ---
    # Qsim = qs (surface) + qb (baseflow)
    # Ea = ea
    Qsim = flux_qs + flux_qb
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new, S3_new
