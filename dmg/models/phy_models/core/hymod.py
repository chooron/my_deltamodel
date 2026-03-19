import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7
from ..flux.saturation import saturation_2
from ..flux.split import split_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_29_hymod_5p_5s)
HYMOD_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "b_exp": [0.0, 10.0],  # Soil depth distribution parameter [-]
    "a_split": [0.0, 1.0],  # Runoff distribution fraction [-]
    "kf": [0.0, 1.0],  # Fast flow time parameter [d-1]
    "ks": [0.0, 1.0],  # Base flow time parameter [d-1]
}

# Parameter description dictionary
HYMOD_PARAMS_DESC = {
    "smax": "Maximum soil moisture storage [mm]",
    "b_exp": "Soil depth distribution parameter [-]",
    "a_split": "Runoff distribution fraction [-]",
    "kf": "Fast flow time parameter [d-1]",
    "ks": "Base flow time parameter [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Create initial states for HyMOD model.
    S1: Soil moisture store
    S2: Fast routing store 1
    S3: Fast routing store 2
    S4: Fast routing store 3
    S5: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


def hymod_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching HYMOD_PARAMS_BOUNDS keys
    smax: torch.Tensor,
    b_exp: torch.Tensor,
    a_split: torch.Tensor,
    kf: torch.Tensor,
    ks: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
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
    """
    HyMOD model single-step calculation.

    Model reference:
    Wagener, T., Boyle, D. P., Lees, M. J., Wheater, H. S., Gupta, Hoshin,
    V., & Sorooshian, S. (2001). A framework for development and application
    of hydrological models. Hydrology and Earth System Sciences, 5, 13-26.
    """

    # --- 1. Soil Moisture Process (S1) ---
    # flux_pe: Saturation excess (Potential runoff)
    # saturation_2(S, Smax, b, P)
    flux_pe = saturation_2(S1, smax, b_exp, P, nearzero=nearzero)
    zeros = torch.zeros_like(flux_pe)
    flux_pe = torch.clamp(flux_pe, min=zeros, max=P)

    # Update S1 for infiltration and excess
    S1_tmp = S1 + P - flux_pe
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_ea: Actual evaporation from soil
    flux_ea = evap_7(S1_tmp, smax, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = F.relu(flux_ea)

    # Final S1 update
    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Runoff Splitting ---
    # Split flux_pe into fast (rf) and slow (rs) components
    flux_pf = split_1(a_split, flux_pe, nearzero=nearzero)
    flux_ps = F.relu(flux_pe - flux_pf)

    # --- 3. Fast Routing Processes (S2, S3, S4 in series) ---
    # Fast Tank 1 (S2)
    S2_tmp = S2 + flux_pf
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    flux_qf1 = baseflow_1(kf, S2_tmp, nearzero=nearzero)
    flux_qf1 = torch.minimum(flux_qf1, S2_tmp - nearzero)
    S2_new = S2_tmp - flux_qf1

    # Fast Tank 2 (S3)
    S3_tmp = S3 + flux_qf1
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    flux_qf2 = baseflow_1(kf, S3_tmp, nearzero=nearzero)
    flux_qf2 = torch.minimum(flux_qf2, S3_tmp - nearzero)
    S3_new = S3_tmp - flux_qf2

    # Fast Tank 3 (S4)
    S4_tmp = S4 + flux_qf2
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    flux_qf3 = baseflow_1(kf, S4_tmp, nearzero=nearzero)
    flux_qf3 = torch.minimum(flux_qf3, S4_tmp - nearzero)
    S4_new = S4_tmp - flux_qf3

    # --- 4. Slow Routing Process (S5) ---
    S5_tmp = S5 + flux_ps
    S5_tmp = torch.clamp(S5_tmp, min=nearzero)
    flux_qs = baseflow_1(ks, S5_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S5_tmp - nearzero)
    S5_new = S5_tmp - flux_qs

    # --- 5. Output Aggregation ---
    # Qsim = Final fast tank output + Slow tank output
    # Ea = ea
    Qsim = flux_qf3 + flux_qs
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new
