import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.snowfall import snowfall_1
from ..flux.rainfall import rainfall_1
from ..flux.melt import melt_1
from ..flux.evap import evap_1
from ..flux.saturation import saturation_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_06_alpine1_4p_2s)
ALPINE1_PARAMS_BOUNDS = {
    "tt": [-3.0, 5.0],  # Threshold temperature [Celsius]
    "ddf": [0.0, 20.0],  # Degree-day-factor [mm/d/Celsius]
    "Smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "tc": [0.0, 1.0],  # Runoff coefficient [d-1]
}

# Parameter description dictionary
ALPINE1_PARAMS_DESC = {
    "tt": "Threshold temperature for snowfall/snowmelt [Celsius]",
    "ddf": "Degree-day-factor for snowmelt [mm/d/Celsius]",
    "Smax": "Maximum soil moisture storage [mm]",
    "tc": "Runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Alpine model v1.
    S1: Snow storage
    S2: Soil moisture storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def alpine1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching ALPINE1_PARAMS_BOUNDS keys
    tt: torch.Tensor,
    ddf: torch.Tensor,
    Smax: torch.Tensor,
    tc: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Alpine model v1 single-step calculation.

    Model reference:
    Eder, G., Sivapalan, M., & Nachtnebel, H. P. (2003). Modelling water
    balances in an Alpine catchment through exploitation of emergent
    properties over changing time scales. Hydrological Processes, 17(11),
    2125-2149.
    """

    # --- 1. Snow process (S1) ---
    # Snowfall calculation
    flux_ps = snowfall_1(P, T, tt, nearzero=nearzero)
    # Rainfall calculation
    flux_pr = rainfall_1(P, T, tt, nearzero=nearzero)
    # Snowmelt calculation
    flux_qn = melt_1(ddf, tt, T, S1, nearzero=nearzero)

    # Ensure melt doesn't exceed snow storage
    flux_qn = torch.minimum(flux_qn, S1 + flux_ps - nearzero)
    flux_qn = F.relu(flux_qn)

    # Update snow store S1
    S1_new = S1 + flux_ps - flux_qn
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil moisture process (S2) ---
    # Fast-First, Sequential Update

    # Inflow to soil store: rainfall + melt
    inflow_S2 = flux_pr + flux_qn

    # Saturation excess calculation (Rule: Fast processes)
    flux_qse = saturation_1(inflow_S2, S2, Smax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=inflow_S2)

    # Immediate update for evaporation
    S2_tmp = S2 + inflow_S2 - flux_qse
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Actual evapotranspiration (Rule: Evaporation)
    flux_ea = evap_1(S2_tmp, PET, nearzero=nearzero)
    # Apply constraint to prevent negative storage
    flux_ea = torch.minimum(flux_ea, S2_tmp - nearzero)
    flux_ea = torch.minimum(flux_ea, PET)
    flux_ea = F.relu(flux_ea)

    # Slow process: Baseflow
    S2_tmp2 = S2_tmp - flux_ea
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    flux_qss = baseflow_1(tc, S2_tmp2, nearzero=nearzero)
    # Apply flow constraint
    flux_qss = torch.minimum(flux_qss, S2_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # Update soil store S2
    S2_new = S2_tmp2 - flux_qss
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Output Aggregation ---
    # Qsim = qse (saturation excess) + qss (subsurface/baseflow)
    Qsim = flux_qse + flux_qss
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new
