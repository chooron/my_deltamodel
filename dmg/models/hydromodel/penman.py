import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.evap import evap_1, evap_16
from ..marrmot.saturation import saturation_1, saturation_9
from ..marrmot.split import split_1
from ..marrmot.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_17_penman_4p_3s)
PENMAN_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "phi": [0.0, 1.0],       # Fraction of direct runoff [-]
    "gam": [0.0, 1.0],       # Evaporation reduction in lower zone [-]
    "k1": [0.0, 1.0],        # Runoff coefficient [d-1]
}

# Parameter description dictionary
PENMAN_PARAMS_DESC = {
    "smax": "Maximum soil moisture storage [mm]",
    "phi": "Fraction of direct runoff [-]",
    "gam": "Evaporation reduction in lower zone [-]",
    "k1": "Runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Penman model.
    S1: Upper soil moisture store
    S2: Lower zone deficit store (Deficit store)
    S3: Groundwater/Routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def penman_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching PENMAN_PARAMS_BOUNDS keys
    smax: torch.Tensor,
    phi: torch.Tensor,
    gam: torch.Tensor,
    k1: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Penman model single-step calculation.
    
    Model reference:
    Penman, H. L. (1950). the Dependence of Transpiration on Weather and Soil
    Conditions. Journal of Soil Science, 1(1), 74-89.
    """

    # --- 1. Upper Store Process (S1) ---
    # flux_qex: Saturation excess from precipitation (Fast process)
    flux_qex = saturation_1(P, S1, smax, nearzero=nearzero)
    flux_qex = torch.clamp(flux_qex, min=0.0, max=P)
    
    # Split flux_qex into direct runoff (u1) and recharge to lower zone (q12)
    flux_u1 = split_1(phi, flux_qex, nearzero=nearzero)
    flux_q12 = F.relu(flux_qex - flux_u1)
    
    # Evaporation from upper store (ea)
    S1_tmp = S1 + P - flux_qex
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)
    
    flux_ea = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = F.relu(flux_ea)
    
    # Update S1 final
    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Lower Zone / Deficit Store Process (S2) ---
    # flux_u2: Saturated recharge that overflows directly to groundwater (saturated lower zone)
    # saturation_9(incoming, S_deficit, threshold_deficit)
    flux_u2 = saturation_9(flux_q12, S2, torch.tensor(0.01, device=P.device), nearzero=nearzero)
    flux_u2 = torch.clamp(flux_u2, min=0.0, max=flux_q12)
    
    # Ep remaining after upper store evaporation
    pet_rem = F.relu(PET - flux_ea)
    
    # flux_et: Transpiration from lower zone (increases deficit)
    # evap_16(p1=gam, S1=Inf, S2=S1_new, S2min=0.01, Ep=pet_rem)
    # Note: In MARRMoT m_17, S1 is passed as Inf, S2 is the store being depleted.
    inf_tensor = torch.full_like(S1, float('inf'))
    flux_et = evap_16(gam, inf_tensor, S1_new, torch.tensor(0.01, device=P.device), pet_rem, nearzero=nearzero)
    # Since S2 is a deficit store, we usually limit ET to prevent deficit exceeding a physical limit,
    # but Penman logic typically treats S2 as a potentially unbounded deficit or uses specific limits.
    flux_et = F.relu(flux_et)
    
    # Update S2 (Deficit increases with ET and decreases with recharge/excess)
    # dS2 = flux_et + flux_u2 - flux_q12
    S2_new = S2 + flux_et + flux_u2 - flux_q12
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Groundwater/Routing Store Process (S3) ---
    # Inflow to S3: direct runoff (u1) and lower zone excess (u2)
    inflow_S3 = flux_u1 + flux_u2
    
    S3_tmp = S3 + inflow_S3
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    
    # flux_q: Groundwater discharge (baseflow)
    flux_q = baseflow_1(k1, S3_tmp, nearzero=nearzero)
    flux_q = torch.minimum(flux_q, S3_tmp - nearzero)
    flux_q = F.relu(flux_q)
    
    # Update S3 final
    S3_new = S3_tmp - flux_q
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = q (Final streamflow)
    # Ea = ea (Upper) + et (Lower)
    Qsim = flux_q
    Ea = flux_ea + flux_et

    return Qsim, Ea, S1_new, S2_new, S3_new