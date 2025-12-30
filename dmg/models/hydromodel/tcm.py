import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.effective import effective_1
from ..marrmot.split import split_1
from ..marrmot.evap import evap_1, evap_16
from ..marrmot.saturation import saturation_1, saturation_9
from ..marrmot.baseflow import baseflow_1, baseflow_6

# Parameter range dictionary (based on MARRMoT m_25_tcm_6p_4s)
TCM_PARAMS_BOUNDS = {
    "phi": [0.0, 1.0],      # Fraction preferential recharge [-]
    "rc": [1.0, 2000.0],    # Maximum soil moisture depth [mm]
    "gam": [0.0, 1.0],      # Fraction of Ep reduction with depth [-]
    "k1": [0.0, 1.0],       # Runoff coefficient [d-1]
    "ca": [0.0, 10.0],      # Abstraction rate [mm/d] (Derived from fa * mean(P))
    "k2": [0.0, 1.0],       # Runoff coefficient [mm-1 d-1]
}

# Parameter description dictionary
TCM_PARAMS_DESC = {
    "phi": "Fraction preferential recharge [-]",
    "rc": "Maximum soil moisture depth [mm]",
    "gam": "Fraction of Ep reduction with depth [-]",
    "k1": "Runoff coefficient [d-1]",
    "ca": "Abstraction rate [mm/day]",
    "k2": "Runoff coefficient [mm-1 d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for TCM model.
    S1: Upper soil moisture store
    S2: Soil moisture deficit store (0 = fully saturated)
    S3: Fast routing reservoir
    S4: Slow routing reservoir
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def tcm_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TCM_PARAMS_BOUNDS keys
    phi: torch.Tensor,
    rc: torch.Tensor,
    gam: torch.Tensor,
    k1: torch.Tensor,
    ca: torch.Tensor,
    k2: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Thames Catchment Model (TCM) single-step calculation.
    
    Model reference:
    Moore, R. J., & Bell, V. A. (2001). Comparison of rainfall-runoff models 
    for flood forecasting. Part 1: Literature review of models.
    """

    # --- 1. Effective Precipitation and Splitting ---
    # flux_pn: Precipitation effectively contributing to moisture/flow
    flux_pn = effective_1(P, PET, nearzero=nearzero)
    flux_pn = torch.clamp(flux_pn, min=0.0, max=P)
    
    # flux_en: Portion of P that "evaporates" before reaching soil (per MATLAB code index 2)
    flux_en = F.relu(P - flux_pn)
    
    # Split effective precipitation
    flux_pby = split_1(phi, flux_pn, nearzero=nearzero) # preferential recharge to S3
    flux_pin = F.relu(flux_pn - flux_pby)              # infiltration to S1

    # --- 2. Upper Store Process (S1) ---
    # flux_qex1: Saturation excess from upper store
    flux_qex1 = saturation_1(flux_pin, S1, rc, nearzero=nearzero)
    flux_qex1 = torch.clamp(flux_qex1, min=0.0, max=flux_pin)
    
    # Interim update for ET
    S1_tmp = S1 + flux_pin - flux_qex1
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)
    
    # flux_ea: Evaporation from S1
    flux_ea = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = torch.minimum(flux_ea, PET)
    flux_ea = F.relu(flux_ea)
    
    # Final S1 update
    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 3. Deficit Store Process (S2) ---
    # S2 is a deficit store. S2=0 means saturated.
    # flux_qex2: Percolation to saturated routing (S3) when deficit is filled
    # saturation_9(incoming, S_deficit, threshold_deficit)
    flux_qex2 = saturation_9(flux_qex1, S2, torch.tensor(0.01, device=P.device), nearzero=nearzero)
    flux_qex2 = torch.clamp(flux_qex2, min=0.0, max=flux_qex1)
    
    # flux_et: Transpiration from deficit store (increases deficit)
    # Ep remaining after S1 ea
    pet_rem = F.relu(PET - flux_ea)
    inf_tensor = torch.full_like(S1, float('inf'))
    flux_et = evap_16(gam, inf_tensor, S1_new, torch.tensor(0.01, device=P.device), pet_rem, nearzero=nearzero)
    flux_et = F.relu(flux_et)
    
    # Update S2 (Defict increases with ET and qex1-overflow but decreases with recharge)
    # MATLAB: dS2 = et + qex2 - qex1
    S2_new = S2 + flux_et + flux_qex2 - flux_qex1
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 4. Fast Routing Store (S3) ---
    # Inflow is percolation (qex2) and bypass flow (pby)
    inflow_S3 = flux_qex2 + flux_pby
    S3_tmp = S3 + inflow_S3
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    
    # flux_quz: Upper reservoir flow to S4
    flux_quz = baseflow_1(k1, S3_tmp, nearzero=nearzero)
    flux_quz = torch.minimum(flux_quz, S3_tmp - nearzero)
    flux_quz = F.relu(flux_quz)
    
    # Update S3
    S3_new = S3_tmp - flux_quz
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 5. Slow Routing Store (S4) ---
    # Inflow is quz
    # flux_a: Abstraction rate
    flux_a = torch.minimum(ca, S4 + flux_quz - nearzero)
    flux_a = F.relu(flux_a)
    
    S4_tmp = S4 + flux_quz - flux_a
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    
    # flux_q: Groundwater streamflow
    # baseflow_6(p1=k2, p2=0, S) -> out = p1 * S^2
    flux_q = baseflow_6(k2, torch.tensor(0.0, device=P.device), S4_tmp, nearzero=nearzero)
    flux_q = torch.minimum(flux_q, S4_tmp - nearzero)
    flux_q = F.relu(flux_q)
    
    # Update S4
    S4_new = S4_tmp - flux_q
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 6. Output Aggregation ---
    # Qsim = q (Final Slow Flow / Groundwater component)
    # Ea = en + ea + et
    Qsim = flux_q
    Ea = flux_en + flux_ea + flux_et

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new