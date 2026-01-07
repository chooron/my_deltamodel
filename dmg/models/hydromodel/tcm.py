import torch
import torch.nn.functional as F
from typing import Tuple
from .flux.effective import effective_1
from .flux.split import split_1
from .flux.evap import evap_1, evap_16
from .flux.saturation import saturation_1, saturation_9
from .flux.baseflow import baseflow_1, baseflow_6

# Parameter range dictionary (based on MARRMoT m_25_tcm_6p_4s)
TCM_PARAMS_BOUNDS = {
    "phi": [0.0, 1.0],  # Fraction preferential recharge [-]
    "rc": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "gam": [0.0, 1.0],  # Fraction of Ep reduction with depth [-]
    "k1": [0.0, 1.0],  # Runoff coefficient [d-1]
    "ca": [0.0, 10.0],  # Abstraction rate [mm/d] (Derived from fa * mean(P))
    "k2": [0.0, 1.0],  # Runoff coefficient [mm-1 d-1]
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
    phi: torch.Tensor, # Fraction preferential recharge [-]
    rc: torch.Tensor,  # Maximum soil moisture depth (Applied to S1) [mm]
    gam: torch.Tensor, # Fraction of Ep reduction with depth [-]
    k1: torch.Tensor,  # Runoff coefficient [d-1]
    ca: torch.Tensor,  # Abstraction rate [mm/d]
    k2: torch.Tensor,  # Runoff coefficient [mm-1 d-1]
    # State variables
    S1: torch.Tensor,  # Upper soil moisture store
    S2: torch.Tensor,  # Soil moisture deficit store (0 = fully saturated)
    S3: torch.Tensor,  # Fast routing reservoir
    S4: torch.Tensor,  # Slow routing reservoir
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Thames Catchment Model (TCM) single-step calculation.
    
    Aligned with MARRMoT m_25_tcm_6p_4s.m but optimized for gradients.
    """

    # --- 0. Numerical Guards for Parameters ---
    # Ensure strictly positive capacity to avoid division by zero
    rc = torch.clamp(rc, min=nearzero)

    # --- 1. Effective Precipitation and Splitting ---
    # flux_pn: Effective P (P - Interception Loss)
    flux_pn = effective_1(P, PET, nearzero=nearzero)
    zeros_tensor = torch.zeros_like(flux_pn)
    flux_pn = torch.clamp(flux_pn, min=zeros_tensor, max=P)

    # flux_en: Interception Evaporation
    flux_en = F.relu(P - flux_pn)

    # Split effective precipitation
    # phi goes to S3 (bypass), rest to S1
    flux_pby = split_1(phi, flux_pn, nearzero=nearzero)
    flux_pin = F.relu(flux_pn - flux_pby)

    # --- 2. Upper Store Process (S1) ---
    # MATLAB: flux_qex1 = saturation_1(flux_pin,S1,rc);
    # Meaning: S1 has capacity 'rc'.
    flux_qex1 = saturation_1(flux_pin, S1, rc, nearzero=nearzero)
    
    # Update S1 temp state
    S1_tmp = S1 + flux_pin - flux_qex1
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_ea: Evaporation from S1 (Linear)
    # MATLAB: flux_ea = evap_1(S1,Ep,delta_t);
    flux_ea = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero) # Mass balance constraint
    
    # Final S1 update
    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero) # S1 cannot exceed rc usually, handled by qex1

    # --- 3. Deficit Store Process (S2) ---
    # S2 is a deficit store. S2=0 means saturated.
    
    # flux_qex2: Percolation from S1 (qex1) filling the deficit (S2)
    # MATLAB: flux_qex2 = saturation_9(flux_qex1,S2,0.01);
    # Logic: If Inflow (qex1) > Deficit (S2), Deficit becomes 0, Excess (qex2) flows out.
    flux_qex2 = saturation_9(flux_qex1, S2, torch.tensor(0.01, device=P.device), nearzero=nearzero)

    # flux_et: Transpiration
    # MATLAB: flux_et = evap_16(gam,Inf,S1,0.01,Ep,delta_t);
    # PROBLEM: Inf kills gradients.
    # FIX: Use 'rc' instead of 'Inf'. This implies transpiration depends on S1 saturation (S1/rc).
    # Logic: As S1 fills up (approaches rc), transpiration -> Ep.
    # Note: Using S1_new to be consistent with time-stepping.
    pet_rem = F.relu(PET - flux_ea)
    
    flux_et = evap_16(
        gam,
        rc,     # Capacity
        S1_new,
        torch.zeros_like(rc), # 将阈值改为 0，避免 rc < 0.01 时分母为负
        pet_rem,
        nearzero=nearzero
    )
    flux_et = F.relu(flux_et)

    # Update S2 (Deficit)
    # MATLAB: dS2 = flux_et + flux_qex2 - flux_qex1;
    # Deficit INCREASES with Transpiration (et)
    # Deficit DECREASES with Inflow (qex1) (Net change is + qex2 - qex1)
    S2_new = S2 + flux_et + flux_qex2 - flux_qex1
    S2_new = torch.clamp(S2_new, min=nearzero)
    # Note: S2 technically can grow indefinitely in TCM if dry, but usually bounded by physics implicitly.

    # --- 4. Fast Routing Store (S3) ---
    # MATLAB: dS3 = flux_qex2 + flux_pby - flux_quz;
    inflow_S3 = flux_qex2 + flux_pby
    S3_tmp = S3 + inflow_S3
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    # flux_quz: Linear reservoir
    flux_quz = baseflow_1(k1, S3_tmp, nearzero=nearzero)
    flux_quz = torch.minimum(flux_quz, S3_tmp - nearzero)
    
    S3_new = S3_tmp - flux_quz
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 5. Slow Routing Store (S4) ---
    # MATLAB: dS4 = flux_quz - flux_a - flux_q;
    
    # flux_a: Abstraction (Loss)
    flux_a = torch.minimum(ca, S4 + flux_quz) # Limit to available water
    
    S4_tmp = S4 + flux_quz - flux_a
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)

    # flux_q: Groundwater flow (Non-linear k2 * S^2 usually, baseflow_6)
    # baseflow_6(k, offset, S, dt) -> k * (S - offset)^2
    flux_q = baseflow_6(k2, torch.tensor(0.0, device=P.device), S4_tmp, nearzero=nearzero)
    flux_q = torch.minimum(flux_q, S4_tmp - nearzero)

    S4_new = S4_tmp - flux_q
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 6. Output ---
    Qsim = flux_q
    Ea = flux_en + flux_ea + flux_et

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new