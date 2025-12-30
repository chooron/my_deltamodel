import torch
from typing import Optional, Tuple
from ..marrmot.evap import evap_7, evap_3
from ..marrmot.saturation import saturation_1
from ..marrmot.interflow import interflow_9
from ..marrmot.baseflow import baseflow_2
from ..marrmot.split import split_1

# Parameter range dictionary (matching MARRMoT m_11_collie3_6p_2s)
COLLIE_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],       # Smax, Maximum soil moisture storage [mm]
    "fc": [0.05, 0.95],          # fc, Field capacity as fraction of Smax [-]
    "a": [0.0, 1.0],             # a, Subsurface runoff coefficient [d-1]
    "m": [0.05, 0.95],           # M, Fraction forest cover [-]
    "b": [1.0, 5.0],             # b, Non-linearity coefficient [-]
    "lambda_par": [0.0, 1.0],    # lambda, Flow distribution parameter [-]
}

# Parameter physical descriptions
COLLIE_PARAMS_DESC = {
    "smax": "Maximum soil moisture storage [mm]",
    "fc": "Field capacity as fraction of Smax [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "m": "Fraction forest cover [-]",
    "b": "Non-linearity coefficient [-]",
    "lambda_par": "Flow distribution parameter [-]",
}

def create_initial_state(
    n_grid: int, 
    nmul: int, 
    device: torch.device, 
    nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Collie3 model.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2

def collie3_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching dict keys exactly
    smax: torch.Tensor,
    fc: torch.Tensor,
    a: torch.Tensor,
    m: torch.Tensor,
    b: torch.Tensor,
    lambda_par: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collie River v3 model single step calculation.
    """
    # --- 1. S1 Process (Soil Moisture) ---
    # Step 1: Inflow + Saturation Excess Runoff
    # saturation_1(P, S, Smax) handles the overflow calculation
    flux_qse = saturation_1(P, S1, smax)
    S1 = S1 + P - flux_qse
    S1 = torch.clamp(S1, min=nearzero)
    
    # Step 2: Evapotranspiration
    # flux_eb (non-vegetated): evap_7(S1, smax, (1-M)*PET)
    flux_eb = evap_7(S1, smax, (1.0 - m) * PET)
    S1 = torch.clamp(S1 - flux_eb, min=nearzero)
    
    # flux_ev (vegetated): evap_3(Sfc, S1, Smax, M*PET)
    # Note: In MARRMoT m_11, evap_3 uses Sfc as the threshold (LP)
    flux_ev = evap_3(fc, S1, smax, m * PET)
    S1 = torch.clamp(S1 - flux_ev, min=nearzero)
    
    # Step 3: Interflow
    # flux_qss = interflow_9(S1, a, Sfc*S1max, b)
    sfc_mm = fc * smax
    flux_qss = interflow_9(S1, a, sfc_mm, b)
    S1 = torch.clamp(S1 - flux_qss, min=nearzero)
    
    # Split interflow
    # flux_qsss goes to S2, flux_qss_direct goes to channel
    flux_qsss = split_1(lambda_par, flux_qss)
    flux_qss_direct = split_1(1.0 - lambda_par, flux_qss)
    
    # --- 2. S2 Process (Groundwater) ---
    # Step 4: S2 Update (Inflow from split interflow)
    S2 = S2 + flux_qsss
    
    # Step 5: Baseflow (Slow Process)
    # flux_qsg = baseflow_2(S2, 1/a, 1/b)
    inv_a = 1.0 / (a + nearzero)
    inv_b = 1.0 / (b + nearzero)
    flux_qsg = baseflow_2(S2, inv_a, inv_b)
    S2 = torch.clamp(S2 - flux_qsg, min=nearzero)
    
    # --- 3. Output Aggregation ---
    # Q = qse + (1-lambda)*qss + qsg
    Q = flux_qse + flux_qss_direct + flux_qsg
    Ea = flux_eb + flux_ev
    
    return Q, Ea, S1, S2

