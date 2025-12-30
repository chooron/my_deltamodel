import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.snowfall import snowfall_1
from ..marrmot.rainfall import rainfall_1
from ..marrmot.melt import melt_1
from ..marrmot.evap import evap_7
from ..marrmot.saturation import saturation_1
from ..marrmot.recharge import recharge_3
from ..marrmot.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_30_mopex2_7p_5s)
MOPEX2_PARAMS_BOUNDS = {
    "tcrit": [-3.0, 3.0],      # Snowfall & snowmelt temperature [oC]
    "ddf": [0.0, 20.0],       # Degree-day factor for snowmelt [mm/oC/d]
    "s2max": [1.0, 2000.0],   # Maximum soil moisture storage [mm]
    "tw": [0.0, 1.0],         # Groundwater leakage time [d-1]
    "tu": [0.0, 1.0],         # Slow flow routing response time [d-1]
    "se": [1.0, 2000.0],      # Root zone storage capacity [mm]
    "tc": [0.0, 1.0],         # Mean residence time [d-1]
}

# Parameter description dictionary
MOPEX2_PARAMS_DESC = {
    "tcrit": "Snowfall & snowmelt temperature [oC]",
    "ddf": "Degree-day factor for snowmelt [mm/oC/d]",
    "s2max": "Maximum soil moisture storage [mm]",
    "tw": "Groundwater leakage time [d-1]",
    "tu": "Slow flow routing response time [d-1]",
    "se": "Root zone storage capacity [mm]",
    "tc": "Mean residence time [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for MOPEX-2 model.
    S1: Snow store
    S2: Surface soil moisture store
    S3: Root zone storage
    S4: Fast routing store
    S5: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


def mopex2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching MOPEX2_PARAMS_BOUNDS keys
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    s2max: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    se: torch.Tensor,
    tc: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-2 model single-step calculation.
    
    Model reference:
    Ye, S., Yaeger, M., Coopersmith, E., Cheng, L., & Sivapalan, M. (2012). 
    Exploring the physical controls of regional patterns of flow duration 
    curves - Part 2: Role of seasonality, the regime curve, and associated 
    process controls. Hydrology and Earth System Sciences, 16(11).
    """

    # --- 1. Snow Process (S1) ---
    # flux_ps: snowfall
    flux_ps = snowfall_1(P, T, tcrit, nearzero=nearzero)
    # flux_pr: rainfall
    flux_pr = rainfall_1(P, T, tcrit, nearzero=nearzero)
    
    # flux_qn: snowmelt
    flux_qn = melt_1(ddf, tcrit, T, S1, nearzero=nearzero)
    flux_qn = torch.minimum(flux_qn, S1 - nearzero)
    flux_qn = F.relu(flux_qn)
    
    # Final S1 update
    S1_new = S1 + flux_ps - flux_qn
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Surface Soil Moisture Process (S2) ---
    # Inflow to S2 is rainfall and snowmelt
    inflow_S2 = flux_pr + flux_qn
    
    # flux_q1f: saturation excess runoff (Fast runoff component)
    flux_q1f = saturation_1(inflow_S2, S2, s2max, nearzero=nearzero)
    flux_q1f = torch.clamp(flux_q1f, min=0.0, max=inflow_S2)
    
    # Update S2 for infiltration
    S2_tmp = S2 + inflow_S2 - flux_q1f
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    
    # flux_et1: Actual evaporation from S2
    flux_et1 = evap_7(S2_tmp, s2max, PET, nearzero=nearzero)
    flux_et1 = torch.minimum(flux_et1, S2_tmp - nearzero)
    flux_et1 = F.relu(flux_et1)
    
    S2_tmp2 = S2_tmp - flux_et1
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)
    
    # flux_qw: Groundwater leakage (Recharge to S3)
    flux_qw = recharge_3(tw, S2_tmp2, nearzero=nearzero)
    flux_qw = torch.minimum(flux_qw, S2_tmp2 - nearzero)
    flux_qw = F.relu(flux_qw)
    
    # Final S2 update
    S2_new = S2_tmp2 - flux_qw
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Root Zone Storage Process (S3) ---
    # Inflow is leakage from S2
    S3_tmp = S3 + flux_qw
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    
    # flux_et2: Actual evaporation from S3 (using remaining PET)
    PET_rem = F.relu(PET - flux_et1)
    flux_et2 = evap_7(S3_tmp, se, PET_rem, nearzero=nearzero)
    flux_et2 = torch.minimum(flux_et2, S3_tmp - nearzero)
    flux_et2 = F.relu(flux_et2)
    
    S3_tmp2 = S3_tmp - flux_et2
    S3_tmp2 = torch.clamp(S3_tmp2, min=nearzero)
    
    # flux_q2u: Slow flow routing response time (to S5)
    flux_q2u = baseflow_1(tu, S3_tmp2, nearzero=nearzero)
    flux_q2u = torch.minimum(flux_q2u, S3_tmp2 - nearzero)
    flux_q2u = F.relu(flux_q2u)
    
    # Final S3 update
    S3_new = S3_tmp2 - flux_q2u
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Fast Routing Store (S4) ---
    # Inflow is fast saturation excess runoff q1f
    S4_tmp = S4 + flux_q1f
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    
    # flux_qf: Fast runoff discharge
    flux_qf = baseflow_1(tc, S4_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S4_tmp - nearzero)
    flux_qf = F.relu(flux_qf)
    
    # Final S4 update
    S4_new = S4_tmp - flux_qf
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 5. Slow Routing Store (S5) ---
    # Inflow is subsurface routing flow from root zone q2u
    S5_tmp = S5 + flux_q2u
    S5_tmp = torch.clamp(S5_tmp, min=nearzero)
    
    # flux_qs: Slow runoff discharge (using same tc as fast?? Yes, per MATLAB code)
    flux_qs = baseflow_1(tc, S5_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S5_tmp - nearzero)
    flux_qs = F.relu(flux_qs)
    
    # Final S5 update
    S5_new = S5_tmp - flux_qs
    S5_new = torch.clamp(S5_new, min=nearzero)

    # --- 6. Output Aggregation ---
    # Qsim = Fast discharge + Slow discharge
    # Ea = et1 + et2
    Qsim = flux_qf + flux_qs
    Ea = flux_et1 + flux_et2

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new
