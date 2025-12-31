import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.evap import evap_7
from ..marrmot.saturation import saturation_1
from ..marrmot.recharge import recharge_3
from ..marrmot.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_24_mopex1_5p_4s)
MOPEX1_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "tw": [0.0, 1.0],  # Groundwater leakage time [d-1]
    "tu": [0.0, 1.0],  # Slow flow routing response time [d-1]
    "se": [1.0, 2000.0],  # Root zone storage capacity [mm]
    "tc": [0.0, 1.0],  # Mean residence time [d-1]
}

# Parameter description dictionary
MOPEX1_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "tw": "Groundwater leakage time [d-1]",
    "tu": "Slow flow routing response time [d-1]",
    "se": "Root zone storage capacity [mm]",
    "tc": "Mean residence time [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for MOPEX-1 model.
    S1: Surface soil moisture store
    S2: Root zone storage
    S3: Fast routing store
    S4: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def mopex1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching MOPEX1_PARAMS_BOUNDS keys
    s1max: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    se: torch.Tensor,
    tc: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    MOPEX-1 model single-step calculation.

    Model reference:
    Ye, S., Yaeger, M., Coopersmith, E., Cheng, L., & Sivapalan, M. (2012).
    Exploring the physical controls of regional patterns of flow duration
    curves - Part 2: Role of seasonality, the regime curve, and associated
    process controls. Hydrology and Earth System Sciences, 16(11).
    """

    # --- 1. S1 Process (Surface soil moisture) ---
    # Fast runoff (saturation excess)
    flux_q1f = saturation_1(P, S1, s1max, nearzero=nearzero)
    zeros = torch.zeros_like(flux_q1f)
    flux_q1f = torch.clamp(flux_q1f, min=zeros, max=P)

    # Store update after fast runoff
    S1_tmp = S1 + P - flux_q1f
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # Actual ET from S1
    flux_et1 = evap_7(S1_tmp, s1max, PET, nearzero=nearzero)
    flux_et1 = torch.minimum(flux_et1, S1_tmp - nearzero)
    flux_et1 = F.relu(flux_et1)

    S1_tmp2 = S1_tmp - flux_et1
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # Groundwater leakage (Recharge to S2)
    flux_qw = recharge_3(tw, S1_tmp2, nearzero=nearzero)
    flux_qw = torch.minimum(flux_qw, S1_tmp2 - nearzero)
    flux_qw = F.relu(flux_qw)

    # Update S1
    S1_new = S1_tmp2 - flux_qw
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. S2 Process (Root zone storage) ---
    # Inflow is qw
    S2_tmp = S2 + flux_qw
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Actual ET from S2 (using remaining PET)
    PET_rem = F.relu(PET - flux_et1)
    flux_et2 = evap_7(S2_tmp, se, PET_rem, nearzero=nearzero)
    flux_et2 = torch.minimum(flux_et2, S2_tmp - nearzero)
    flux_et2 = F.relu(flux_et2)

    S2_tmp2 = S2_tmp - flux_et2
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # Slow flow routing response (to S4)
    flux_q2u = baseflow_1(tu, S2_tmp2, nearzero=nearzero)
    flux_q2u = torch.minimum(flux_q2u, S2_tmp2 - nearzero)
    flux_q2u = F.relu(flux_q2u)

    # Update S2
    S2_new = S2_tmp2 - flux_q2u
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. S3 Process (Fast routing store) ---
    # Inflow is q1f
    S3_tmp = S3 + flux_q1f
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    # Fast discharge
    flux_qf = baseflow_1(tc, S3_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S3_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    # Update S3
    S3_new = S3_tmp - flux_qf
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. S4 Process (Slow routing store) ---
    # Inflow is q2u
    S4_tmp = S4 + flux_q2u
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)

    # Slow discharge
    flux_qs = baseflow_1(tc, S4_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S4_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    # Update S4
    S4_new = S4_tmp - flux_qs
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 5. Output Aggregation ---
    # Qsim = qf + qs
    # Ea = et1 + et2
    Qsim = flux_qf + flux_qs
    Ea = flux_et1 + flux_et2

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new
