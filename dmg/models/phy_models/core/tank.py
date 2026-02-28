import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_1
from ..flux.interflow import interflow_8
from ..flux.baseflow import baseflow_1
from ..flux.recharge import recharge_3

# Parameter range dictionary (based on MARRMoT m_27_tank_12p_4s)
TANK_PARAMS_BOUNDS = {
    "a0": [0.0, 1.0],           # Time parameter for drainage 1>2 [d-1]
    "b0": [0.0, 1.0],           # Time parameter for drainage 2>3 [d-1]
    "c0": [0.0, 1.0],           # Time parameter for drainage 3>4 [d-1]
    "a1": [0.0, 1.0],           # Time parameter for surface runoff 1 [d-1]
    "fa": [0.0, 1.0],           # Fraction of a1 that is a2 [-]
    "fb": [0.0, 1.0],           # Fraction of a2 that is b1 [-]
    "fc": [0.0, 1.0],           # Fraction of b1 that is c1 [-]
    "fd": [0.0, 1.0],           # Fraction of c1 that is d1 [-]
    "st": [1.0, 2000.0],        # Maximum soil depth (sum of runoff thresholds) [mm]
    "f2": [0.01, 0.99],         # Fraction of st that constitutes threshold t2 [-]
    "f1": [0.01, 0.99],         # Fraction of st-t2 that is added to t2 to find t1 [-]
    "f3": [0.01, 0.99],         # Fraction of st-t1-t2 that constitutes threshold t3 [-]
}

# Parameter description dictionary
TANK_PARAMS_DESC = {
    "a0": "Time parameter for drainage 1>2 [d-1]",
    "b0": "Time parameter for drainage 2>3 [d-1]",
    "c0": "Time parameter for drainage 3>4 [d-1]",
    "a1": "Time parameter for surface runoff 1 [d-1]",
    "fa": "Fraction of a1 that is a2 [-]",
    "fb": "Fraction of a2 that is b1 [-]",
    "fc": "Fraction of b1 that is c1 [-]",
    "fd": "Fraction of c1 that is d1 [-]",
    "st": "Maximum soil depth (sum of runoff thresholds) [mm]",
    "f2": "Fraction of st that constitutes threshold t2 [-]",
    "f1": "Fraction of st-t2 that is added to t2 to find t1 [-]",
    "f3": "Fraction of st-t1-t2 that constitutes threshold t3 [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Tank model.
    S1: Top tank
    S2: Second tank
    S3: Third tank
    S4: Bottom tank
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def tank_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TANK_PARAMS_BOUNDS keys
    a0: torch.Tensor,
    b0: torch.Tensor,
    c0: torch.Tensor,
    a1: torch.Tensor,
    fa: torch.Tensor,
    fb: torch.Tensor,
    fc: torch.Tensor,
    fd: torch.Tensor,
    st: torch.Tensor,
    f2: torch.Tensor,
    f1: torch.Tensor,
    f3: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tank model single-step calculation.
    
    Model reference:
    Sugawara, M. (1995). Tank model. In V. P. Singh (Ed.), Computer models of 
    watershed hydrology.
    """

    # --- 0. Auxiliary Parameter Setup ---
    # Thresholds
    t2 = f2 * st
    t1 = t2 + f1 * F.relu(st - t2)
    t3 = f3 * F.relu(st - t1)
    t4 = F.relu(st - t1 - t3)
    
    # Time parameters for runoff
    a2 = fa * a1
    b1 = fb * a2
    c1 = fc * b1
    d1 = fd * c1

    # --- 1. S1 Process (Top Tank) ---
    S1_tmp = S1 + P
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)
    
    # Runoff from S1 (two holes)
    flux_y1 = interflow_8(S1_tmp, a1, t1, nearzero=nearzero)
    flux_y1 = torch.minimum(flux_y1, S1_tmp - nearzero)
    
    S1_tmp_y1 = S1_tmp - flux_y1
    S1_tmp_y1 = torch.clamp(S1_tmp_y1, min=nearzero)
    
    flux_y2 = interflow_8(S1_tmp_y1, a2, t2, nearzero=nearzero)
    flux_y2 = torch.minimum(flux_y2, S1_tmp_y1 - nearzero)
    
    S1_tmp_y2 = S1_tmp_y1 - flux_y2
    S1_tmp_y2 = torch.clamp(S1_tmp_y2, min=nearzero)
    
    # Drainage to S2
    flux_f12 = recharge_3(a0, S1_tmp_y2, nearzero=nearzero)
    flux_f12 = torch.minimum(flux_f12, S1_tmp_y2 - nearzero)
    
    S1_tmp_f12 = S1_tmp_y2 - flux_f12
    S1_tmp_f12 = torch.clamp(S1_tmp_f12, min=nearzero)
    
    # Evaporation from S1
    flux_e1 = evap_1(S1_tmp_f12, PET, nearzero=nearzero)
    flux_e1 = torch.minimum(flux_e1, S1_tmp_f12 - nearzero)
    flux_e1 = torch.minimum(flux_e1, PET)
    flux_e1 = F.relu(flux_e1)
    
    # Final S1 update
    S1_new = S1_tmp_f12 - flux_e1
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. S2 Process (Second Tank) ---
    S2_tmp = S2 + flux_f12
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    
    # Runoff from S2
    flux_y3 = interflow_8(S2_tmp, b1, t3, nearzero=nearzero)
    flux_y3 = torch.minimum(flux_y3, S2_tmp - nearzero)
    
    S2_tmp_y3 = S2_tmp - flux_y3
    S2_tmp_y3 = torch.clamp(S2_tmp_y3, min=nearzero)
    
    # Drainage to S3
    flux_f23 = recharge_3(b0, S2_tmp_y3, nearzero=nearzero)
    flux_f23 = torch.minimum(flux_f23, S2_tmp_y3 - nearzero)
    
    S2_tmp_f23 = S2_tmp_y3 - flux_f23
    S2_tmp_f23 = torch.clamp(S2_tmp_f23, min=nearzero)
    
    # Evaporation from S2
    pet_rem_s2 = F.relu(PET - flux_e1)
    flux_e2 = evap_1(S2_tmp_f23, pet_rem_s2, nearzero=nearzero)
    flux_e2 = torch.minimum(flux_e2, S2_tmp_f23 - nearzero)
    flux_e2 = F.relu(flux_e2)
    
    # Final S2 update
    S2_new = S2_tmp_f23 - flux_e2
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. S3 Process (Third Tank) ---
    S3_tmp = S3 + flux_f23
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    
    # Runoff from S3
    flux_y4 = interflow_8(S3_tmp, c1, t4, nearzero=nearzero)
    flux_y4 = torch.minimum(flux_y4, S3_tmp - nearzero)
    
    S3_tmp_y4 = S3_tmp - flux_y4
    S3_tmp_y4 = torch.clamp(S3_tmp_y4, min=nearzero)
    
    # Drainage to S4
    flux_f34 = recharge_3(c0, S3_tmp_y4, nearzero=nearzero)
    flux_f34 = torch.minimum(flux_f34, S3_tmp_y4 - nearzero)
    
    S3_tmp_f34 = S3_tmp_y4 - flux_f34
    S3_tmp_f34 = torch.clamp(S3_tmp_f34, min=nearzero)
    
    # Evaporation from S3
    pet_rem_s3 = F.relu(pet_rem_s2 - flux_e2)
    flux_e3 = evap_1(S3_tmp_f34, pet_rem_s3, nearzero=nearzero)
    flux_e3 = torch.minimum(flux_e3, S3_tmp_f34 - nearzero)
    flux_e3 = F.relu(flux_e3)
    
    # Final S3 update
    S3_new = S3_tmp_f34 - flux_e3
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. S4 Process (Bottom Tank) ---
    S4_tmp = S4 + flux_f34
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    
    # Runoff from S4
    flux_y5 = baseflow_1(d1, S4_tmp, nearzero=nearzero)
    flux_y5 = torch.minimum(flux_y5, S4_tmp - nearzero)
    
    S4_tmp_y5 = S4_tmp - flux_y5
    S4_tmp_y5 = torch.clamp(S4_tmp_y5, min=nearzero)
    
    # Evaporation from S4
    pet_rem_s4 = F.relu(pet_rem_s3 - flux_e3)
    flux_e4 = evap_1(S4_tmp_y5, pet_rem_s4, nearzero=nearzero)
    flux_e4 = torch.minimum(flux_e4, S4_tmp_y5 - nearzero)
    flux_e4 = F.relu(flux_e4)
    
    # Final S4 update
    S4_new = S4_tmp_y5 - flux_e4
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 5. Output Aggregation ---
    # Qsim = all surface, intermediate, and base flow components
    # Ea = all evaporation components
    Qsim = flux_y1 + flux_y2 + flux_y3 + flux_y4 + flux_y5
    Ea = flux_e1 + flux_e2 + flux_e3 + flux_e4

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new
