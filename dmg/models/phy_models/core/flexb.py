import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.saturation import saturation_3
from ..flux.evap import evap_3
from ..flux.percolation import percolation_2
from ..flux.split import split_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_21_flexb_9p_3s)
FLEXB_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "beta": [0.0, 10.0],  # Unsaturated zone shape parameter [-]
    "d_split": [0.0, 1.0],  # Fast/slow runoff distribution parameter [-]
    "percmax": [0.0, 20.0],  # Maximum percolation rate [mm/d]
    "lp": [0.05, 0.95],  # Wilting point as fraction of s1max [-]
    "nlagf": [1.0, 5.0],  # Flow delay before fast runoff [d]
    "nlags": [1.0, 15.0],  # Flow delay before slow runoff [d]
    "kf": [0.0, 1.0],  # Fast runoff coefficient [d-1]
    "ks": [0.0, 1.0],  # Slow runoff coefficient [d-1]
}

# Parameter description dictionary
FLEXB_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "beta": "Unsaturated zone shape parameter [-]",
    "d_split": "Fast/slow runoff distribution parameter [-]",
    "percmax": "Maximum percolation rate [mm/d]",
    "lp": "Wilting point as fraction of s1max [-]",
    "nlagf": "Flow delay before fast runoff [d]",
    "nlags": "Flow delay before slow runoff [d]",
    "kf": "Fast runoff coefficient [d-1]",
    "ks": "Slow runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Flex-B model.
    S1: Unsaturated soil moisture store
    S2: Fast routing store
    S3: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def flexb_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching FLEXB_PARAMS_BOUNDS keys
    s1max: torch.Tensor,
    beta: torch.Tensor,
    d_split: torch.Tensor,
    percmax: torch.Tensor,
    lp: torch.Tensor,
    nlagf: torch.Tensor,
    nlags: torch.Tensor,
    kf: torch.Tensor,
    ks: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Flex-B model single-step calculation.

    Model reference:
    Fenicia, F., McDonnell, J. J., & Savenije, H. H. G. (2008).
    Learning from model improvement: On the contribution of complementary
    data to process understanding. Water Resources Research, 44(6).
    """

    # UH parameters are unused (identity routing)
    _ = (nlagf, nlags)

    # --- 1. Unsaturated Zone Processes (S1) ---

    # flux_ru: Infiltration into S1
    # saturation_3 calculates how much of P is infiltrated based on storage
    flux_ru = saturation_3(S1, s1max, beta, P, nearzero=nearzero)
    zeros = torch.zeros_like(flux_ru)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=P)

    # Surface excess (not infiltrated)
    p_excess = F.relu(P - flux_ru)

    # Split surface excess into fast (rf) and slow (rs) components
    # flux_rf = (1-d) * p_excess
    # flux_rs = d * p_excess
    flux_rf = split_1(1.0 - d_split, p_excess, nearzero=nearzero)
    flux_rs = F.relu(p_excess - flux_rf)

    # Update state for evaporation and percolation
    # Sequential discrete update
    S1_tmp = S1 + flux_ru
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_eur: Actual evaporation from S1
    flux_eur = evap_3(lp, S1_tmp, s1max, PET, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S1_tmp - nearzero)
    flux_eur = torch.minimum(flux_eur, PET)
    flux_eur = F.relu(flux_eur)

    S1_tmp2 = S1_tmp - flux_eur
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # flux_ps: Percolation to slow store
    flux_ps = percolation_2(percmax, S1_tmp2, s1max, nearzero=nearzero)
    flux_ps = torch.minimum(flux_ps, S1_tmp2 - nearzero)
    flux_ps = F.relu(flux_ps)

    # Final S1 update
    S1_new = S1_tmp2 - flux_ps
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Routing Processes (S2 and S3) ---
    # Use unit hydrograph (half-triangle) to route fast and slow components.
    flux_rfl = flux_rf
    flux_rsl = flux_ps + flux_rs

    # Fast store process (S2)
    S2_tmp = S2 + flux_rfl
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    flux_qf = baseflow_1(kf, S2_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S2_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S2_new = S2_tmp - flux_qf
    S2_new = torch.clamp(S2_new, min=nearzero)

    # Slow store process (S3)
    S3_tmp = S3 + flux_rsl
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    flux_qs = baseflow_1(ks, S3_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S3_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    S3_new = S3_tmp - flux_qs
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 3. Output Aggregation ---
    # Qsim = qf (fast) + qs (slow)
    # Ea = eur
    Qsim = flux_qf + flux_qs
    Ea = flux_eur

    return Qsim, Ea, S1_new, S2_new, S3_new
