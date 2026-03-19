import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.interception import interception_1
from ..flux.evap import evap_1, evap_3
from ..flux.saturation import saturation_3
from ..flux.percolation import percolation_2
from ..flux.split import split_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_26_flexi_10p_4s)
FLEXI_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "beta": [0.0, 10.0],  # Unsaturated zone shape parameter [-]
    "d_split": [0.0, 1.0],  # Fast/slow runoff distribution parameter [-]
    "percmax": [0.0, 20.0],  # Maximum percolation rate [mm/d]
    "lp": [0.05, 0.95],  # Wilting point as fraction of smax [-]
    "nlagf": [1.0, 5.0],  # Flow delay before fast runoff [d]
    "nlags": [1.0, 15.0],  # Flow delay before slow runoff [d]
    "kf": [0.0, 1.0],  # Fast runoff coefficient [d-1]
    "ks": [0.0, 1.0],  # Slow runoff coefficient [d-1]
    "imax": [0.0, 5.0],  # Maximum interception storage [mm]
}

# Parameter description dictionary
FLEXI_PARAMS_DESC = {
    "smax": "Maximum soil moisture storage [mm]",
    "beta": "Unsaturated zone shape parameter [-]",
    "d_split": "Fast/slow runoff distribution parameter [-]",
    "percmax": "Maximum percolation rate [mm/d]",
    "lp": "Wilting point as fraction of smax [-]",
    "nlagf": "Flow delay before fast runoff [d]",
    "nlags": "Flow delay before slow runoff [d]",
    "kf": "Fast runoff coefficient [d-1]",
    "ks": "Slow runoff coefficient [d-1]",
    "imax": "Maximum interception storage [mm]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Flex-I model.
    S1: Interception store
    S2: Soil moisture store
    S3: Fast routing store
    S4: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def flexi_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching FLEXI_PARAMS_BOUNDS keys
    smax: torch.Tensor,
    beta: torch.Tensor,
    d_split: torch.Tensor,
    percmax: torch.Tensor,
    lp: torch.Tensor,
    nlagf: torch.Tensor,
    nlags: torch.Tensor,
    kf: torch.Tensor,
    ks: torch.Tensor,
    imax: torch.Tensor,
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
    Flex-I model single-step calculation.

    Model reference:
    Fenicia, F., McDonnell, J. J., & Savenije, H. H. G. (2008).
    Learning from model improvement: On the contribution of complementary
    data to process understanding. Water Resources Research, 44(6).
    """

    # UH parameters are unused (identity routing)
    _ = (nlagf, nlags)

    # --- 1. Interception Process (S1) ---
    # flux_peff: Throughfall (Saturation excess from S1)
    flux_peff = interception_1(P, S1, imax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_peff)
    flux_peff = torch.clamp(flux_peff, min=zeros, max=P)

    # Update S1 for evaporation
    S1_tmp = S1 + P - flux_peff
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_ei: Evaporation from interception
    flux_ei = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S1_tmp - nearzero)
    flux_ei = F.relu(flux_ei)

    # Final S1 update
    S1_new = S1_tmp - flux_ei
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil Moisture Process (S2) ---
    # flux_ru: Infiltration into S2 soil store
    flux_ru = saturation_3(S2, smax, beta, flux_peff, nearzero=nearzero)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=flux_peff)

    # Surface excess after infiltration
    rem_peff = F.relu(flux_peff - flux_ru)

    # Split excess into fast (rf) and slow (rs) components
    flux_rf = split_1(1.0 - d_split, rem_peff, nearzero=nearzero)
    flux_rs = F.relu(rem_peff - flux_rf)

    # Update S2 for actual ET and percolation
    S2_tmp = S2 + flux_ru
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Remaining PET after interception ET
    PET_rem = F.relu(PET - flux_ei)

    # flux_eur: Evapotranspiration from soil
    flux_eur = evap_3(lp, S2_tmp, smax, PET_rem, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S2_tmp - nearzero)
    flux_eur = F.relu(flux_eur)

    S2_tmp2 = S2_tmp - flux_eur
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # flux_ps: Percolation to slow reservoir
    flux_ps = percolation_2(percmax, S2_tmp2, smax, nearzero=nearzero)
    flux_ps = torch.minimum(flux_ps, S2_tmp2 - nearzero)
    flux_ps = F.relu(flux_ps)

    # Final S2 update
    S2_new = S2_tmp2 - flux_ps
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Routing Processes (S3 and S4) ---

    # TODO: Inner Routing using DplTri3, using nlagf and nlags as delay parameters
    # Instantaneous routing for flux_rf (fast) and (flux_ps + flux_rs) (slow)
    flux_rfl = flux_rf
    flux_rsl = flux_ps + flux_rs

    # S3: Fast Routing Store
    S3_tmp = S3 + flux_rfl
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    flux_qf = baseflow_1(kf, S3_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S3_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S3_new = S3_tmp - flux_qf
    S3_new = torch.clamp(S3_new, min=nearzero)

    # S4: Slow Routing Store
    S4_tmp = S4 + flux_rsl
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)

    flux_qs = baseflow_1(ks, S4_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S4_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    S4_new = S4_tmp - flux_qs
    S4_new = torch.clamp(S4_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = qf + qs
    # Ea = ei + eur
    Qsim = flux_qf + flux_qs
    Ea = flux_ei + flux_eur

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new
