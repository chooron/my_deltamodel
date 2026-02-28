import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.snowfall import snowfall_1
from ..flux.rainfall import rainfall_1
from ..flux.melt import melt_1
from ..flux.interception import interception_1
from ..flux.evap import evap_1, evap_3
from ..flux.saturation import saturation_3
from ..flux.percolation import percolation_2
from ..flux.split import split_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_34_flexis_12p_5s)
FLEXIS_PARAMS_BOUNDS = {
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
    "tt": [-3.0, 5.0],  # Threshold temperature for snowfall/snowmelt [oC]
    "ddf": [0.0, 20.0],  # Degree-day factor for snowmelt [mm/d/oC]
}

# Parameter description dictionary
FLEXIS_PARAMS_DESC = {
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
    "tt": "Threshold temperature for snowfall/snowmelt [oC]",
    "ddf": "Degree-day factor for snowmelt [mm/d/oC]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Create initial states for Flex-IS model.
    S1: Snow store
    S2: Interception store
    S3: Soil moisture store
    S4: Fast routing store
    S5: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


def flexis_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching FLEXIS_PARAMS_BOUNDS keys
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
    tt: torch.Tensor,
    ddf: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Flex-IS model single-step calculation.

    Model reference:
    Fenicia, F., McDonnell, J. J., & Savenije, H. H. G. (2008). Learning from
    model improvement: On the contribution of complementary data to process
    understanding. Water Resources Research, 44(6).
    """

    # UH parameters are unused (identity routing)
    _ = (nlagf, nlags)

    # --- 1. Snow Process (S1) ---
    # flux_ps: snowfall
    flux_ps = snowfall_1(P, T, tt, nearzero=nearzero)
    # flux_pi: rainfall
    flux_pi = rainfall_1(P, T, tt, nearzero=nearzero)

    # flux_m: snowmelt
    flux_m = melt_1(ddf, tt, T, S1, nearzero=nearzero)
    flux_m = torch.minimum(flux_m, S1 - nearzero)
    flux_m = F.relu(flux_m)

    # Final S1 update
    S1_new = S1 + flux_ps - flux_m
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Interception Process (S2) ---
    # Inflow to S2 is melt and rainfall
    inflow_S2 = flux_m + flux_pi

    # flux_peff: Throughfall (Effective precipitation)
    flux_peff = interception_1(inflow_S2, S2, imax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_peff)
    flux_peff = torch.clamp(flux_peff, min=zeros, max=inflow_S2)

    S2_tmp = S2 + inflow_S2 - flux_peff
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # flux_ei: Evaporation from interception
    flux_ei = evap_1(S2_tmp, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S2_tmp - nearzero)
    flux_ei = F.relu(flux_ei)

    # Final S2 update
    S2_new = S2_tmp - flux_ei
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Soil Moisture Process (S3) ---
    # flux_ru: Infiltration into soil store
    flux_ru = saturation_3(S3, smax, beta, flux_peff, nearzero=nearzero)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=flux_peff)

    # Surface excess after infiltration
    rem_peff = F.relu(flux_peff - flux_ru)

    # Split excess into fast and slow components
    flux_rf = split_1(1.0 - d_split, rem_peff, nearzero=nearzero)
    flux_rs = F.relu(rem_peff - flux_rf)

    S3_tmp = S3 + flux_ru
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    # Remaining PET after interception ET
    PET_rem = F.relu(PET - flux_ei)

    # flux_eur: Transpiration from soil
    flux_eur = evap_3(lp, S3_tmp, smax, PET_rem, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S3_tmp - nearzero)
    flux_eur = F.relu(flux_eur)

    S3_tmp2 = S3_tmp - flux_eur
    S3_tmp2 = torch.clamp(S3_tmp2, min=nearzero)

    # flux_rp: Percolation from soil
    flux_rp = percolation_2(percmax, S3_tmp2, smax, nearzero=nearzero)
    flux_rp = torch.minimum(flux_rp, S3_tmp2 - nearzero)
    flux_rp = F.relu(flux_rp)

    # Final S3 update
    S3_new = S3_tmp2 - flux_rp
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Routing Processes (S4 and S5) ---

    # TODO: Unit hydrograph routing (route/uh_) not supported yet
    # Fast inflow is rf, Slow inflow is rs + rp
    flux_rfl = flux_rf
    flux_rsl = flux_rs + flux_rp

    # S4: Fast Routing Store
    S4_tmp = S4 + flux_rfl
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)

    flux_qf = baseflow_1(kf, S4_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S4_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S4_new = S4_tmp - flux_qf
    S4_new = torch.clamp(S4_new, min=nearzero)

    # S5: Slow Routing Store
    S5_tmp = S5 + flux_rsl
    S5_tmp = torch.clamp(S5_tmp, min=nearzero)

    flux_qs = baseflow_1(ks, S5_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S5_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    S5_new = S5_tmp - flux_qs
    S5_new = torch.clamp(S5_new, min=nearzero)

    # --- 5. Output Aggregation ---
    # Qsim = qf + qs
    # Ea = ei + eur
    Qsim = flux_qf + flux_qs
    Ea = flux_ei + flux_eur

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new
