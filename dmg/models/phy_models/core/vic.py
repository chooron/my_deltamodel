import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7
from ..flux.interception import interception_1
from ..flux.excess import excess_1
from ..flux.saturation import saturation_1, saturation_2
from ..flux.effective import effective_1
from ..flux.percolation import percolation_5
from ..flux.baseflow import baseflow_5
from ..flux.phenology import phenology_2

# Parameter range dictionary (based on MARRMoT m_22_vic_10p_3s)
VIC_PARAMS_BOUNDS = {
    "ibar": [0.1, 5.0],  # Mean interception capacity [mm]
    "idelta": [
        0.0,
        1.0,
    ],  # Seasonal interception change as fraction of mean [-]
    "ishift": [1.0, 365.0],  # Maximum interception peak timing [d]
    "stot": [1.0, 2000.0],  # Total available storage [mm]
    "fsm": [0.01, 0.99],  # Fraction of stot that constitutes smmax [-]
    "b": [0.0, 10.0],  # Infiltration excess shape parameter [-]
    "k1": [0.0, 1.0],  # Percolation time parameter [d-1]
    "c1": [0.0, 10.0],  # Percolation non-linearity parameter [-]
    "k2": [0.0, 1.0],  # Baseflow time parameter [d-1]
    "c2": [1.0, 5.0],  # Baseflow non-linearity parameter [-]
}

# Parameter description dictionary
VIC_PARAMS_DESC = {
    "ibar": "Mean interception capacity [mm]",
    "idelta": "Seasonal interception change as fraction of mean [-]",
    "ishift": "Maximum interception peak timing [d]",
    "stot": "Total available storage [mm]",
    "fsm": "Fraction of stot that constitutes maximum soil moisture smmax [-]",
    "b": "Infiltration excess shape parameter [-]",
    "k1": "Percolation time parameter [d-1]",
    "c1": "Percolation non-linearity parameter [-]",
    "k2": "Baseflow time parameter [d-1]",
    "c2": "Baseflow non-linearity parameter [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for VIC model.
    S1: Interception storage
    S2: Soil moisture storage
    S3: Groundwater storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def vic_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # t_idx: torch.Tensor,  # todo t_idx
    # Parameters matching VIC_PARAMS_BOUNDS keys
    ibar: torch.Tensor,
    idelta: torch.Tensor,
    ishift: torch.Tensor,
    stot: torch.Tensor,
    fsm: torch.Tensor,
    b: torch.Tensor,
    k1: torch.Tensor,
    c1: torch.Tensor,
    k2: torch.Tensor,
    c2: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Variable Infiltration Capacity (VIC) model single-step calculation.

    Model reference:
    Liang, X., Lettenmaier, D. P., Wood, E. F., & Burges, S. J. (1994).
    A simple hydrologically based model of land surface water and energy fluxes
    for general circulation models. Journal of Geophysical Research, 99.
    """

    # --- 0. Setup Auxiliary Parameters ---
    # Derived storage capacities
    smmax = fsm * stot
    gwmax = (1.0 - fsm) * stot
    tmax = torch.tensor(
        365.25, device=P.device
    )  # Length of one growing cycle [d]

    # --- 1. Interception Store (S1) ---
    # Interception capacity varies seasonally (Phenology)
    t_idx = torch.ones_like(P)
    aux_imax = phenology_2(ibar, idelta, ishift, t_idx, tmax, nearzero=nearzero)

    # flux_ei: Evaporation from interception
    flux_ei = evap_7(S1, aux_imax, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S1 - nearzero)
    flux_ei = F.relu(flux_ei)

    # flux_peff: Throughfall (Precipitation effectively reaching the soil)
    flux_peff = interception_1(P, S1, aux_imax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_peff)
    flux_peff = torch.clamp(flux_peff, min=zeros, max=P)

    # flux_iex: Interception excess (Overflow when storage capacity is exceeded)
    flux_iex = excess_1(S1 + P - flux_peff, aux_imax, nearzero=nearzero)
    flux_iex = F.relu(flux_iex)

    # Update S1 final
    S1_new = S1 + P - flux_ei - flux_peff - flux_iex
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil Moisture Store (S2) ---
    # Total potential infiltration from above
    potential_inf = flux_peff + flux_iex

    # flux_qie: Infiltration excess runoff (VIC-specific formulation)
    flux_qie = saturation_2(S2, smmax, b, potential_inf, nearzero=nearzero)
    flux_qie = torch.clamp(flux_qie, min=zeros, max=potential_inf)

    # flux_inf: Effective infiltration into the soil store
    flux_inf = effective_1(potential_inf, flux_qie, nearzero=nearzero)

    # flux_et1: Evapotranspiration from soil (uses available potential ET)
    pet_rem_s2 = F.relu(PET - flux_ei)
    flux_et1 = evap_7(S2, smmax, pet_rem_s2, nearzero=nearzero)
    flux_et1 = torch.minimum(flux_et1, S2 + flux_inf - nearzero)
    flux_et1 = torch.minimum(flux_et1, pet_rem_s2)
    flux_et1 = F.relu(flux_et1)

    # flux_qex1: Saturation excess from soil store
    flux_qex1 = saturation_1(flux_inf, S2, smmax, nearzero=nearzero)
    flux_qex1 = torch.clamp(flux_qex1, min=zeros, max=flux_inf)

    # flux_pc: Percolation to groundwater store
    flux_pc = percolation_5(k1, c1, S2, smmax, nearzero=nearzero)

    # Update S2 sequentially
    S2_tmp = S2 + flux_inf - flux_et1 - flux_qex1
    flux_pc = torch.minimum(flux_pc, S2_tmp - nearzero)
    flux_pc = F.relu(flux_pc)

    S2_new = S2_tmp - flux_pc
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Groundwater Store (S3) ---
    # Inflow to S3 is percolation flux_pc

    # flux_et2: Evapotranspiration from groundwater (uses remaining potential ET)
    pet_rem_s3 = F.relu(pet_rem_s2 - flux_et1)
    flux_et2 = evap_7(S3, gwmax, pet_rem_s3, nearzero=nearzero)
    flux_et2 = torch.minimum(flux_et2, S3 + flux_pc - nearzero)
    flux_et2 = torch.minimum(flux_et2, pet_rem_s3)
    flux_et2 = F.relu(flux_et2)

    # flux_qex2: Saturation excess from groundwater store
    flux_qex2 = saturation_1(flux_pc, S3, gwmax, nearzero=nearzero)
    flux_qex2 = torch.clamp(flux_qex2, min=zeros, max=flux_pc)

    # flux_qb: Baseflow from groundwater
    flux_qb = baseflow_5(k2, c2, S3, gwmax, nearzero=nearzero)

    # Update S3 sequentially
    S3_tmp = S3 + flux_pc - flux_et2 - flux_qex2
    flux_qb = torch.minimum(flux_qb, S3_tmp - nearzero)
    flux_qb = F.relu(flux_qb)

    S3_new = S3_tmp - flux_qb
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = Sum of runoff components
    # Ea = Sum of actual ET components
    Qsim = flux_qie + flux_qex1 + flux_qex2 + flux_qb
    Ea = flux_ei + flux_et1 + flux_et2

    return Qsim, Ea, S1_new, S2_new, S3_new
