import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_1, evap_6, evap_5
from ..flux.interception import interception_1
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_9
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_16_newzealand2_8p_2s)
NEWZEALAND2_PARAMS_BOUNDS = {
    "s1max": [0.0, 5.0],  # Maximum interception storage [mm]
    "s2max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "sfc_frac": [0.05, 0.95],  # Field capacity fraction [-]
    "m": [0.05, 0.95],  # Fraction forest [-]
    "a": [0.0, 1.0],  # Subsurface runoff coefficient [d-1]
    "b": [1.0, 5.0],  # Runoff non-linearity [-]
    "tcbf": [0.0, 1.0],  # Baseflow runoff coefficient [d-1]
    "d_delay": [1.0, 30.0],  # Routing time delay [d] (UH base)
}


# Parameter description dictionary
NEWZEALAND2_PARAMS_DESC = {
    "s1max": "Maximum interception storage [mm]",
    "s2max": "Maximum soil moisture storage [mm]",
    "sfc_frac": "Field capacity as fraction of maximum soil moisture [-]",
    "m": "Fraction forest [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "b": "Runoff non-linearity [-]",
    "tcbf": "Baseflow runoff coefficient [d-1]",
    "d_delay": "Routing time delay [d]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for New Zealand model v2.
    S1: Interception storage
    S2: Soil moisture storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def newzealand2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching NEWZEALAND2_PARAMS_BOUNDS keys
    s1max: torch.Tensor,
    s2max: torch.Tensor,
    sfc_frac: torch.Tensor,
    m: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tcbf: torch.Tensor,
    d_delay: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    New Zealand model v2 single-step calculation.

    Model reference:
    Atkinson, S. E., Sivapalan, M., Woods, R. A., & Viney, N. R. (2003).
    Dominant physical controls on hourly flow predictions and the role of
    spatial variability: Mahurangi catchment, New Zealand. Advances in Water
    Resources, 26(3), 219-235.
    """

    # Routing delay d_delay is unused (identity routing)
    _ = d_delay

    # --- 1. Interception process (S1) ---
    # flux_qtf: Throughfall (Fast process for S1)
    flux_qtf = interception_1(P, S1, s1max, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qtf)
    flux_qtf = torch.clamp(flux_qtf, min=zeros, max=P)

    # Update S1 for evaporation
    S1_tmp = S1 + P - flux_qtf
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_eint: Evaporation from interception store
    flux_eint = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_eint = torch.minimum(flux_eint, S1_tmp - nearzero)
    flux_eint = torch.minimum(flux_eint, PET)
    flux_eint = F.relu(flux_eint)

    # Final S1 update
    S1_new = S1_tmp - flux_eint
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil moisture process (S2) ---
    # flux_qse: Saturation excess from throughfall (Fast process for S2)
    flux_qse = saturation_1(flux_qtf, S2, s2max, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=flux_qtf)

    # Update S2 for evaporation
    S2_tmp = S2 + flux_qtf - flux_qse
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Ep remaining after interception evaporation
    pet_rem = F.relu(PET - flux_eint)

    # Evaporation from soil (Vegetated + Bare soil)
    flux_veg = evap_6(m, sfc_frac, S2_tmp, s2max, pet_rem, nearzero=nearzero)
    flux_ebs = evap_5(m, S2_tmp, s2max, pet_rem, nearzero=nearzero)

    flux_ea_s2 = flux_veg + flux_ebs
    flux_ea_s2 = torch.minimum(flux_ea_s2, S2_tmp - nearzero)
    flux_ea_s2 = torch.minimum(flux_ea_s2, pet_rem)
    flux_ea_s2 = F.relu(flux_ea_s2)

    # Update S2 for slow processes
    S2_tmp2 = S2_tmp - flux_ea_s2
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # Slow processes: Interflow and Baseflow
    # flux_qss: Interflow
    sfc_threshold = sfc_frac * s2max
    flux_qss = interflow_9(S2_tmp2, a, sfc_threshold, b, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S2_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # Update S2 before baseflow
    S2_tmp3 = S2_tmp2 - flux_qss
    S2_tmp3 = torch.clamp(S2_tmp3, min=nearzero)

    # flux_qbf: Baseflow
    flux_qbf = baseflow_1(tcbf, S2_tmp3, nearzero=nearzero)
    flux_qbf = torch.minimum(flux_qbf, S2_tmp3 - nearzero)
    flux_qbf = F.relu(flux_qbf)

    # Final S2 update
    S2_new = S2_tmp3 - flux_qbf
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Output Aggregation and Routing ---
    # TODO: Unit hydrograph routing (route/uh_) not supported yet for (qse + qss + qbf)
    flux_q_total = flux_qse + flux_qss + flux_qbf

    Qsim = flux_q_total  # No routing delay for now
    Ea = flux_eint + flux_ea_s2

    return Qsim, Ea, S1_new, S2_new
