import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.evap import evap_12
from ..marrmot.saturation import saturation_5
from ..marrmot.split import split_1

# Parameter range dictionary (based on MARRMoT m_05_ihacres_7p_1s)
IHACRES_PARAMS_BOUNDS = {
    "lp": [1.0, 2000.0],  # Wilting point [mm]
    "d": [1.0, 2000.0],  # Threshold for flow generation [mm]
    "p": [0.0, 10.0],  # Flow response non-linearity [-]
    "alpha": [0.0, 1.0],  # Fast/slow flow division [-]
    "tau_q": [1.0, 700.0],  # Fast flow routing delay [d]
    "tau_s": [1.0, 700.0],  # Slow flow routing delay [d]
}

# Parameter description dictionary
IHACRES_PARAMS_DESC = {
    "lp": "Wilting point [mm]",
    "d": "Threshold for flow generation [mm]",
    "p": "Flow response non-linearity [-]",
    "alpha": "Fast/slow flow division [-]",
    "tau_q": "Fast flow routing delay [d]",
    "tau_s": "Slow flow routing delay [d]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """
    Create initial state for IHACRES model.
    Note: S1 is a deficit store.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def ihacres_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching IHACRES_PARAMS_BOUNDS keys
    lp: torch.Tensor,
    d: torch.Tensor,
    p: torch.Tensor,
    alpha: torch.Tensor,
    tau_q: torch.Tensor,
    tau_s: torch.Tensor,
    # State variable (Deficit store)
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    IHACRES model single-step calculation.

    Model references:
    Croke, B. F. W., & Jakeman, A. J. (2004). A catchment moisture deficit
    module for the IHACRES rainfall-runoff model. Environmental Modelling and
    Software, 19(1), 1-5.
    """

    # 1. Evapotranspiration calculation (from deficit store)
    # flux_ea = evap_12(S1, lp, PET)
    flux_ea = evap_12(S1, lp, PET, nearzero=nearzero)
    flux_ea = F.relu(flux_ea)

    # 2. Flow generation (effective rainfall)
    # flux_u = saturation_5(S1, d, p, P)
    flux_u = saturation_5(S1, d, p, P, nearzero=nearzero)
    zeros = torch.zeros_like(flux_u)
    flux_u = torch.clamp(flux_u, min=zeros, max=P)

    # 3. Flow splitting (Fast/Slow)
    # TODO: Unit hydrograph routing (route/uh_) not supported yet
    # flux_uq = split_1(alpha, flux_u)
    # flux_us = split_1(1-alpha, flux_u)
    # The routed flows (xq, xs, xt) are not implemented.
    # Qsim currently returns the total generated flow u.
    flux_uq = split_1(alpha, flux_u, nearzero=nearzero)
    flux_us = split_1(1.0 - alpha, flux_u, nearzero=nearzero)

    # 4. State update (Deficit store S1)
    # dS1 = -P + flux_ea + flux_u
    S1_new = S1 - P + flux_ea + flux_u

    # Normally deficit stores can be negative (representing surplus) or positive (deficit)
    # but we usually keep a lower bound or handle physical limits based on model logic.
    # In IHACRES, S1 is catchment moisture deficit.

    # 5. Output aggregation
    Qsim = flux_u  # Total moisture excess generated this step
    Ea = flux_ea

    return Qsim, Ea, S1_new
