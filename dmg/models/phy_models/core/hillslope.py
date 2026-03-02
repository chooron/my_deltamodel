import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.interception import interception_2
from ..flux.evap import evap_1
from ..flux.saturation import saturation_2
from ..flux.split import split_1
from ..flux.capillary import capillary_2
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_13_hillslope_7p_2s)
HILLSLOPE_PARAMS_BOUNDS = {
    "dw": [0.0, 5.0],  # Interception capacity [mm]
    "betaw": [0.0, 10.0],  # Soil moisture distribution parameter [-]
    "swmax": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "a": [0.0, 1.0],  # Surface/groundwater split fraction [-]
    "th": [1.0, 120.0],  # Routing delay [d]
    "c_rad": [0.0, 4.0],  # Rate of capillary rise [mm/d]
    "kh": [0.0, 1.0],  # Groundwater runoff coefficient [d-1]
}

# Parameter description dictionary
HILLSLOPE_PARAMS_DESC = {
    "dw": "Daily interception capacity [mm]",
    "betaw": "Soil moisture storage distribution parameter [-]",
    "swmax": "Maximum soil moisture storage [mm]",
    "a": "Division parameter for surface and groundwater flow [-]",
    "th": "Time delay for routing [d]",
    "c_rad": "Rate of capillary rise [mm/d]",
    "kh": "Groundwater runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Hillslope model.
    S1: Soil moisture store
    S2: Groundwater store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def hillslope_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching HILLSLOPE_PARAMS_BOUNDS keys
    dw: torch.Tensor,
    betaw: torch.Tensor,
    swmax: torch.Tensor,
    a: torch.Tensor,
    th: torch.Tensor,
    c_rad: torch.Tensor,
    kh: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hillslope model (FLEX-Topo) single-step calculation.

    Model reference:
    Savenije, H. H. G. (2010). Topography driven conceptual modelling
    (FLEX-Topo). Hydrology and Earth System Sciences, 14(12), 2681-2692.
    """

    # Routing parameter th is unused (identity routing)
    _ = th

    # 1. Inflow + Interception
    # flux_pe: interception excess
    # flux_ei: intercepted rainfall (tracks tracks 'intercepted' rainfall for Ea)
    flux_pe = interception_2(P, dw, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pe)

    # 2. Fast Process (Saturation Excess)
    # flux_qse: saturation excess calculation
    flux_qse = saturation_2(S1, swmax, betaw, flux_pe, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=flux_pe)

    # 3. Flow splitting
    # Split qse into surface (qses) and groundwater (qseg) branches
    flux_qses = split_1(a, flux_qse, nearzero=nearzero)
    flux_qseg = F.relu(flux_qse - flux_qses)

    # 4. Sequential state updates (Inflow and Runoff)
    # S1 interim update
    S1_tmp = S1 + flux_pe - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # S2 interim update
    S2_tmp = S2 + flux_qseg
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # 5. Evaporation (Actual ET from S1)
    flux_ea = evap_1(S1_tmp, PET, nearzero=nearzero)
    # Constraint to prevent negative storage
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = torch.minimum(flux_ea, PET)
    flux_ea = F.relu(flux_ea)

    S1_tmp2 = S1_tmp - flux_ea
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # 6. Slow Processes (Capillary Rise and Baseflow)

    # flux_c: capillary rise from S2 to S1
    # capillary_2(p1=c_rad, S2, nearzero)
    flux_c = capillary_2(c_rad, S2_tmp, nearzero=nearzero)
    flux_c = torch.minimum(flux_c, S2_tmp - nearzero)
    flux_c = F.relu(flux_c)

    # S2 update for baseflow
    S2_tmp2 = S2_tmp - flux_c
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # flux_qhgw: baseflow from S2
    flux_qhgw = baseflow_1(kh, S2_tmp2, nearzero=nearzero)
    flux_qhgw = torch.minimum(flux_qhgw, S2_tmp2 - nearzero)
    flux_qhgw = F.relu(flux_qhgw)

    # 7. Final State Updates Mass Balance
    S1_new = S1_tmp2 + flux_c
    S1_new = torch.clamp(S1_new, min=nearzero)

    S2_new = S2_tmp2 - flux_qhgw
    S2_new = torch.clamp(S2_new, min=nearzero)

    # 8. Routing (TODO)
    # TODO: Unit hydrograph routing (route/uh_) not supported yet for flux_qses (routing delay th)
    # flux_qhsrf = route(flux_qses, uh)
    flux_qhsrf = flux_qses  # Assume no delay if routing not implemented

    # 9. Output Aggregation
    Qsim = flux_qhsrf + flux_qhgw
    Ea = flux_ei + flux_ea

    return Qsim, Ea, S1_new, S2_new
