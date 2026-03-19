import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.interception import interception_2
from ..flux.infiltration import infiltration_4
from ..flux.evap import evap_4
from ..flux.capillary import capillary_2
from ..flux.saturation import saturation_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_15_plateau_8p_2s)
PLATEAU_PARAMS_BOUNDS = {
    "fmax": [0.0, 200.0],  # maximum infiltration rate [mm/d]
    "dp": [0.0, 5.0],  # interception capacity [mm]
    "sumax": [1.0, 2000.0],  # soil moisture depth [mm]
    "lp": [0.05, 0.95],  # wilting point as fraction of Sumax [-]
    "p_coeff": [
        0.0,
        1.0,
    ],  # coefficient for moisture constrained evaporation [-]
    "tp": [1.0, 120.0],  # time delay for routing [d]
    "c_rise": [0.0, 4.0],  # capillary rise [mm/d]
    "kp": [0.0, 1.0],  # base flow time parameter [d-1]
}

# Parameter description dictionary
PLATEAU_PARAMS_DESC = {
    "fmax": "Maximum infiltration rate [mm/d]",
    "dp": "Interception capacity [mm]",
    "sumax": "Soil moisture depth [mm]",
    "lp": "Wilting point as fraction of Sumax [-]",
    "p_coeff": "Coefficient for moisture constrained evaporation [-]",
    "tp": "Time delay for routing [d]",
    "c_rise": "Rate of capillary rise [mm/d]",
    "kp": "Base flow time parameter [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Plateau model.
    S1: Unsaturated store
    S2: Saturated store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def plateau_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching PLATEAU_PARAMS_BOUNDS keys
    fmax: torch.Tensor,
    dp: torch.Tensor,
    sumax: torch.Tensor,
    lp: torch.Tensor,
    p_coeff: torch.Tensor,
    tp: torch.Tensor,
    c_rise: torch.Tensor,
    kp: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Plateau model (FLEX-Topo) single-step calculation.

    Model reference:
    Savenije, H. H. G. (2010). Topography driven conceptual modelling
    (FLEX-Topo). Hydrology and Earth System Sciences, 14(12), 2681-2692.
    """

    # 1. Precipitation and Interception
    # flux_pe: interception excess
    # flux_ei: intercepted rainfall (tracks tracks 'intercepted' rainfall for Ea)
    flux_pe = interception_2(P, dp, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pe)

    # 2. Infiltration and Surface Runoff
    # flux_pi: infiltration into S1
    # flux_pie: surface runoff excess (to be routed)
    flux_pi = infiltration_4(flux_pe, fmax, nearzero=nearzero)
    flux_pi = torch.minimum(flux_pi, flux_pe)
    flux_pie = F.relu(flux_pe - flux_pi)

    # 3. Capillary Rise (from S2 to S1)
    flux_c = capillary_2(c_rise, S2, nearzero=nearzero)
    flux_c = torch.minimum(flux_c, S2 - nearzero)
    flux_c = F.relu(flux_c)

    # 4. Evapotranspiration from S1
    # Update S1 with infiltration and capillary rise for ET calculation
    S1_tmp = S1 + flux_pi + flux_c
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # evap_4(Ep, p1=p_coeff, S=S1, p2=lp, Smax=sumax)
    flux_et = evap_4(PET, p_coeff, S1_tmp, lp, sumax, nearzero=nearzero)
    # Apply constraint to prevent negative storage
    flux_et = torch.minimum(flux_et, S1_tmp - nearzero)
    flux_et = torch.minimum(flux_et, PET)
    flux_et = F.relu(flux_et)

    # 5. Percolation / Saturation Excess from S1 to S2
    # S1_tmp2 is storage after ET
    S1_tmp2 = S1_tmp - flux_et
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # flux_r: saturation excess from inflows (flux_pi + flux_c)
    # saturation_1(incoming_flux, S, Smax)
    inflow_s1 = flux_pi + flux_c
    flux_r = saturation_1(inflow_s1, S1_tmp2, sumax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_r)
    flux_r = torch.clamp(flux_r, min=zeros, max=inflow_s1)

    # Final S1 update
    S1_new = S1_tmp2 - flux_r
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 6. Saturated Store process (S2)
    # Update S2 with percolation and capillary rise
    S2_tmp = S2 + flux_r - flux_c
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # flux_qpgw: baseflow from S2
    flux_qpgw = baseflow_1(kp, S2_tmp, nearzero=nearzero)
    flux_qpgw = torch.minimum(flux_qpgw, S2_tmp - nearzero)
    flux_qpgw = F.relu(flux_qpgw)

    # Final S2 update
    S2_new = S2_tmp - flux_qpgw
    S2_new = torch.clamp(S2_new, min=nearzero)

    # 7. Routing and Output (TODO)
    # TODO: Unit hydrograph routing (route/uh_) not supported yet for flux_pie (delay tp)
    flux_qpieo = flux_pie  # Instantaneous routing for now

    Qsim = flux_qpgw + flux_qpieo
    Ea = flux_ei + flux_et

    return Qsim, Ea, S1_new, S2_new
