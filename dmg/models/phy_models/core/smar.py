import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.effective import effective_1
from ..flux.saturation import saturation_1, saturation_6
from ..flux.infiltration import infiltration_4
from ..flux.evap import evap_13, evap_14
from ..flux.split import split_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_40_smar_8p_6s)
SMAR_PARAMS_BOUNDS = {
    "h_runoff": [0.0, 1.0],  # Maximum fraction of direct runoff [-]
    "y_inf": [0.0, 200.0],  # Infiltration rate [mm/d]
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "c_evap": [0.0, 1.0],  # Evaporation reduction coefficient [-]
    "g_rech": [0.0, 1.0],  # Groundwater recharge coefficient [-]
    "kg": [0.0, 1.0],  # Groundwater time parameter [d-1]
    "n_res": [1.0, 10.0],  # Number of Nash cascade reservoirs [-]
    "nk_delay": [1.0, 120.0],  # Routing delay [d] (n*k)
}

# Parameter description dictionary
SMAR_PARAMS_DESC = {
    "h_runoff": "Maximum fraction of direct runoff [-]",
    "y_inf": "Infiltration rate [mm/d]",
    "smax": "Maximum soil moisture storage [mm]",
    "c_evap": "Evaporation reduction coefficient [-]",
    "g_rech": "Groundwater recharge coefficient [-]",
    "kg": "Groundwater time parameter [d-1]",
    "n_res": "Number of Nash cascade reservoirs [-]",
    "nk_delay": "Routing delay [d] (n*k)",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create initial states for SMAR model.
    S1-S5: Five layers of soil moisture storage
    S6: Groundwater storage
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S6 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5, S6


def smar_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching SMAR_PARAMS_BOUNDS keys
    h_runoff: torch.Tensor,
    y_inf: torch.Tensor,
    smax: torch.Tensor,
    c_evap: torch.Tensor,
    g_rech: torch.Tensor,
    kg: torch.Tensor,
    n_res: torch.Tensor,
    nk_delay: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    S6: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Soil Moisture Accounting and Routing (SMAR) model single-step calculation.

    Model reference:
    O'Connell, P. E., Nash, J. E., & Farrell, J. P. (1970).
    River flow forecasting through conceptual models part II -
    the Brosna catchment at Ferbane. Journal of Hydrology, 10.
    """

    # --- 1. Effective precipitation and baseline evap ---
    flux_pstar = effective_1(P, PET, nearzero=nearzero)
    flux_estar = effective_1(PET, P, nearzero=nearzero)
    flux_evap_base = torch.minimum(P, PET)

    # --- 2. Runoff generation and infiltration ---
    S_tot = S1 + S2 + S3 + S4 + S5
    flux_r1 = saturation_6(h_runoff, S_tot, smax, flux_pstar, nearzero=nearzero)
    zeros = torch.zeros_like(flux_r1)
    flux_r1 = torch.clamp(flux_r1, min=zeros, max=flux_pstar)

    inflow_after_r1 = F.relu(flux_pstar - flux_r1)
    flux_i = infiltration_4(inflow_after_r1, y_inf, nearzero=nearzero)
    flux_i = torch.clamp(flux_i, min=zeros, max=inflow_after_r1)
    flux_r2 = F.relu(inflow_after_r1 - flux_i)

    layer_cap = smax / 5.0

    # --- 3. Soil layers evap + overflow chain ---
    flux_e1 = evap_13(
        c_evap,
        torch.tensor(0.0, device=P.device),
        flux_estar,
        S1,
        nearzero=nearzero,
    )
    flux_e1 = torch.minimum(flux_e1, S1 - nearzero)
    flux_q1 = saturation_1(flux_i, S1, layer_cap, nearzero=nearzero)
    flux_q1 = torch.clamp(flux_q1, min=zeros, max=flux_i)
    S1_new = torch.clamp(S1 + flux_i - flux_e1 - flux_q1, min=nearzero)

    flux_e2 = evap_14(
        c_evap,
        torch.tensor(1.0, device=P.device),
        flux_estar,
        S2,
        S1,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e2 = torch.minimum(flux_e2, S2 - nearzero)
    flux_q2 = saturation_1(flux_q1, S2, layer_cap, nearzero=nearzero)
    flux_q2 = torch.clamp(flux_q2, min=zeros, max=flux_q1)
    S2_new = torch.clamp(S2 + flux_q1 - flux_e2 - flux_q2, min=nearzero)

    flux_e3 = evap_14(
        c_evap,
        torch.tensor(2.0, device=P.device),
        flux_estar,
        S3,
        S2,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e3 = torch.minimum(flux_e3, S3 - nearzero)
    flux_q3 = saturation_1(flux_q2, S3, layer_cap, nearzero=nearzero)
    flux_q3 = torch.clamp(flux_q3, min=zeros, max=flux_q2)
    S3_new = torch.clamp(S3 + flux_q2 - flux_e3 - flux_q3, min=nearzero)

    flux_e4 = evap_14(
        c_evap,
        torch.tensor(3.0, device=P.device),
        flux_estar,
        S4,
        S3,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e4 = torch.minimum(flux_e4, S4 - nearzero)
    flux_q4 = saturation_1(flux_q3, S4, layer_cap, nearzero=nearzero)
    flux_q4 = torch.clamp(flux_q4, min=zeros, max=flux_q3)
    S4_new = torch.clamp(S4 + flux_q3 - flux_e4 - flux_q4, min=nearzero)

    flux_e5 = evap_14(
        c_evap,
        torch.tensor(4.0, device=P.device),
        flux_estar,
        S5,
        S4,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e5 = torch.minimum(flux_e5, S5 - nearzero)
    flux_r3 = saturation_1(flux_q4, S5, layer_cap, nearzero=nearzero)
    flux_r3 = torch.clamp(flux_r3, min=zeros, max=flux_q4)
    S5_new = torch.clamp(S5 + flux_q4 - flux_e5 - flux_r3, min=nearzero)

    # --- 4. Groundwater store ---
    flux_rg = split_1(g_rech, flux_r3, nearzero=nearzero)
    flux_r3star = split_1(1.0 - g_rech, flux_r3, nearzero=nearzero)
    flux_qg = baseflow_1(kg, S6, nearzero=nearzero)
    flux_qg = torch.minimum(flux_qg, S6 - nearzero)
    flux_qg = F.relu(flux_qg)
    S6_new = torch.clamp(S6 + flux_rg - flux_qg, min=nearzero)

    # --- 5. Aggregation (identity UH) ---
    flux_qr = flux_r1 + flux_r2 + flux_r3star
    Qsim = flux_qr + flux_qg
    Ea = flux_evap_base + flux_e1 + flux_e2 + flux_e3 + flux_e4 + flux_e5

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new, S6_new
