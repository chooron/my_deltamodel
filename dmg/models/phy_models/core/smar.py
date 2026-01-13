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

    # --- 1. Effective Precipitation Calculation ---
    flux_pstar = effective_1(P, PET, nearzero=nearzero)
    flux_estar = effective_1(PET, P, nearzero=nearzero)
    flux_evap_base = torch.minimum(P, PET)

    # --- 2. Surface and Infiltration Processes ---
    # Total soil storage for saturation excess calculation
    S_tot = S1 + S2 + S3 + S4 + S5

    # Direct runoff r1
    # saturation_6(p1, S_tot, Smax, incoming)
    flux_r1 = saturation_6(h_runoff, S_tot, smax, flux_pstar, nearzero=nearzero)
    zeros = torch.zeros_like(flux_r1)
    flux_r1 = torch.clamp(flux_r1, min=zeros, max=flux_pstar)

    # Remaining rainfall available for infiltration
    P_rem = F.relu(flux_pstar - flux_r1)

    # Infiltration into top layer
    # infiltration_4(incoming, infiltration_capacity)
    flux_i = infiltration_4(P_rem, y_inf, nearzero=nearzero)
    flux_i = torch.clamp(flux_i, min=zeros, max=P_rem)

    # Second runoff (excess after infiltration)
    flux_r2 = F.relu(P_rem - flux_i)

    # --- 3. Sequential Updates for Multi-Layer Soil Stores (S1-S5) ---
    # Capacity of each layer is smax/5
    layer_cap = smax / 5.0

    # Layer 1 (S1)
    S1_tmp = S1 + flux_i
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)
    flux_e1 = evap_13(
        c_evap,
        torch.tensor(0.0, device=P.device),
        flux_estar,
        S1_tmp,
        nearzero=nearzero,
    )
    flux_e1 = torch.minimum(flux_e1, S1_tmp - nearzero)
    S1_tmp2 = S1_tmp - flux_e1
    flux_q1 = saturation_1(
        torch.zeros_like(P), S1_tmp2, layer_cap, nearzero=nearzero
    )  # This follows logic: overflow from filling
    # Re-calculating flux_q1 based on inflow:
    flux_q1 = saturation_1(flux_i, S1, layer_cap, nearzero=nearzero)
    flux_q1 = torch.minimum(flux_q1, S1_tmp2 - nearzero)
    S1_new = S1_tmp2 - flux_q1

    # Layer 2 (S2)
    S2_tmp = S2 + flux_q1
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)
    flux_e2 = evap_14(
        c_evap,
        torch.tensor(1.0, device=P.device),
        flux_estar,
        S2_tmp,
        S1_new,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e2 = torch.minimum(flux_e2, S2_tmp - nearzero)
    S2_tmp2 = S2_tmp - flux_e2
    flux_q2 = saturation_1(flux_q1, S2, layer_cap, nearzero=nearzero)
    flux_q2 = torch.minimum(flux_q2, S2_tmp2 - nearzero)
    S2_new = S2_tmp2 - flux_q2

    # Layer 3 (S3)
    S3_tmp = S3 + flux_q2
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)
    flux_e3 = evap_14(
        c_evap,
        torch.tensor(2.0, device=P.device),
        flux_estar,
        S3_tmp,
        S2_new,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e3 = torch.minimum(flux_e3, S3_tmp - nearzero)
    S3_tmp2 = S3_tmp - flux_e3
    flux_q3 = saturation_1(flux_q2, S3, layer_cap, nearzero=nearzero)
    flux_q3 = torch.minimum(flux_q3, S3_tmp2 - nearzero)
    S3_new = S3_tmp2 - flux_q3

    # Layer 4 (S4)
    S4_tmp = S4 + flux_q3
    S4_tmp = torch.clamp(S4_tmp, min=nearzero)
    flux_e4 = evap_14(
        c_evap,
        torch.tensor(3.0, device=P.device),
        flux_estar,
        S4_tmp,
        S3_new,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e4 = torch.minimum(flux_e4, S4_tmp - nearzero)
    S4_tmp2 = S4_tmp - flux_e4
    flux_q4 = saturation_1(flux_q3, S4, layer_cap, nearzero=nearzero)
    flux_q4 = torch.minimum(flux_q4, S4_tmp2 - nearzero)
    S4_new = S4_tmp2 - flux_q4

    # Layer 5 (S5)
    S5_tmp = S5 + flux_q4
    S5_tmp = torch.clamp(S5_tmp, min=nearzero)
    flux_e5 = evap_14(
        c_evap,
        torch.tensor(4.0, device=P.device),
        flux_estar,
        S5_tmp,
        S4_new,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )
    flux_e5 = torch.minimum(flux_e5, S5_tmp - nearzero)
    S5_tmp2 = S5_tmp - flux_e5
    flux_r3 = saturation_1(flux_q4, S5, layer_cap, nearzero=nearzero)
    flux_r3 = torch.minimum(flux_r3, S5_tmp2 - nearzero)
    S5_new = S5_tmp2 - flux_r3

    # --- 4. Groundwater Process (S6) ---
    # Excess from soil split into groundwater recharge and surface routing inflow
    flux_rg = split_1(g_rech, flux_r3, nearzero=nearzero)
    flux_r3star = F.relu(flux_r3 - flux_rg)

    S6_tmp = S6 + flux_rg
    S6_tmp = torch.clamp(S6_tmp, min=nearzero)

    flux_qg = baseflow_1(kg, S6_tmp, nearzero=nearzero)
    flux_qg = torch.minimum(flux_qg, S6_tmp - nearzero)
    S6_new = S6_tmp - flux_qg

    # --- 5. Routing Aggregation ---
    # TODO: Nash cascade routing (nk_delay, n_res) not supported yet.
    # Instantaneous routing for combined runoff components:
    flux_qr = flux_r1 + flux_r2 + flux_r3star

    # Qsim = Routed runoff + Groundwater discharge
    # Ea = Initial overlap evap + All layer ETs
    Qsim = flux_qr + flux_qg
    Ea = flux_evap_base + flux_e1 + flux_e2 + flux_e3 + flux_e4 + flux_e5

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new, S5_new, S6_new
