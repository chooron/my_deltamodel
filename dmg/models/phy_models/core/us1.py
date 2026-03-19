import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.interception import interception_3
from ..flux.evap import evap_8, evap_9, evap_10, evap_5
from ..flux.saturation import saturation_1
from ..flux.excess import excess_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_08_us1_5p_2s)
US1_PARAMS_BOUNDS = {
    "alpha_ei": [0.0, 1.0],  # Fraction of intercepted rainfall [-]
    "m": [0.05, 0.95],  # Fraction forest [-]
    "smax": [1.0, 2000.0],  # Maximum soil moisture [mm]
    "fc": [0.05, 0.95],  # Field capacity as fraction of Smax [-]
    "alpha_ss": [0.0, 1.0],  # Subsurface routing delay [d-1]
}

# Parameter description dictionary
US1_PARAMS_DESC = {
    "alpha_ei": "Fraction of intercepted rainfall [-]",
    "m": "Fraction forest [-]",
    "smax": "Maximum soil moisture [mm]",
    "fc": "Field capacity as fraction of smax [-]",
    "alpha_ss": "Subsurface routing delay [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for US1 model.
    S1: Unsaturated store
    S2: Saturated store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def us1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching US1_PARAMS_BOUNDS keys
    alpha_ei: torch.Tensor,
    m: torch.Tensor,
    smax: torch.Tensor,
    fc: torch.Tensor,
    alpha_ss: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    United States model v1 single-step calculation.

    Model reference:
    Bai, Y., Wagener, T., & Reed, P. (2009). A top-down framework for
    watershed model evaluation and selection under uncertainty. Environmental
    Modelling & Software, 24(8), 901-916.
    """

    # 1. Interception and Soil Recharge
    # flux_eusei: Evaporation from intercepted rainfall
    flux_eusei = interception_3(alpha_ei, P, nearzero=nearzero)
    zeros = torch.zeros_like(flux_eusei)
    flux_eusei = torch.clamp(flux_eusei, min=zeros, max=P)

    # Net precipitation after interception
    p_eff = F.relu(P - flux_eusei)

    # Field capacity of the lower store S2 (influences S1 overflow)
    s_fc_limit = F.relu(fc * (smax - S2))

    # flux_rg: Recharge to S2 (saturation-based)
    flux_rg = saturation_1(p_eff, S1, s_fc_limit, nearzero=nearzero)
    flux_rg = torch.clamp(flux_rg, min=zeros, max=p_eff)

    # flux_se: Overflow recharge to S2
    # excess_1 handles overflow when S > threshold
    flux_se = excess_1(S1 + p_eff - flux_rg, s_fc_limit, nearzero=nearzero)
    flux_se = F.relu(flux_se)

    # 2. Unsaturated zone process (S1 updates and ET)
    # Temporary update for S1 before ET
    S1_tmp = S1 + p_eff - flux_rg - flux_se
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_eusveg: Evapotranspiration from vegetation (unsaturated)
    flux_eusveg = evap_8(S1_tmp, S2, m, s_fc_limit, PET, nearzero=nearzero)
    # flux_eusbs: Evapotranspiration from bare soil (unsaturated)
    flux_eusbs = evap_9(S1_tmp, S2, m, smax, PET, nearzero=nearzero)

    # Limit unsaturated ET by available water in S1
    flux_ea_s1 = flux_eusveg + flux_eusbs
    flux_ea_s1 = torch.minimum(flux_ea_s1, S1_tmp - nearzero)
    flux_ea_s1 = F.relu(flux_ea_s1)

    # Update S1
    S1_new = S1_tmp - flux_ea_s1
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 3. Saturated zone process (S2 updates and ET)
    # Total inflow to S2
    qin_s2 = flux_rg + flux_se

    # flux_qse: Saturation excess runoff from S2
    flux_qse = saturation_1(qin_s2, S2, smax, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=qin_s2)

    # Temporary update for S2 before ET
    S2_tmp = S2 + qin_s2 - flux_qse
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # flux_esatveg: Evapotranspiration from vegetation (saturated)
    # flux_esatbs: Evapotranspiration from bare soil (saturated)
    # Note: Using total storage S1+S2 as reference for some of these functions
    s_total = S1_new + S2_tmp
    flux_esatveg = evap_10(m, S2_tmp, s_total, PET, nearzero=nearzero)
    flux_esatbs = evap_5(m, S2_tmp, s_total, PET, nearzero=nearzero)

    # Limit saturated ET by available water in S2
    flux_ea_s2 = flux_esatveg + flux_esatbs
    flux_ea_s2 = torch.minimum(flux_ea_s2, S2_tmp - nearzero)
    flux_ea_s2 = F.relu(flux_ea_s2)

    # 4. Final S2 update and baseflow
    S2_tmp2 = S2_tmp - flux_ea_s2
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # flux_qss: Subsurface flow/Baseflow
    flux_qss = baseflow_1(alpha_ss, S2_tmp2, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S2_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # Update S2
    S2_new = S2_tmp2 - flux_qss
    S2_new = torch.clamp(S2_new, min=nearzero)

    # 5. Output Aggregation
    # Qsim = qse (Fast) + qss (Slow)
    # Ea = sum of all ET components
    Qsim = flux_qse + flux_qss
    Ea = flux_eusei + flux_ea_s1 + flux_ea_s2

    return Qsim, Ea, S1_new, S2_new
