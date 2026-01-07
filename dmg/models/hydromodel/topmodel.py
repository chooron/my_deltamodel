import torch
import torch.nn.functional as F
from typing import Tuple
from .flux.saturation import saturation_7, saturation_1
from .flux.evap import evap_3
from .flux.interflow import interflow_10
from .flux.baseflow import baseflow_4

# Parameter range dictionary (based on MARRMoT m_14_topmodel_7p_2s)
TOPMODEL_PARAMS_BOUNDS = {
    "suzmax": [1.0, 2000.0],  # Max soil moisture storage in unsatured zone [mm]
    "st": [0.05, 0.95],  # Threshold fraction [-]
    "kd": [0.0, 1.0],  # Leakage coefficient [mm/d]
    "q0": [0.1, 200.0],  # Zero deficit base flow speed [mm/d]
    "f": [0.0, 1.0],  # Baseflow scaling coefficient [mm-1]
    "chi": [1.0, 7.5],  # Gamma distribution parameter [-]
    "phi": [0.1, 5.0],  # Gamma distribution parameter [-]
}

# Parameter description dictionary
TOPMODEL_PARAMS_DESC = {
    "suzmax": "Maximum soil moisture storage in unsaturated zone [mm]",
    "st": "Threshold for flow generation and evap change as fraction of suzmax [-]",
    "kd": "Leakage to saturated zone flow coefficient [mm/d]",
    "q0": "Zero deficit base flow speed [mm/d]",
    "f": "Baseflow scaling coefficient [mm-1]",
    "chi": "Gamma distribution parameter [-]",
    "phi": "Gamma distribution parameter [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for TOPMODEL.
    S1: Unsaturated storage
    S2: Saturated zone deficit (0 = fully saturated)
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def topmodel_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TOPMODEL_PARAMS_BOUNDS keys
    suzmax: torch.Tensor,
    st: torch.Tensor,
    kd: torch.Tensor,
    q0: torch.Tensor,
    f: torch.Tensor,
    chi: torch.Tensor,
    phi: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TOPMODEL single-step calculation.

    Model reference:
    Beven, K., & Kirkby, M. J. (1979). A physically based, variable contributing area
    model of basin hydrology. Hydrological Sciences Bulletin, 24(1), 43-69.
    """

    # --- 1. Saturated Zone Surface Processes (S2) ---

    # mu is fixed (based on Clark et al, 2008)
    mu_fixed = torch.tensor(3.0, device=P.device)
    # lambda_para: Mean of the gamma distribution
    lambda_para = chi * phi + mu_fixed
    # flux_qof: Saturation excess from variable contributing area (Gamma distribution)
    # saturation_7(p1=chi, p2=phi, p3=mu, p4=lambda, p5=f, S=S2, In=P)
    flux_qof = saturation_7(
        chi, phi, mu_fixed, lambda_para, f, S2, P, nearzero=nearzero
    )
    zeros = torch.zeros_like(flux_qof)
    flux_qof = torch.clamp(flux_qof, min=zeros, max=P)

    # Peff: Precipitation entering unsaturated zone
    flux_peff = P - flux_qof

    # --- 2. Unsaturated Zone Process (S1) ---

    # flux_ea: Evapotranspiration from S1
    # evap_3(p1=st, S=S1, Smax=suzmax, Ep=PET)
    flux_ea = evap_3(st, S1, suzmax, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1 - nearzero)
    flux_ea = F.relu(flux_ea)

    # flux_qex: Saturation excess overflow from S1
    flux_qex = saturation_1(flux_peff, S1, suzmax, nearzero=nearzero)
    flux_qex = torch.clamp(flux_qex, min=zeros, max=flux_peff)

    # Interim update for leakage
    S1_tmp = S1 + flux_peff - flux_ea - flux_qex
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_qv: Leakage to saturated zone
    # interflow_10(S, p1=kd, p2=threshold, p3=capacity)
    threshold_s1 = st * suzmax
    capacity_s1 = suzmax - threshold_s1
    flux_qv = interflow_10(
        S1_tmp, kd, threshold_s1, capacity_s1, nearzero=nearzero
    )
    flux_qv = torch.minimum(flux_qv, S1_tmp - nearzero)
    flux_qv = F.relu(flux_qv)

    # Update S1 final
    S1_new = S1_tmp - flux_qv
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 3. Saturated Zone Delay/Deficit Process (S2) ---

    # flux_qb: Baseflow speed (increases with decreasing deficit)
    # baseflow_4(p1=q0, p2=f, S=S2)
    flux_qb = baseflow_4(q0, f, S2, nearzero=nearzero)
    # Baseflow increases deficit, recharge decreases it
    S2_new = S2 + flux_qb - flux_qv
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = qof (saturation surface) + qex (unsaturated overflow) + qb (baseflow)
    # Ea = ea
    # Note: peff is an intermediate internal flux
    Qsim = flux_qof + flux_qex + flux_qb
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new
