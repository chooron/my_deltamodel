import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_1, evap_2
from ..flux.interception import interception_1
from ..flux.infiltration import infiltration_1
from ..flux.interflow import interflow_1
from ..flux.recharge import recharge_1
from ..flux.saturation import saturation_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_18_simhyd_7p_3s)
SIMHYD_PARAMS_BOUNDS = {
    "insc": [0.0, 5.0],  # Maximum interception capacity [mm]
    "coeff": [0.0, 600.0],  # Maximum infiltration loss parameter [mm]
    "sq": [0.0, 15.0],  # Infiltration loss exponent [-]
    "smsc": [1.0, 2000.0],  # Maximum soil moisture capacity [mm]
    "sub": [0.0, 1.0],  # Interflow proportionality constant [-]
    "crak": [0.0, 1.0],  # Recharge proportionality constant [-]
    "k": [0.0, 1.0],  # Slow flow time scale [d-1]
}

# Parameter description dictionary
SIMHYD_PARAMS_DESC = {
    "insc": "Maximum interception capacity [mm]",
    "coeff": "Maximum infiltration loss parameter [mm]",
    "sq": "Infiltration loss exponent [-]",
    "smsc": "Maximum soil moisture capacity [mm]",
    "sub": "Proportionality constant for interflow [-]",
    "crak": "Proportionality constant for recharge [-]",
    "k": "Slow flow time scale [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for SimHyd model.
    S1: Interception store
    S2: Soil moisture store
    S3: Groundwater store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def simhyd_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching SIMHYD_PARAMS_BOUNDS keys
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    SimHyd model single-step calculation.

    Model reference:
    Chiew, F. H. S., Peel, M. C., & Western, A. W. (2002). Application and
    testing of the simple rainfall-runoff model SIMHYD.
    """

    # --- 1. Interception Process (S1) ---
    # flux_EXC: Excess rainfall after interception
    flux_EXC = interception_1(P, S1, insc, nearzero=nearzero)
    zeros = torch.zeros_like(flux_EXC)
    flux_EXC = torch.clamp(flux_EXC, min=zeros, max=P)

    # State update for interception evaporation
    S1_tmp = S1 + P - flux_EXC
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # flux_Ei: Evaporation from interception store
    flux_Ei = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_Ei = torch.minimum(flux_Ei, S1_tmp - nearzero)
    flux_Ei = F.relu(flux_Ei)

    # Final S1 update
    S1_new = S1_tmp - flux_Ei
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 2. Soil Moisture Process (S2) ---
    # Step 2.1: Surface processes (Infiltration and Runoff)
    # flux_INF: Infiltration into the soil
    flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC, nearzero=nearzero)
    flux_INF = torch.minimum(flux_INF, flux_EXC)

    # flux_SRUN: Surface runoff (Saturation excess before infiltration)
    flux_SRUN = F.relu(flux_EXC - flux_INF)

    # Step 2.2: Internal soil moisture split
    # flux_INT: Interflow from infiltrated water
    flux_INT = interflow_1(sub, S2, smsc, flux_INF, nearzero=nearzero)
    flux_INT = torch.minimum(flux_INT, flux_INF)

    # flux_REC: Groundwater recharge
    flux_rem_inf = F.relu(flux_INF - flux_INT)
    flux_REC = recharge_1(crak, S2, smsc, flux_rem_inf, nearzero=nearzero)
    flux_REC = torch.minimum(flux_REC, flux_rem_inf)

    # flux_SMF: Soil moisture filling flux
    flux_SMF = F.relu(flux_rem_inf - flux_REC)

    # flux_GWF: Saturation excess from soil moisture store to groundwater
    flux_GWF = saturation_1(flux_SMF, S2, smsc, nearzero=nearzero)
    flux_GWF = torch.clamp(flux_GWF, min=zeros, max=flux_SMF)

    # Step 2.3: State update and Evapotranspiration
    S2_tmp = S2 + flux_SMF - flux_GWF
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Remaining PET after interception ET
    pet_rem = F.relu(PET - flux_Ei)

    # flux_Et: Transpiration from soil
    # MATLAB: evap_2(10, S2, smsc, Ep) - p1=10 is used as a constant
    p1_const = torch.tensor(10.0, device=P.device)
    flux_Et = evap_2(p1_const, S2_tmp, smsc, pet_rem, nearzero=nearzero)
    flux_Et = torch.minimum(flux_Et, S2_tmp - nearzero)
    flux_Et = torch.minimum(flux_Et, pet_rem)
    flux_Et = F.relu(flux_Et)

    # Final S2 update
    S2_new = S2_tmp - flux_Et
    S2_new = torch.clamp(S2_new, min=nearzero)

    # --- 3. Groundwater Process (S3) ---
    # Inflow to S3: groundwater recharge (REC) and saturation overflow (GWF)
    inflow_S3 = flux_REC + flux_GWF

    S3_tmp = S3 + inflow_S3
    S3_tmp = torch.clamp(S3_tmp, min=nearzero)

    # flux_BAS: Baseflow from groundwater
    flux_BAS = baseflow_1(k, S3_tmp, nearzero=nearzero)
    flux_BAS = torch.minimum(flux_BAS, S3_tmp - nearzero)
    flux_BAS = F.relu(flux_BAS)

    # Final S3 update
    S3_new = S3_tmp - flux_BAS
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 4. Output Aggregation ---
    # Qsim = Surface Runoff + Interflow + Baseflow
    # Ea = Interception ET + Soil Transpiration
    Qsim = flux_SRUN + flux_INT + flux_BAS
    Ea = flux_Ei + flux_Et

    return Qsim, Ea, S1_new, S2_new, S3_new
