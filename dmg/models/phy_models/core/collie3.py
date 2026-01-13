import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7, evap_3
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_9
from ..flux.baseflow import baseflow_2
from ..flux.split import split_1

# Parameter range dictionary (matching MARRMoT m_11_collie3_6p_2s)
COLLIE_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],       # Smax, Maximum soil moisture storage [mm]
    "fc": [0.05, 0.95],          # fc, Field capacity as fraction of Smax [-]
    "a": [0.0, 1.0],             # a, Subsurface runoff coefficient [d-1]
    "m": [0.05, 0.95],           # M, Fraction forest cover [-]
    "b": [1.0, 5.0],             # b, Non-linearity coefficient [-]
    "lambda_par": [0.0, 1.0],    # lambda, Flow distribution parameter [-]
}

# Parameter physical descriptions
COLLIE_PARAMS_DESC = {
    "smax": "Maximum soil moisture storage [mm]",
    "fc": "Field capacity as fraction of Smax [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "m": "Fraction forest cover [-]",
    "b": "Non-linearity coefficient [-]",
    "lambda_par": "Flow distribution parameter [-]",
}

def create_initial_state(
    n_grid: int, 
    nmul: int, 
    device: torch.device, 
    nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for Collie3 model.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2

# todo collie3 精度偏低
def collie3_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching dict keys exactly
    smax: torch.Tensor,
    fc: torch.Tensor,
    a: torch.Tensor,
    m: torch.Tensor,
    b: torch.Tensor,
    lambda_par: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collie River v3 model single step calculation.
    """
    # 1) Saturation excess runoff from the top store
    flux_qse = saturation_1(P, S1, smax, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=torch.zeros_like(flux_qse), max=P)
    S1_tmp = torch.clamp(S1 + P - flux_qse, min=nearzero)

    # 2) Evapotranspiration split into bare soil and vegetation components
    pet_bare = (1.0 - m) * PET
    pet_veg = m * PET
    flux_eb = evap_7(S1_tmp, smax, pet_bare, nearzero=nearzero)
    flux_ev = evap_3(fc, S1_tmp, smax, pet_veg, nearzero=nearzero)

    flux_ea_total = flux_eb + flux_ev
    flux_ea_total = torch.minimum(flux_ea_total, S1_tmp - nearzero)
    flux_ea_total = torch.minimum(flux_ea_total, PET)
    flux_ea_total = F.relu(flux_ea_total)

    S1_tmp2 = torch.clamp(S1_tmp - flux_ea_total, min=nearzero)

    # 3) Interflow from the top store
    sfc_mm = fc * smax
    flux_qss = interflow_9(S1_tmp2, a, sfc_mm, b, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S1_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    S1_new = torch.clamp(S1_tmp2 - flux_qss, min=nearzero)

    # 4) Split interflow between groundwater and direct flow
    flux_qsss = split_1(lambda_par, flux_qss)
    flux_qss_direct = split_1(1.0 - lambda_par, flux_qss)

    # 5) Groundwater store update and baseflow
    S2_tmp = torch.clamp(S2 + flux_qsss, min=nearzero)
    inv_a = 1.0 / (a + nearzero)
    inv_b = 1.0 / (b + nearzero)
    flux_qsg = baseflow_2(S2_tmp, inv_a, inv_b, nearzero=nearzero)
    flux_qsg = torch.minimum(flux_qsg, S2_tmp - nearzero)
    flux_qsg = F.relu(flux_qsg)

    S2_new = torch.clamp(S2_tmp - flux_qsg, min=nearzero)

    # 6) Aggregate outputs
    Q = flux_qse + flux_qss_direct + flux_qsg
    Ea = flux_ea_total

    return Q, Ea, S1_new, S2_new

