import torch
import torch.nn.functional as F
from typing import Tuple
from ..marrmot.evap import evap_11
from ..marrmot.saturation import saturation_4
from ..marrmot.percolation import percolation_3
from ..marrmot.recharge import recharge_2
from ..marrmot.baseflow import baseflow_3

# Parameter range dictionary (based on MARRMoT m_07_gr4j_4p_2s)
GR4J_PARAMS_BOUNDS = {
    "x1": [1.0, 2000.0],    # Max soil moisture storage [mm]
    "x2": [-20.0, 20.0],    # Water exchange coefficient [mm/d]
    "x3": [1.0, 300.0],     # Max routing store storage [mm]
    "x4": [0.5, 15.0],      # Flow delay [d]
}

# Parameter description dictionary
GR4J_PARAMS_DESC = {
    "x1": "Maximum soil moisture storage [mm]",
    "x2": "Water exchange coefficient [mm/d]",
    "x3": "Maximum routing store storage [mm]",
    "x4": "Flow delay [d]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for GR4J model.
    S1: Production store
    S2: Routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def gr4j_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching GR4J_PARAMS_BOUNDS keys
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GR4J model single-step calculation.
    
    Model references:
    Perrin, C., Michel, C., & Andrassian, V. (2003). Improvement of a 
    parsimonious model for streamflow simulation. Journal of Hydrology, 
    279(1-4), 275-289.
    """

    # 1. Net precipitation and evaporation
    # flux_pn: net precipitation
    # flux_en: net evaporation
    # flux_ef: evaporation satisfied directly by precipitation
    flux_pn = F.relu(P - PET)
    flux_en = F.relu(PET - P)
    flux_ef = P - flux_pn

    # 2. Production store (S1) process
    # flux_ps: part of net rain entering S1
    flux_ps = saturation_4(S1, x1, flux_pn, nearzero=nearzero)
    flux_ps = torch.clamp(flux_ps, min=0.0, max=flux_pn)
    
    # flux_es: evaporation from S1
    flux_es = evap_11(S1, x1, flux_en, nearzero=nearzero)
    flux_es = torch.minimum(flux_es, S1 - nearzero)
    flux_es = F.relu(flux_es)
    
    # Update S1 for percolation calculation
    S1_tmp = S1 + flux_ps - flux_es
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)
    
    # flux_perc: percolation from S1
    flux_perc = percolation_3(S1_tmp, x1, nearzero=nearzero)
    flux_perc = torch.minimum(flux_perc, S1_tmp - nearzero)
    flux_perc = F.relu(flux_perc)
    
    # Final S1 update
    S1_new = S1_tmp - flux_perc
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 3. Routing and Exchange (TODO: Unit hydrograph routing)
    # TODO: Unit hydrograph routing (route/uh_) not supported yet
    # Total effective rainfall to be routed
    pr = (flux_pn - flux_ps) + flux_perc
    
    # Split into two branches: 90% to routing store S2, 10% direct
    # In this implementation, we assume instantaneous routing if uh_ is missing
    flux_q9_in = 0.9 * pr
    flux_q1_direct = 0.1 * pr

    # 4. Routing store (S2) process
    # flux_fr: groundwater exchange
    # recharge_2(p1=3.5, S, Smax=x3, p2=x2)
    flux_fr = recharge_2(torch.tensor(3.5, device=P.device), S2, x3, x2, nearzero=nearzero)
    
    # flux_qr: outflow from routing store
    # baseflow_3(S, Smax=x3)
    flux_qr = baseflow_3(S2, x3, nearzero=nearzero)
    flux_qr = torch.minimum(flux_qr, S2 + flux_q9_in + flux_fr - nearzero)
    flux_qr = F.relu(flux_qr)
    
    # Update S2 store
    S2_new = S2 + flux_q9_in + flux_fr - flux_qr
    S2_new = torch.clamp(S2_new, min=nearzero)

    # 5. Output Aggregation
    # Direct branch also receives the same exchange flux (fq = fr)
    flux_qt = flux_qr + F.relu(flux_q1_direct + flux_fr)
    
    Qsim = flux_qt
    Ea = flux_ef + flux_es

    return Qsim, Ea, S1_new, S2_new