import torch
import torch.nn.functional as F
from typing import Tuple
from .mopex1 import (
    baseflow_1,
    evap_7,
    saturation_1,
)
from .mopex4 import interception_seasonal

def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero
    )

def gsi_1(T: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
    t_range = torch.clamp(tmax - tmin, min=0.1)
    return torch.clamp((T - tmin) / t_range, 0.0, 1.0)

def mopex5_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # Parameters
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    
    # Guards
    S1, S2, Sc1, Sc2, Sn = F.relu(S1), F.relu(S2), F.relu(Sc1), F.relu(Sc2), F.relu(Sn)
    
    # 0. Phenology (GSI) Module
    flux_gsi = gsi_1(T, tmin, tmax)
    PET_effective = PET * flux_gsi

    # 1. Interception
    flux_i_pot = interception_seasonal(P, doy, alpha, is_time, nearzero)
    flux_i = torch.minimum(flux_i_pot, PET_effective)
    P_through = P - flux_i
    pet_for_soil = F.relu(PET_effective - flux_i)

    # 2. Snow Module (Unified Softplus for tcrit)
    # 使用 softplus 替代 relu，保证在 tcrit 处的导数连续
    is_rain = torch.sigmoid(T - tcrit)
    flux_qn = torch.minimum(is_rain * F.softplus(T - tcrit) * ddf * delta_t, Sn)
    Ps = P_through * (1.0 - is_rain)
    Pr = P_through * is_rain
    Sn_new = Sn + Ps - flux_qn
    P_eff  = Pr + flux_qn

    # 3. Soil: overflow -> ET -> infiltration
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    # ET 优先，保障水量守恒
    flux_et1 = torch.minimum(evap_7(S1, Sb1, pet_for_soil, delta_t, nearzero), S1)
    S1 = S1 - flux_et1
    
    # 从扣除 ET 后的 S1 计算下渗
    flux_qw = S1 * (1.0 - torch.exp(-delta_t / torch.clamp(tw, min=nearzero)))
    S1_new  = S1 - flux_qw

    # 4. Subsurface
    S2 = S2 + flux_qw
    flux_q2f = F.relu(S2 - Sb2)
    S2 = S2 - flux_q2f
    
    flux_q2u = S2 * (1.0 - torch.exp(-delta_t / torch.clamp(tu, min=nearzero)))
    S2 = S2 - flux_q2u

    remaining_pet = F.relu(pet_for_soil - flux_et1)
    flux_et2 = torch.minimum(evap_7(S2, Se, remaining_pet, delta_t, nearzero), S2)
    S2_new = S2 - flux_et2

    # 5. Routing
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = Sc1 * (1.0 - torch.exp(-delta_t / torch.clamp(tc, min=nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    flux_qs = Sc2 * (1.0 - torch.exp(-delta_t / torch.clamp(tc, min=nearzero)))
    Sc2_new = Sc2 - flux_qs

    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total  = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new