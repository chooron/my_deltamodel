import torch
import torch.nn.functional as F
from typing import Tuple

MOPEX_PARAMS_BOUNDS = {
    "Sb1": [0.01, 50.0],
    "tw": [0.01, 5.0],
    "tu": [1.0, 2000.0],
    "Se": [1.0, 1000.0],
    "tc": [0.1, 30.0],
    "ddf": [0.0, 20.0],
    "tcrit": [-3.0, 3.0],
    "Sb2": [1.0, 1500.0],
    "alpha": [0.0, 1.0],
    "is_time": [0.0, 365.0],
    "tmin": [-10.0, 5.0],
    "tmax": [5.0, 30.0],
}

def mopex_step(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights (New Inputs) ---
    w_phen: torch.Tensor,  # Phenology control [0, 1]
    w_int: torch.Tensor,  # Interception area [0, 1]
    w_snow: torch.Tensor,  # Snow accumulation area [0, 1]
    w_sub: torch.Tensor,  # Subsurface fast-flow connectivity [0, 1]
    # --- Parameters ---
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
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # ============================================================
    # 0. Guards - 在开始就保护所有状态和参数（参考mopex5.py）
    # ============================================================
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    # Sb1 = torch.clamp(Sb1, min=nearzero)
    # tw = torch.clamp(tw, min=nearzero)
    # tu = torch.clamp(tu, min=nearzero)
    # Se = torch.clamp(Se, min=nearzero)
    # tc = torch.clamp(tc, min=nearzero)
    # ddf = torch.clamp(ddf, min=0.0)
    # Sb2 = torch.clamp(Sb2, min=nearzero)
    # alpha = torch.clamp(alpha, 0.0, 1.0)

    # ============================================================
    # 1. Phenology Module (Soft Switch) - inline GSI
    # ============================================================
    t_range = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / t_range, 0.0, 1.0)

    # w_phen = 0: Use raw PET (Physics control)
    # w_phen = 1: Use GSI-modified PET (Biology control)
    PET_bio = PET * flux_gsi
    PET_effective = w_phen * PET_bio + (1.0 - w_phen) * PET

    # ============================================================
    # 2. Interception Module (Flux Gating) - inline seasonal interception
    # ============================================================
    is_time_safe = torch.clamp(is_time, 0.0, 365.0)
    rad = 2.0 * 3.1415926535 * (doy - is_time_safe) / 365.0
    season_factor = 0.5 * (torch.cos(rad) + 1.0)

    flux_potential = alpha * P * season_factor
    flux_i_pot = torch.minimum(flux_potential, P)

    # Only w_int fraction of the area intercepts water
    flux_i = flux_i_pot * w_int
    P_through = P - flux_i

    # ============================================================
    # 3. Snow Module (Input Splitting / Bypass)
    # ============================================================
    is_rain = (T > tcrit).float()

    # Path A: Bypass (Direct to soil)
    P_bypass = P_through * is_rain + P_through * (1 - is_rain) * (1.0 - w_snow)

    # Path B: Storage (Enters Snowpack)
    P_to_snow = P_through * (1 - is_rain) * w_snow

    # Physics: Melt logic
    melt_pot = F.relu(T - tcrit) * ddf
    flux_qn = torch.minimum(melt_pot, Sn)

    # Update State
    Sn_new = torch.clamp(Sn + P_to_snow - flux_qn, min=0.0)

    # Recombine: Effective P entering soil
    P_eff = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # ============================================================
    S1 = S1 + P_eff

    # Surface Runoff (Saturation excess) - 参考saturation_1函数
    flux_q1f = F.relu(S1 - Sb1)
    S1 = S1 - flux_q1f

    # Percolation to S2 (recharge) - 参考recharge_3函数
    flux_qw_pot = S1 / (tw + 1e-6)
    flux_qw = torch.minimum(flux_qw_pot, S1)
    S1 = S1 - flux_qw

    # Evaporation - 参考evap_7函数
    ratio_s1 = S1 / (Sb1 + 1e-6)
    flux_et1_pot = PET_effective * ratio_s1
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1_new = torch.clamp(S1 - flux_et1, min=0.0)

    # ============================================================
    # 5. Subsurface Module (S2) (State Leakage)
    # ============================================================
    S2 = S2 + flux_qw

    # Calculate Potential Overflow
    flux_q2f_pot = F.relu(S2 - Sb2)

    # Apply Weight: Only w_sub fraction flows out rapidly
    flux_q2f = flux_q2f_pot * w_sub
    S2 = S2 - flux_q2f

    # Baseflow - 参考baseflow_1函数
    flux_q2u_pot = S2 / (tu + 1e-6)
    flux_q2u = torch.minimum(flux_q2u_pot, S2)
    S2 = S2 - flux_q2u

    # Evaporation from S2 - 参考evap_7函数
    ratio_s2 = S2 / (Se + 1e-6)
    flux_et2_pot = PET_effective * ratio_s2
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = torch.clamp(S2 - flux_et2, min=0.0)

    # ============================================================
    # 6. Routing (baseflow inlined)
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf_pot = Sc1 / (tc + 1e-6)
    flux_qf = torch.minimum(flux_qf_pot, Sc1)
    Sc1_new = torch.clamp(Sc1 - flux_qf, min=0.0)

    Sc2 = Sc2 + flux_q2u
    flux_qs_pot = Sc2 / (tc + 1e-6)
    flux_qs = torch.minimum(flux_qs_pot, Sc2)
    Sc2_new = torch.clamp(Sc2 - flux_qs, min=0.0)

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new


def mopex_step_static(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Parameters ---
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
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    MOPEX step without structural weights - all processes are fully active
    """
    # ============================================================
    # 0. Guards
    # ============================================================
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    # ============================================================
    # 1. Phenology Module - GSI control (fully active)
    # ============================================================
    t_range = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / t_range, 0.0, 1.0)
    PET_effective = PET * flux_gsi

    # ============================================================
    # 2. Interception Module (fully active)
    # ============================================================
    is_time_safe = torch.clamp(is_time, 0.0, 365.0)
    rad = 2.0 * 3.1415926535 * (doy - is_time_safe) / 365.0
    season_factor = 0.5 * (torch.cos(rad) + 1.0)

    flux_potential = alpha * P * season_factor
    flux_i = torch.minimum(flux_potential, P)
    P_through = P - flux_i

    # ============================================================
    # 3. Snow Module (fully active)
    # ============================================================
    is_rain = (T > tcrit).float()
    P_to_snow = P_through * (1 - is_rain)
    P_bypass = P_through * is_rain

    # Melt logic
    melt_pot = F.relu(T - tcrit) * ddf
    flux_qn = torch.minimum(melt_pot, Sn)

    # Update State
    Sn_new = torch.clamp(Sn + P_to_snow - flux_qn, min=0.0)

    # Effective P entering soil
    P_eff = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # ============================================================
    S1 = S1 + P_eff

    # Surface Runoff
    flux_q1f = F.relu(S1 - Sb1)
    S1 = S1 - flux_q1f

    # Percolation to S2
    flux_qw_pot = S1 / (tw + 1e-6)
    flux_qw = torch.minimum(flux_qw_pot, S1)
    S1 = S1 - flux_qw

    # Evaporation
    ratio_s1 = S1 / (Sb1 + 1e-6)
    flux_et1_pot = PET_effective * ratio_s1
    flux_et1 = torch.minimum(flux_et1_pot, S1)
    S1_new = torch.clamp(S1 - flux_et1, min=0.0)

    # ============================================================
    # 5. Subsurface Module (S2) - fully active fast flow
    # ============================================================
    S2 = S2 + flux_qw

    # Fast flow (fully active)
    flux_q2f = F.relu(S2 - Sb2)
    S2 = S2 - flux_q2f

    # Baseflow
    flux_q2u_pot = S2 / (tu + 1e-6)
    flux_q2u = torch.minimum(flux_q2u_pot, S2)
    S2 = S2 - flux_q2u

    # Evaporation from S2
    ratio_s2 = S2 / (Se + 1e-6)
    flux_et2_pot = PET_effective * ratio_s2
    flux_et2 = torch.minimum(flux_et2_pot, S2)
    S2_new = torch.clamp(S2 - flux_et2, min=0.0)

    # ============================================================
    # 6. Routing
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf_pot = Sc1 / (tc + 1e-6)
    flux_qf = torch.minimum(flux_qf_pot, Sc1)
    Sc1_new = torch.clamp(Sc1 - flux_qf, min=0.0)

    Sc2 = Sc2 + flux_q2u
    flux_qs_pot = Sc2 / (tc + 1e-6)
    flux_qs = torch.minimum(flux_qs_pot, Sc2)
    Sc2_new = torch.clamp(Sc2 - flux_qs, min=0.0)

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
