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
    # 0. Guards
    # ============================================================
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn  = F.relu(Sn)

    # ============================================================
    # 1. Phenology Module (Soft Switch) - GSI with trange
    # tmax 在接口中保留，内部作为 trange 使用（tmax = tmin + trange → trange = tmax - tmin）
    # ============================================================
    trange = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)

    # w_phen = 0: Use raw PET; w_phen = 1: Use GSI-modified PET
    PET_bio = PET * flux_gsi
    PET_effective = w_phen * PET_bio + (1.0 - w_phen) * PET

    # ============================================================
    # 2. Interception Module (Flux Gating)
    # 截留受 PET_effective 约束，防止能量不守恒
    # ============================================================
    is_time_safe = torch.clamp(is_time, 0.0, 365.0)
    rad = 2.0 * torch.pi * (doy - is_time_safe) / 365.0
    season_factor = 0.5 * (torch.cos(rad) + 1.0)

    flux_potential = alpha * P * season_factor
    flux_i_pot = torch.minimum(flux_potential, torch.minimum(P, PET_effective))

    # Only w_int fraction of the area intercepts water
    flux_i = flux_i_pot * w_int
    P_through = P - flux_i

    # 截留消耗后的剩余 PET 交给土壤蒸发
    pet_for_soil = F.relu(PET_effective - flux_i)

    # ============================================================
    # 3. Snow Module (Soft Switch)
    # 用 sigmoid 替代硬阈值，打通 tcrit 的梯度回传
    # ============================================================
    is_rain = torch.sigmoid(T - tcrit)

    # Path A: Bypass (Direct to soil)
    P_bypass  = P_through * is_rain + P_through * (1.0 - is_rain) * (1.0 - w_snow)

    # Path B: Storage (Enters Snowpack)
    P_to_snow = P_through * (1.0 - is_rain) * w_snow

    # 融雪：sigmoid × softplus，消除冰点以下虚假融雪，保证 tcrit/ddf 全程可导
    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    flux_qn    = torch.minimum(melt_drive * ddf, Sn)

    Sn_new = Sn + P_to_snow - flux_qn
    P_eff  = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # 顺序：溢流 → 蒸发 → 下渗
    # ============================================================
    # Step 1: 溢流
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    # Step 2: 蒸发（使用截留后剩余 PET）
    ratio_s1 = torch.clamp(S1 / (Sb1 + nearzero), max=1.0)
    flux_et1 = torch.minimum(pet_for_soil * ratio_s1, S1)
    S1 = S1 - flux_et1

    # Step 3: 下渗（指数解析解）
    flux_qw = S1 * (1.0 - torch.exp(-1.0 / (tw + nearzero)))
    S1_new  = S1 - flux_qw

    # ============================================================
    # 5. Subsurface Module (S2) (State Leakage)
    # ============================================================
    S2 = S2 + flux_qw

    # 地下溢流：仅 w_sub 比例快速流出
    flux_q2f_pot = F.relu(S2 - Sb2)
    flux_q2f = flux_q2f_pot * w_sub
    S2 = S2 - flux_q2f

    # 基流（指数解析解）
    flux_q2u = S2 * (1.0 - torch.exp(-1.0 / (tu + nearzero)))
    S2 = S2 - flux_q2u

    # 蒸发（逐层扣减后的剩余 PET）
    remaining_pet = F.relu(pet_for_soil - flux_et1)
    ratio_s2  = torch.clamp(S2 / (Se + nearzero), max=1.0)
    flux_et2  = torch.minimum(remaining_pet * ratio_s2, S2)
    S2_new    = S2 - flux_et2

    # ============================================================
    # 6. Routing（指数解析解）
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = Sc1 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    flux_qs = Sc2 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total  = flux_qf + flux_qs

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
    MOPEX step without structural weights - all processes are fully active.
    """
    # ============================================================
    # 0. Guards
    # ============================================================
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn  = F.relu(Sn)

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
    # ✅ 修复：使用 torch.pi 替代字面量
    rad = 2.0 * torch.pi * (doy - is_time_safe) / 365.0
    season_factor = 0.5 * (torch.cos(rad) + 1.0)

    flux_potential = alpha * P * season_factor
    # ✅ 修复：截留蒸发同时受 P 和 PET_effective 约束，防止能量不守恒
    flux_i    = torch.minimum(flux_potential, torch.minimum(P, PET_effective))
    P_through = P - flux_i

    # ✅ 修复：截留消耗后的剩余 PET 交给土壤蒸发，保证各蒸发项逐层扣减
    pet_for_soil = F.relu(PET_effective - flux_i)

    # ============================================================
    # 3. Snow Module (fully active)
    # ============================================================
    # ✅ 修复：硬阈值替换为 Sigmoid，打通 tcrit 的梯度回传
    is_rain = torch.sigmoid(T - tcrit)
    P_to_snow = P_through * (1.0 - is_rain)
    P_bypass  = P_through * is_rain

    # ✅ 修复：融雪通量使用 sigmoid × softplus，消除冰点以下虚假融雪，
    #         同时保证 tcrit 和 ddf 两个参数均全程可导
    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    flux_qn    = torch.minimum(melt_drive * ddf, Sn)

    Sn_new = Sn + P_to_snow - flux_qn
    P_eff  = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # 顺序：溢流 → 蒸发 → 下渗（与 Ye et al. 2012 及其他模型一致）
    # ============================================================
    # Step 1: 加雨 + 溢流
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    # Step 2: 蒸发（优先于下渗）
    # ✅ 修复：ET1 使用截留后的剩余 PET（pet_for_soil），不再与截留蒸发竞争同一能量
    ratio_s1     = torch.clamp(S1 / (Sb1 + nearzero), max=1.0)
    flux_et1_pot = pet_for_soil * ratio_s1
    flux_et1     = torch.minimum(flux_et1_pot, S1)
    S1           = S1 - flux_et1

    # Step 3: 下渗
    # ✅ 修复：指数解析解替代欧拉近似，天然保证 flux_qw <= S1，梯度平滑
    flux_qw = S1 * (1.0 - torch.exp(-1.0 / (tw + nearzero)))
    S1_new  = S1 - flux_qw

    # ============================================================
    # 5. Subsurface Module (S2) - fully active fast flow
    # ============================================================
    S2 = S2 + flux_qw

    flux_q2f = F.relu(S2 - Sb2)
    S2 = S2 - flux_q2f

    # ✅ 修复：指数解析解替代欧拉近似
    flux_q2u = S2 * (1.0 - torch.exp(-1.0 / (tu + nearzero)))
    S2 = S2 - flux_q2u

    # ✅ 修复：ET2 使用逐层扣减后的 remaining_pet，保证三项蒸发之和不超过 PET_effective
    remaining_pet = F.relu(pet_for_soil - flux_et1)
    ratio_s2      = torch.clamp(S2 / (Se + nearzero), max=1.0)
    flux_et2_pot  = remaining_pet * ratio_s2
    flux_et2      = torch.minimum(flux_et2_pot, S2)
    S2_new        = S2 - flux_et2

    # ============================================================
    # 6. Routing
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    # ✅ 修复：指数解析解替代欧拉近似
    flux_qf = Sc1 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    # ✅ 修复：指数解析解替代欧拉近似
    flux_qs = Sc2 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total  = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new