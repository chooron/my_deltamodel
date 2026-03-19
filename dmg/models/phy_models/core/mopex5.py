import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .mopex1 import (
    evap_7,
    saturation_1,
    baseflow_1,
    recharge_3,
)
from .mopex2 import (
    snowfall_1,
    rainfall_1,
    melt_1,
)
from .mopex4 import interception_4

# ================================================================
# 1. Parameter Configuration
# 在 MOPEX-4 基础上新增 tmin、trange 两个物候参数
# ================================================================

MOPEX5_PARAMS_BOUNDS = {
    "tcrit":   [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":     [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max":   [1.0,  2000.0], # Maximum soil moisture storage [mm]
    "tw":      [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "alpha":   [0.0,  1.0],    # Intercepted fraction of rainfall [-]
    "is_time": [1.0,  365.0],  # Timing of maximum interception [d]
    "tmin":    [-10.0, 0.0],   # GSI minimum temperature (ET stops below) [°C]
    "trange":  [1.0,  20.0],   # GSI temperature range (ET ramps over) [°C]
    "tu":      [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":      [0.05, 0.95],   # Root zone ET capacity fraction [-]
    "s3max":   [1.0,  2000.0], # Root zone (subsurface) storage capacity [mm]
    "tc":      [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX5_PARAMS_DESC = {
    "tcrit":   "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":     "Degree-day factor [mm/°C/d]",
    "s2max":   "Maximum soil moisture storage [mm]",
    "tw":      "Groundwater leakage rate [d⁻¹]",
    "alpha":   "Mean interception fraction [-]",
    "is_time": "Day-of-year of maximum interception [d]",
    "tmin":    "GSI lower temperature threshold; ET=0 when T <= tmin [°C]",
    "trange":  "GSI temperature range; ET=Ep when T >= tmin+trange [°C]",
    "tu":      "Slow flow routing rate [d⁻¹]",
    "se":      "Root zone ET capacity fraction [-], ET2 Smax = se * s3max",
    "s3max":   "Root zone (subsurface) storage capacity [mm]",
    "tc":      "Mean residence rate [d⁻¹]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (Sn, S1, S2, Sc1, Sc2).
    与 MOPEX-4 完全相同的状态变量布局：
        Sn  ↔ MATLAB S1 (snow)
        S1  ↔ MATLAB S2 (soil),       容量参数 Sb1 = s2max
        S2  ↔ MATLAB S3 (subsurface), 容量参数 Sb2 = s3max
        Sc1 ↔ MATLAB S4 (fast route)
        Sc2 ↔ MATLAB S5 (slow route)
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
    )


# ================================================================
# 2. Phenology Flux Function
# ================================================================

def phenology_1(
    T: torch.Tensor,
    tmin: torch.Tensor,
    trange: torch.Tensor,
    PET: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    MATLAB 原版：out = min(1, max(0, (T-p1)/(p2-p1))) * Ep
    其中 p1=tmin, p2=tmin+trange，即在 [tmin, tmin+trange] 上线性爬坡的 GSI。

    物理含义：
        T <= tmin          → GSI = 0，植被休眠，有效 PET = 0
        T >= tmin+trange   → GSI = 1，植被全活跃，有效 PET = PET
        tmin < T < tmax    → GSI 线性插值，有效 PET 按比例缩减

    梯度处理：
        MATLAB 的 min/max 双重截断等价于 torch.clamp(gsi, 0, 1)。
        clamp 在线性区间内对 tmin、trange 的梯度完整传递：
            ∂gsi/∂tmin   = -1/trange  （T 在线性区时）
            ∂gsi/∂trange = -(T-tmin)/trange²
        在饱和区（GSI=0 或 GSI=1）梯度为零，符合物理预期
        （极端温度下调整 tmin/trange 不改变 GSI 输出）。
        对于需要完全光滑梯度的场景，可改用双 sigmoid 近似，
        但对大样本优化实践而言 clamp 已经足够。
    """
    gsi = torch.clamp((T - tmin) / (trange + nearzero), 0.0, 1.0)
    return gsi * PET


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex5_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # Parameters — order matches MOPEX5_PARAMS_BOUNDS keys
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    Sb1: torch.Tensor,       # s2max
    tw: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    trange: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,        # se
    Sb2: torch.Tensor,       # s3max
    tc: torch.Tensor,
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-5 离散单步计算。

    MATLAB ODE 对应关系：
        dS1 = ps   - qn                                  (Sn：积雪)
        dS2 = pr   + qn - et1 - i - q1f - qw            (S1：土壤)
        dS3 = qw   - et2 - q2f - q2u                    (S2：地下)
        dS4 = q1f  + q2f - qf                            (Sc1：快速流)
        dS5 = q2u  - qs                                  (Sc2：慢速流)

    与 MOPEX-4 的唯一结构差异：
        新增 flux_epc = phenology_1(T, tmin, tmin+trange, PET)
        ET1 和 ET2 均使用 flux_epc（物候修正后的有效 PET）而非原始 PET，
        其余所有通量、状态更新顺序与 MOPEX-4 完全一致。
    """

    # ── Guards ────────────────────────────────────────────────────
    Sn  = F.relu(Sn)
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Phenology Module
    # MATLAB: flux_epc = phenology_1(T, tmin, tmin+trange, Ep)
    # 物候修正后的有效 PET，仅影响 ET1 和 ET2，
    # 截留蒸发（flux_i）不受物候影响（冠层拦截与植被活性无关）
    # ============================================================
    PET_epc = phenology_1(T, tmin, trange, PET, nearzero)

    # ============================================================
    # Snow Bucket (Sn = MATLAB S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # 守恒：flux_ps + flux_pr = P

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn               # flux_qn ≤ Sn，非负保证

    # ============================================================
    # Soil Bucket (S1 = MATLAB S2)
    # MATLAB: dS2 = pr + qn - et1 - i - q1f - qw
    # 顺序：加入有效降水 → 蒸发(PET_epc) → 截留 → 饱和径流 → 下渗
    # ============================================================

    S1 = S1 + flux_pr + flux_qn

    # Step 1：蒸发（使用物候修正后的 PET_epc）
    # MATLAB: flux_et1 = evap_7(S2, s2max, flux_epc, dt)
    flux_et1 = evap_7(S1, Sb1, PET_epc, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1

    # Step 2：截留（使用原始 PET，与 MOPEX-4 一致）
    # MATLAB: flux_i = interception_4(i_alpha, i_s, t, tmax, flux_pr, dt)
    flux_i = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_i = torch.minimum(flux_i, S1)
    S1 = S1 - flux_i

    # Step 3：饱和径流
    # MATLAB: flux_q1f = saturation_1(flux_pr+flux_qn, S2, s2max)
    flux_q1f = saturation_1(flux_pr + flux_qn, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S1)
    S1 = S1 - flux_q1f

    # Step 4：下渗
    # MATLAB: flux_qw = recharge_3(tw, S2) → tw * S2
    flux_qw = recharge_3(tw, S1)
    S1_new  = S1 - flux_qw               # S1_new ≥ 0 保证

    # ============================================================
    # Subsurface Bucket (S2 = MATLAB S3)
    # MATLAB: dS3 = qw - et2 - q2f - q2u
    # 顺序：加入下渗 → 地下溢流 → 基流 → 蒸发(PET_epc)
    # ============================================================

    S2 = S2 + flux_qw

    # Step 1：地下溢流
    # MATLAB: flux_q2f = saturation_1(flux_qw, S3, s3max)
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f

    # Step 2：基流
    # MATLAB: flux_q2u = baseflow_1(tu, S3) → tu * S3
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u

    # Step 3：蒸发（使用物候修正后的 PET_epc）
    # MATLAB: flux_et2 = evap_7(S3, se*s3max, flux_epc, dt)
    se_abs   = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, PET_epc, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new   = S2 - flux_et2             # S2_new ≥ 0 保证

    # ============================================================
    # Routing Buckets
    # MATLAB: dS4 = q1f + q2f - qf；dS5 = q2u - qs
    # ============================================================

    Sc1      = Sc1 + flux_q1f + flux_q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs

    # ============================================================
    # Output
    # MATLAB: FluxGroups.Ea = [et1, et2]；FluxGroups.Q = [qf, qs]
    # ET_total 含截留蒸发，与 CAMELS ET 口径一致
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new