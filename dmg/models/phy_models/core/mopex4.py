import torch
import torch.nn.functional as F
from typing import Tuple

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

# ================================================================
# 1. Parameter Configuration
# 严格对应 MARRMoT MATLAB 原版参数语义
# ================================================================

MOPEX4_PARAMS_BOUNDS = {
    "tcrit":   [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":     [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max":   [1.0,  2000.0], # Maximum soil moisture storage [mm]      → Sb1
    "tw":      [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "alpha":   [0.0,  1.0],    # Intercepted fraction of rainfall [-]    → i_alpha
    "is_time": [1.0,  365.0],  # Timing of maximum interception [d]      → i_s
    "tu":      [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":      [0.05, 0.95],   # Root zone ET capacity as fraction of s3max [-]
    "s3max":   [1.0,  2000.0], # Root zone (subsurface) storage capacity → Sb2
    "tc":      [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX4_PARAMS_DESC = {
    "tcrit":   "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":     "Degree-day factor [mm/°C/d]",
    "s2max":   "Maximum soil moisture storage [mm]                       (= Sb1)",
    "tw":      "Groundwater leakage rate [d⁻¹], flux = tw * S_soil",
    "alpha":   "Mean interception fraction [-], seasonal cosine modulation",
    "is_time": "Day-of-year of maximum interception [d]                  (= i_s)",
    "tu":      "Slow flow routing rate [d⁻¹], flux = tu * S_sub",
    "se":      "Root zone ET capacity fraction [-], ET2 Smax = se * s3max",
    "s3max":   "Root zone (subsurface) storage capacity [mm]             (= Sb2)",
    "tc":      "Mean residence rate [d⁻¹], flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (Sn, S1, S2, Sc1, Sc2).
    对应 MATLAB: S1=snow, S2=soil, S3=subsurface, S4=fast route, S5=slow route
    Python 接口映射：
        Sn  ↔ MATLAB S1 (snow)
        S1  ↔ MATLAB S2 (soil),       参数 Sb1 = s2max
        S2  ↔ MATLAB S3 (subsurface), 参数 Sb2 = s3max
        Sc1 ↔ MATLAB S4 (fast route)
        Sc2 ↔ MATLAB S5 (slow route)
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sn
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S1 (soil)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S2 (subsurface)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc1
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc2
    )


# ================================================================
# 2. Interception Flux Function
# ================================================================

def interception_4(
    flux_pr: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmax: float = 365.25,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    MATLAB 原版：
        out = max(0, p1 + (1-p1)*cos(2π*(t*dt - p2)/tmax)) * In
        其中 p1=i_alpha, p2=i_s, t*dt≈doy, In=flux_pr

    物理含义：
        截留比例随季节余弦变化，峰值在 doy=is_time 处（LAI 最大时截留最多）。
        alpha=0 时全年无截留；alpha=1 时全年截留全部降雨。
        比例下限截断为 0（max(0,...)），防止出现负截留（即降雨增益）。

    梯度处理：
        MATLAB 的 max(0,...) 在比例为零时梯度截断。
        替换为 F.softplus(x, beta) 的缩放近似，保留平滑下限，
        但 beta 取较大值（50）使其在实践中等价于 relu 而梯度连续。
        alpha 和 is_time 均通过余弦函数全程可导。

    参数：
        flux_pr  - 到达冠层的降雨通量 [mm/d]
        doy      - 当前儒略日 [d]，shape 与 flux_pr 一致
        alpha    - 平均截留比例 [-]，∈ [0,1]
        is_time  - 截留峰值时刻 [d]，∈ [1, 365]
        tmax     - 季节周期长度 [d]，默认 365.25
    """
    # 季节性截留比例（余弦调制）
    # 当 doy = is_time 时，cos=1，截留比例 = alpha + (1-alpha)*1 = 1（最大）
    # 当 doy = is_time ± tmax/2 时，cos=-1，截留比例 = alpha - (1-alpha)（最小）
    rad          = 2.0 * torch.pi * (doy - is_time) / tmax
    interc_frac  = alpha + (1.0 - alpha) * torch.cos(rad)

    # ✅ 梯度处理：softplus(x * beta) / beta ≈ relu(x)，但在 x=0 处光滑可导
    # beta=50 时与 relu 的最大偏差 < 0.014，实践中可忽略
    # 等价简写（直接用大 beta 的 softplus）：
    interc_frac_pos = F.softplus(interc_frac * 50.0) / 50.0   # ≥ 0，光滑

    # 截留量同时受降雨量约束
    flux_i = torch.minimum(interc_frac_pos * flux_pr, flux_pr)
    return flux_i


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex4_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # Parameters — order matches MOPEX4_PARAMS_BOUNDS keys
    tcrit: torch.Tensor,     # tcrit
    ddf: torch.Tensor,       # ddf
    Sb1: torch.Tensor,       # s2max → 土壤水库容量
    tw: torch.Tensor,        # tw
    alpha: torch.Tensor,     # alpha → i_alpha
    is_time: torch.Tensor,   # is_time → i_s
    tu: torch.Tensor,        # tu
    Se: torch.Tensor,        # se
    Sb2: torch.Tensor,       # s3max → 地下水库容量
    tc: torch.Tensor,        # tc
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
    MOPEX-4 离散单步计算。

    MATLAB ODE 对应关系：
        dS1 = ps   - qn                                  (Sn：积雪)
        dS2 = pr   + qn - et1 - i - q1f - qw            (S1：土壤，含截留项)
        dS3 = qw   - et2 - q2f - q2u                    (S2：地下)
        dS4 = q1f  + q2f - qf                            (Sc1：快速流)
        dS5 = q2u  - qs                                  (Sc2：慢速流)

    与 MOPEX-3 的唯一结构差异：
        新增截留通量 flux_i = interception_4(i_alpha, i_s, doy, flux_pr)
        flux_i 从 S2（土壤）中扣除，代表冠层截留后直接蒸发的水量，
        不进入土壤参与后续产流计算。

    离散化策略（顺序显式步进）：
        各通量按顺序从当前状态计算并立即更新，天然保证状态非负。

    通量顺序（S1/土壤）：加入 pr+qn → 蒸发 et1 → 截留 i → 饱和径流 q1f → 下渗 qw
    通量顺序（S2/地下）：加入 qw   → 地下溢流 q2f → 基流 q2u → 蒸发 et2
    """

    # ── Guards ────────────────────────────────────────────────────
    Sn  = F.relu(Sn)
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = MATLAB S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # 守恒：flux_ps + flux_pr = P

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    # melt_1 内部保证 flux_qn ≤ Sn

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn               # flux_qn ≤ Sn，非负保证

    # ============================================================
    # Soil Bucket (S1 = MATLAB S2)
    # MATLAB: dS2 = pr + qn - et1 - i - q1f - qw
    # 顺序：加入有效降水 → 蒸发 → 截留 → 饱和径流 → 下渗
    # ============================================================

    S1 = S1 + flux_pr + flux_qn

    # Step 1：蒸发
    # MATLAB: flux_et1 = evap_7(S2, s2max, Ep, dt)
    flux_et1 = evap_7(S1, Sb1, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1

    # Step 2：截留（冠层截留，季节余弦调制）
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
    flux_qw = recharge_3(tw, S1)          # min(tw*S1, S1) ≤ S1
    S1_new  = S1 - flux_qw                # S1_new ≥ 0 保证

    # ============================================================
    # Subsurface Bucket (S2 = MATLAB S3)
    # MATLAB: dS3 = qw - et2 - q2f - q2u
    # 顺序：加入下渗 → 地下溢流 → 基流 → 蒸发
    # ============================================================

    S2 = S2 + flux_qw

    # Step 1：地下溢流
    # MATLAB: flux_q2f = saturation_1(flux_qw, S3, s3max)
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f

    # Step 2：基流
    # MATLAB: flux_q2u = baseflow_1(tu, S3) → tu * S3
    flux_q2u = baseflow_1(tu, S2)         # min(tu*S2, S2) ≤ S2
    S2 = S2 - flux_q2u

    # Step 3：蒸发
    # MATLAB: flux_et2 = evap_7(S3, se*s3max, Ep, dt)
    se_abs   = Se * Sb2                   # 有效蒸发容量 [mm]
    flux_et2 = evap_7(S2, se_abs, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new   = S2 - flux_et2              # S2_new ≥ 0 保证

    # ============================================================
    # Routing Buckets
    # MATLAB: dS4 = q1f + q2f - qf；dS5 = q2u - qs
    # ============================================================

    Sc1      = Sc1 + flux_q1f + flux_q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf              # Sc1_new ≥ 0 保证

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs              # Sc2_new ≥ 0 保证

    # ============================================================
    # Output
    # MATLAB: FluxGroups.Ea = [et1, et2]；FluxGroups.Q = [qf, qs]
    # 注：flux_i（截留蒸发）也是实际蒸散发的一部分，
    # 若需要与观测 ET 对比，应加入 ET_total
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new