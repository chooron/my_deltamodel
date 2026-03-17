import torch
import torch.nn.functional as F
from typing import Tuple

from .mopex1 import (
    evap_7,
    saturation_1,
    baseflow_1,
    recharge_3,
)

# ================================================================
# 1. Parameter Configuration
# 严格对应 MARRMoT MATLAB 原版参数语义
# ================================================================

MOPEX2_PARAMS_BOUNDS = {
    "tcrit": [-3.0, 3.0],     # Snowfall & snowmelt temperature threshold [°C]
    "ddf":   [0.0,  20.0],    # Degree-day factor for snowmelt [mm/°C/d]
    "s2max": [1.0,  2000.0],  # Maximum soil moisture storage [mm]
    "tw":    [0.0,  1.0],     # Groundwater leakage rate [d⁻¹]
    "tu":    [0.0,  1.0],     # Slow flow routing rate [d⁻¹]
    "se":    [1.0,  2000.0],  # Root zone storage capacity [mm]
    "tc":    [0.0,  1.0],     # Mean residence rate [d⁻¹]
}

MOPEX2_PARAMS_DESC = {
    "tcrit": "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":   "Degree-day factor [mm/°C/d], melt = ddf * max(T - tcrit, 0)",
    "s2max": "Maximum soil moisture storage [mm]",
    "tw":    "Groundwater leakage rate [d⁻¹], flux = tw * S2",
    "tu":    "Slow flow routing rate [d⁻¹],   flux = tu * S3",
    "se":    "Root zone storage capacity [mm]",
    "tc":    "Mean residence rate [d⁻¹],       flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (Sn, S2, S3, Sc1, Sc2).
    对应 MATLAB: S1=snow, S2=soil, S3=subsurface, S4=fast route, S5=slow route
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
    )


# ================================================================
# 2. Snow Flux Functions
# ================================================================

def snowfall_1(
    P: torch.Tensor,
    T: torch.Tensor,
    tcrit: torch.Tensor,
    r: float = 0.01,
) -> torch.Tensor:
    """
    MATLAB 原版：out = P * smoothThreshold_temperature_logistic(T, tcrit)
    即温度低于 tcrit 时降水以降雪形式存在，比例随温度平滑过渡。

    smoothThreshold_temperature_logistic(T, tcrit) ≈ 1 when T << tcrit（全部为雪）
                                                    ≈ 0 when T >> tcrit（全部为雨）

    梯度处理：sigmoid((tcrit - T) / scale) 替代硬阈值，tcrit 全程可导。
    注意：snowfall_1 + rainfall_1 = P，两者严格互补，降水守恒。
    """
    scale = torch.abs(tcrit) * r + r   # 参考 MARRMoT 默认 r=0.01
    snow_frac = torch.sigmoid((tcrit - T) / (scale + 1e-6))
    return P * snow_frac


def rainfall_1(
    P: torch.Tensor,
    T: torch.Tensor,
    tcrit: torch.Tensor,
    r: float = 0.01,
) -> torch.Tensor:
    """
    MATLAB 原版：out = P * (1 - smoothThreshold_temperature_logistic(T, tcrit))
    即温度高于 tcrit 时降水以降雨形式存在。

    梯度处理：sigmoid((T - tcrit) / scale) 替代硬阈值，与 snowfall_1 互补。
    """
    scale = torch.abs(tcrit) * r + r
    rain_frac = torch.sigmoid((T - tcrit) / (scale + 1e-6))
    return P * rain_frac


def melt_1(
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    T: torch.Tensor,
    Sn: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    MATLAB 原版：out = max(min(ddf * (T - tcrit), S/dt), 0)
    即度日因子融雪，融雪量受积雪库容约束，且不能为负（温度低于阈值时无融雪）。

    梯度处理：
    - MATLAB 的 max(0, ...) 对应 F.relu，在 T=tcrit 处梯度不连续。
      替换为 sigmoid(T-tcrit) * softplus(T-tcrit)：
        * sigmoid 保证低温端趋近于 0（抑制虚假融雪）
        * softplus 保证高温端平滑线性增长
        * 乘积在 tcrit 处光滑，tcrit 和 ddf 均全程可导
    - MATLAB 的 min(..., S/dt) 用 torch.minimum 实现，天然可导。
    """
    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    melt_pot   = ddf * melt_drive * dt        # [mm]，对应 ddf*(T-tcrit)*dt
    return torch.minimum(melt_pot, Sn)        # 受积雪库容约束，等价于 min(..., S/dt)*dt


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    Sb2: torch.Tensor,     # 对应 s2max（MATLAB S2 的容量上限）
    tw: torch.Tensor,      # 率参数 [d⁻¹]
    tu: torch.Tensor,      # 率参数 [d⁻¹]
    Se: torch.Tensor,      # 对应 se
    tc: torch.Tensor,      # 率参数 [d⁻¹]
    # States
    Sn: torch.Tensor,      # 积雪水库（对应 MATLAB S1）
    S2: torch.Tensor,      # 土壤水库（对应 MATLAB S2）
    S3: torch.Tensor,      # 地下水库（对应 MATLAB S3）
    Sc1: torch.Tensor,     # 快速流路由水库（对应 MATLAB S4）
    Sc2: torch.Tensor,     # 慢速流路由水库（对应 MATLAB S5）
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-2 离散单步计算。

    MATLAB ODE 对应关系：
        dS1 = ps   - qn                          (Sn：积雪)
        dS2 = pr + qn - et1 - q1f - qw          (S2：土壤)
        dS3 = qw   - et2 - q2u                  (S3：地下)
        dS4 = q1f  - qf                          (Sc1：快速流)
        dS5 = q2u  - qs                          (Sc2：慢速流)

    离散化策略（顺序显式步进）：
        每个通量从当前状态计算后立即更新，下一通量基于更新后的状态，
        天然保证所有状态非负，无需 clamp 截断，梯度链完整。

    通量顺序（S2）：加入 pr+qn → et1 → q1f → qw
    通量顺序（S3）：加入 qw   → q2u → et2
    """

    # ── Guards：消除前一步数值误差的微小负值 ─────────────────────
    Sn  = F.relu(Sn)
    S2  = F.relu(S2)
    S3  = F.relu(S3)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    # 降水分割（雨雪互补，守恒：ps + pr = P）
    # MATLAB: flux_ps = snowfall_1(P, T, tcrit)
    #         flux_pr = rainfall_1(P, T, tcrit)
    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)

    # 融雪（受积雪库容约束）
    # MATLAB: flux_qn = melt_1(ddf, tcrit, T, S1, delta_t)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)

    # 顺序更新：加雪 → 融雪
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn              # flux_qn ≤ Sn，非负保证

    # ============================================================
    # Soil Bucket (S2)
    # MATLAB: dS2 = pr + qn - et1 - q1f - qw
    # 顺序：加入有效降水 → 蒸发 → 饱和径流 → 下渗
    # ============================================================

    # 加入有效降水（雨 + 融雪）
    S2 = S2 + flux_pr + flux_qn

    # Step 1：蒸发
    # MATLAB: flux_et1 = evap_7(S2, s2max, Ep, dt)
    flux_et1 = evap_7(S2, Sb2, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S2)   # 顺序步进安全截断
    S2 = S2 - flux_et1

    # Step 2：饱和径流（比例型，sigmoid 平滑）
    # MATLAB: flux_q1f = saturation_1(flux_pr + flux_qn, S2, s2max)
    # 注意：saturation_1 的入流参数在 MATLAB 中是 flux_pr+flux_qn，
    # 但因为已顺序更新 S2，此处用当前 S2 的蓄满比例乘以原始入流近似，
    # 物理含义等价：S2 越接近 Sb2，本次有效降水中成为径流的比例越大。
    flux_q1f = saturation_1(flux_pr + flux_qn, S2, Sb2, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S2)   # 顺序步进安全截断
    S2 = S2 - flux_q1f

    # Step 3：下渗
    # MATLAB: flux_qw = recharge_3(tw, S2)  →  tw * S2
    flux_qw = recharge_3(tw, S2)             # min(tw*S2, S2) ≤ S2
    S2_new  = S2 - flux_qw                   # S2_new ≥ 0 保证

    # ============================================================
    # Subsurface Bucket (S3)
    # MATLAB: dS3 = qw - et2 - q2u
    # 顺序：加入下渗 → 基流 → 蒸发
    # ============================================================

    S3 = S3 + flux_qw

    # Step 1：基流
    # MATLAB: flux_q2u = baseflow_1(tu, S3)  →  tu * S3
    flux_q2u = baseflow_1(tu, S3)            # min(tu*S3, S3) ≤ S3
    S3 = S3 - flux_q2u

    # Step 2：蒸发
    # MATLAB: flux_et2 = evap_7(S3, se, Ep, dt)
    flux_et2 = evap_7(S3, Se, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S3)   # 顺序步进安全截断
    S3_new   = S3 - flux_et2                 # S3_new ≥ 0 保证

    # ============================================================
    # Routing Buckets (Sc1 = S4, Sc2 = S5)
    # MATLAB: flux_qf = baseflow_1(tc, S4)；flux_qs = baseflow_1(tc, S5)
    # ============================================================

    Sc1      = Sc1 + flux_q1f
    flux_qf  = baseflow_1(tc, Sc1)           # min(tc*Sc1, Sc1) ≤ Sc1
    Sc1_new  = Sc1 - flux_qf                 # Sc1_new ≥ 0 保证

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)           # min(tc*Sc2, Sc2) ≤ Sc2
    Sc2_new  = Sc2 - flux_qs                 # Sc2_new ≥ 0 保证

    # ============================================================
    # Output
    # MATLAB: FluxGroups.Ea = [et1, et2]；FluxGroups.Q = [qf, qs]
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, Sn_new, S2_new, S3_new, Sc1_new, Sc2_new