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
# ================================================================

MOPEX3_PARAMS_BOUNDS = {
    "tcrit": [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":   [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max": [1.0,  2000.0], # Maximum soil moisture storage [mm]
    "tw":    [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "tu":    [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":    [0.05, 0.95],   # Root zone ET capacity as fraction of s3max [-]
    "s3max": [1.0,  2000.0], # Root zone (subsurface) storage capacity [mm]
    "tc":    [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX3_PARAMS_DESC = {
    "tcrit": "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":   "Degree-day factor [mm/°C/d]",
    "s2max": "Maximum soil moisture storage [mm]",
    "tw":    "Groundwater leakage rate [d⁻¹], flux = tw * S2",
    "tu":    "Slow flow routing rate [d⁻¹],   flux = tu * S3",
    "se":    "Root zone ET capacity fraction [-], ET2 uses se * s3max as Smax",
    "s3max": "Root zone (subsurface) storage capacity [mm]",
    "tc":    "Mean residence rate [d⁻¹],       flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (Sn, S2, S3, Sc1, Sc2).
    对应 MATLAB: S1=snow, S2=soil, S3=subsurface, S4=fast route, S5=slow route
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sn  (S1)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S2
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S3
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc1 (S4)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc2 (S5)
    )


# ================================================================
# 2. Main Model Step Function
# ================================================================

def mopex3_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    Sb2: torch.Tensor,     # 对应 s2max
    tw: torch.Tensor,      # 率参数 [d⁻¹]
    tu: torch.Tensor,      # 率参数 [d⁻¹]
    se: torch.Tensor,      # 分数参数 [-]，ET2 的有效容量 = se * Sb3
    Sb3: torch.Tensor,     # 对应 s3max
    tc: torch.Tensor,      # 率参数 [d⁻¹]
    # States
    Sn: torch.Tensor,      # 积雪水库（MATLAB S1）
    S2: torch.Tensor,      # 土壤水库（MATLAB S2）
    S3: torch.Tensor,      # 地下水库（MATLAB S3）
    Sc1: torch.Tensor,     # 快速流路由水库（MATLAB S4）
    Sc2: torch.Tensor,     # 慢速流路由水库（MATLAB S5）
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOPEX-3 离散单步计算。

    MATLAB ODE 对应关系：
        dS1 = ps  - qn                               (Sn：积雪)
        dS2 = pr + qn - et1 - q1f - qw              (S2：土壤)
        dS3 = qw  - et2 - q2f - q2u                 (S3：地下)
        dS4 = q1f + q2f - qf                         (Sc1：快速流，同时接收地表和地下溢流)
        dS5 = q2u - qs                               (Sc2：慢速流)

    与 MOPEX-2 的结构差异：
    1. 新增 s3max（Sb3）：地下水库有独立容量上限
    2. se 变为分数参数：ET2 的有效容量 = se * s3max，而非直接使用 se [mm]
    3. 新增 q2f = saturation_1(qw, S3, s3max)：下渗超出 S3 容量时产生地下溢流
    4. Sc1（快速路由）同时接收 q1f（地表溢流）和 q2f（地下溢流）
    """

    # ── Guards ────────────────────────────────────────────────────
    Sn  = F.relu(Sn)
    S2  = F.relu(S2)
    S3  = F.relu(S3)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # 守恒验证：flux_ps + flux_pr = P（sigmoid 互补）

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    # melt_1 内部保证 flux_qn ≤ Sn

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn               # flux_qn ≤ Sn，非负保证

    # ============================================================
    # Soil Bucket (S2)
    # MATLAB: dS2 = pr + qn - et1 - q1f - qw
    # 顺序：加入有效降水 → 蒸发 → 饱和径流 → 下渗
    # ============================================================

    S2 = S2 + flux_pr + flux_qn

    # Step 1：蒸发（MATLAB: evap_7(S2, s2max, Ep, dt)）
    flux_et1 = evap_7(S2, Sb2, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S2)
    S2 = S2 - flux_et1

    # Step 2：饱和径流（MATLAB: saturation_1(flux_pr+flux_qn, S2, s2max)）
    flux_q1f = saturation_1(flux_pr + flux_qn, S2, Sb2, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S2)
    S2 = S2 - flux_q1f

    # Step 3：下渗（MATLAB: recharge_3(tw, S2) → tw * S2）
    flux_qw = recharge_3(tw, S2)         # min(tw*S2, S2) ≤ S2
    S2_new  = S2 - flux_qw               # S2_new ≥ 0 保证

    # ============================================================
    # Subsurface Bucket (S3)
    # MATLAB: dS3 = qw - et2 - q2f - q2u
    # 顺序：加入下渗 → 地下溢流 → 基流 → 蒸发
    #
    # 与 MOPEX-2 的关键差异：
    #   新增 q2f = saturation_1(qw, S3, s3max)
    #   即当下渗水量使 S3 超过容量 Sb3 时，超出部分作为快速地下径流
    # ============================================================

    S3 = S3 + flux_qw

    # Step 1：地下溢流（MATLAB: saturation_1(flux_qw, S3, s3max)）
    # 物理含义：S3 蓄满时，新进入的下渗水 flux_qw 中超出容量的部分快速产流
    flux_q2f = saturation_1(flux_qw, S3, Sb3, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S3)
    S3 = S3 - flux_q2f

    # Step 2：基流（MATLAB: baseflow_1(tu, S3) → tu * S3）
    flux_q2u = baseflow_1(tu, S3)        # min(tu*S3, S3) ≤ S3
    S3 = S3 - flux_q2u

    # Step 3：蒸发
    # MATLAB: evap_7(S3, se*s3max, Ep, dt)
    # se 为分数参数，有效蒸发容量 = se * Sb3
    se_abs   = se * Sb3                  # [mm]，ET2 的有效 Smax
    flux_et2 = evap_7(S3, se_abs, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S3)
    S3_new   = S3 - flux_et2             # S3_new ≥ 0 保证

    # ============================================================
    # Routing Buckets
    # MATLAB: dS4 = q1f + q2f - qf（快速路由同时接收地表和地下溢流）
    #         dS5 = q2u - qs
    # ============================================================

    Sc1      = Sc1 + flux_q1f + flux_q2f  # 与 MOPEX-2 的关键差异：增加 q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf              # Sc1_new ≥ 0 保证

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs              # Sc2_new ≥ 0 保证

    # ============================================================
    # Output
    # MATLAB: FluxGroups.Ea = [et1, et2]；FluxGroups.Q = [qf, qs]
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, Sn_new, S2_new, S3_new, Sc1_new, Sc2_new