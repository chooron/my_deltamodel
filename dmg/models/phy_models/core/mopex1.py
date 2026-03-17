import torch
import torch.nn.functional as F
from typing import Tuple

# ================================================================
# 1. Parameter Configuration
# Adapted for large-sample hydrology (559 catchments)
# ================================================================

MOPEX1_PARAMS_BOUNDS = {
    # Surface bucket capacity [mm] (Ye et al. 2012: max ~1.0mm)
    # Expanded for large samples, but kept small to maintain saturation excess mechanism
    "s1max": [0.01, 50.0],

    # Infiltration time constant [days] (Ye et al. 2012: mean ~0.19 days)
    # Must be > 0.01 to avoid division instability
    "tw": [0.01, 5.0],

    # Subsurface flow recession constant [days] (Ye et al. 2012: max ~1300 days)
    "tu": [1.0, 2000.0],

    # Root zone storage capacity [mm] (Ye et al. 2012: max ~340mm)
    "se": [1.0, 1000.0],

    # Routing time constant [days]
    "tc": [0.1, 30.0],
}

MOPEX1_PARAMS_DESC = {
    "s1max": "Surface/Depression storage capacity (Threshold for Q1f) [mm]",
    "tw": "Infiltration time constant (Surface -> RootZone) [days]",
    "tu": "Subsurface flow recession constant [days]",
    "se": "Root zone storage capacity (Controls ET2) [mm]",
    "tc": "Streamflow routing time constant [days]",
}

def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (S1, S2, Sc1, Sc2)."""
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero
    )

# ================================================================
# 2. Flux Functions (Modular Physics Operators)
# ================================================================

def saturation_1(P: torch.Tensor, S: torch.Tensor, Smax: torch.Tensor) -> torch.Tensor:
    """Calculate saturation excess flow (Overflow)."""
    return F.relu((S + P) - Smax)

def recharge_3(k: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6) -> torch.Tensor:
    return S / (k + nearzero)

def baseflow_1(k: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6) -> torch.Tensor:
    return S / (k + nearzero)

def evap_7(S, Smax, Ep, dt=1.0, nearzero=1e-6):
    """
    ✅ 修复：加入 max=1.0 限制，确保蒸发比例不会超过 100%，
    防止 S > Smax 时算出比潜在蒸发量 (Ep) 还大的实际蒸发量。
    """
    evap_ratio = torch.clamp(S / (Smax + nearzero), max=1.0)
    return Ep * evap_ratio * dt

# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex1_step(
    P: torch.Tensor,
    T: torch.Tensor,        # 保留原接口，即未使用也放置于此
    PET: torch.Tensor,
    Sb1: torch.Tensor,      # 保留原接口命名，对应 s1max
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,       # 保留原接口命名，对应 se
    tc: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # ── Guards ───────────────────────────────────────────────────
    # 保证每步计算前水库不为负值
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ==========================================
    # 1. Bucket 1 (Surface Soil)
    # 顺序：溢流 → 蒸发 → 下渗（Ye et al. 2012）
    # ==========================================

    # Step 1：加雨，计算饱和溢流
    flux_q1f = F.relu((S1 + P) - Sb1)
    S1 = S1 + P - flux_q1f                          # 更新后 S1 ≤ Sb1

    # Step 2：蒸发（优先于下渗）
    # evap_7 已内含 clamp(max=1)，保障 flux_et1_pot <= PET
    flux_et1_pot = evap_7(S1, Sb1, PET, delta_t, nearzero)
    flux_et1     = torch.minimum(flux_et1_pot, S1)
    S1           = S1 - flux_et1

    # Step 3：下渗到 S2 
    # ✅ 修复：替换为解析解(指数衰减)，避免前向欧拉造成的梯度截断，且天然保证不超抽水库
    flux_qw = S1 * (1.0 - torch.exp(-delta_t / (tw + nearzero)))
    S1_new  = S1 - flux_qw

    # ==========================================
    # 2. Bucket 2 (Subsurface)
    # ==========================================

    S2 = S2 + flux_qw

    # 基流
    # ✅ 修复：替换为解析解(指数衰减)，保证在全参数空间连续可导
    flux_q2u = S2 * (1.0 - torch.exp(-delta_t / (tu + nearzero)))
    S2       = S2 - flux_q2u

    # ✅ 修复(原有)：S2 只能消耗 ET1 剩余的 PET；
    # 且因为 evap_7 内部增加了限制，不用担心 S2 溢出 Se 时带来的能量不守恒
    remaining_pet = F.relu(PET - flux_et1)
    flux_et2_pot  = evap_7(S2, Se, remaining_pet, delta_t, nearzero)
    flux_et2      = torch.minimum(flux_et2_pot, S2)
    S2_new        = S2 - flux_et2

    # ==========================================
    # 3. Routing
    # ==========================================

    # 快速流（地表溢流）
    Sc1      = Sc1 + flux_q1f
    # ✅ 修复：替换为解析解(指数衰减)，天然保证 flux_qf <= Sc1，连续可导
    flux_qf  = Sc1 * (1.0 - torch.exp(-delta_t / (tc + nearzero)))
    Sc1_new  = Sc1 - flux_qf

    # 慢速流（基流）
    Sc2      = Sc2 + flux_q2u
    # ✅ 修复：替换为解析解(指数衰减)
    flux_qs  = Sc2 * (1.0 - torch.exp(-delta_t / (tc + nearzero)))
    Sc2_new  = Sc2 - flux_qs

    # ==========================================
    # 4. Output
    # ==========================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new