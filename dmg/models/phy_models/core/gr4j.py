import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_11
from ..flux.saturation import saturation_4
from ..flux.recharge import recharge_2

# Parameter range dictionary (based on MARRMoT m_07_gr4j_4p_2s)
GR4J_PARAMS_BOUNDS = {
    "x1": [1.0, 2000.0],  # Max soil moisture storage [mm]
    "x2": [-20.0, 20.0],  # Water exchange coefficient [mm/d]
    "x3": [1.0, 300.0],  # Max routing store storage [mm]
    "x4": [0.5, 15.0],  # Flow delay [d]
}

# Parameter description dictionary
GR4J_PARAMS_DESC = {
    "x1": "Maximum soil moisture storage [mm]",
    "x2": "Water exchange coefficient [mm/d]",
    "x3": "Maximum routing store storage [mm]",
    "x4": "Flow delay [d]",
}


def percolation_3(
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Safe nonlinear percolation consistent with specialv2."""
    denom = Smax + nearzero
    ratio = S / denom
    ratio_safe = torch.clamp(ratio, max=1.5)
    const_term = (4.0 / 9.0) ** 4.0 / 4.0
    return const_term * S * ratio_safe.pow(4.0)


def baseflow_3(
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Safe nonlinear baseflow consistent with specialv2."""
    denom = Smax + nearzero
    ratio = S / denom
    ratio_safe = torch.clamp(ratio, max=1.5)
    return 0.25 * S * ratio_safe.pow(4.0)


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
    GR4J model single-step with Fixed Water Balance.
    """

    # 1. Net precipitation and evaporation
    flux_pn = F.relu(P - PET)
    flux_en = F.relu(PET - P)
    flux_ef = P - flux_pn  # Evaporation directly from rainfall

    # ==========================================
    # 2. Production store (S1) process
    # [Sequential Update]
    # ==========================================
    
    # [Step 1] Rainfall enters S1
    flux_ps = saturation_4(S1, x1, flux_pn, nearzero=nearzero)
    flux_ps = torch.clamp(flux_ps, min=torch.zeros_like(flux_pn), max=flux_pn)
    S1 = S1 + flux_ps

    # [Step 2] Evaporation from S1
    flux_es = evap_11(S1, x1, flux_en, nearzero=nearzero)
    flux_es = torch.minimum(flux_es, S1) # Safety
    S1 = S1 - flux_es

    # [Step 3] Percolation
    flux_perc = percolation_3(S1, x1, nearzero=nearzero)
    flux_perc = torch.minimum(flux_perc, S1) # Safety
    S1_new = S1 - flux_perc
    S1_new = torch.clamp(S1_new, min=nearzero)

    # ==========================================
    # 3. Routing Split
    # ==========================================
    pr = (flux_pn - flux_ps) + flux_perc
    flux_q9_in = 0.9 * pr
    flux_q1_direct = 0.1 * pr

    # ==========================================
    # 4. Routing store (S2) process
    # [Sequential Update with Actual Flux Calculation]
    # ==========================================

    # [Step 1] Add Inflow
    S2 = S2 + flux_q9_in

    # [Step 2] Exchange (flux_fr)
    # 计算潜在交换量
    flux_fr_potential = recharge_2(
        torch.tensor(3.5, device=P.device), S2, x3, x2, nearzero=nearzero
    )
    
    # --- 核心修正开始 ---
    # 计算实际交换量 (Actual Exchange)
    # 如果是正数(Gain)，无限制 (除非你想限制不能超过某个物理上限，通常GR4J不限制Gain)
    # 如果是负数(Loss)，不能超过当前 S2 的水量
    
    # 逻辑：S2_temp = S2 + potential
    # 如果 S2_temp < 0，说明亏空大于库存
    # 实际亏空 = -S2 (把库存清零)
    
    # 这种写法利用 clamp 自动处理：
    # S2_after_exchange = max(S2 + F, 0)
    # Actual_F = S2_after_exchange - S2_before_exchange
    
    S2_before_exchange = S2
    S2_temp = S2 + flux_fr_potential
    S2_after_exchange = torch.clamp(S2_temp, min=nearzero)
    
    # 得到这一步真正发生的物理交换量（包含了 Clamp 的截断效果）
    flux_fr_actual = S2_after_exchange - S2_before_exchange
    
    # 更新状态
    S2 = S2_after_exchange
    # --- 核心修正结束 ---

    # [Step 3] Outflow (flux_qr)
    flux_qr = baseflow_3(S2, x3, nearzero=nearzero)
    flux_qr = torch.minimum(flux_qr, S2) # Safety
    
    # Update S2 (Subtract Outflow)
    S2_new = S2 - flux_qr
    S2_new = torch.clamp(S2_new, min=nearzero)

    # ==========================================
    # 5. Output Aggregation
    # ==========================================
    
    # Direct branch receives the POTENTIAL exchange flux? 
    # GR4J 标准：Qd = max(0, 0.1*Pr + F)
    # 这里的 F 通常指公式算出来的 potential F。
    # 即使 S2 没水了，Direct Branch 的 F 是由参数决定的外部通量，
    # 但 Direct Branch 的水量也是有限的 (0.1Pr)。
    
    flux_qd_potential = flux_q1_direct + flux_fr_potential # 注意：这里通常用 Potential F
    flux_qd = F.relu(flux_qd_potential)
    
    # 计算 Direct Branch 的实际交换量
    # 它的逻辑是：如果 (0.1Pr + F) < 0，说明 F 把 0.1Pr 吸干了，且 F 还有剩余吸力没处使
    # 所以实际损失被限制在 -0.1Pr
    actual_exchange_direct = flux_qd - flux_q1_direct

    # Total Streamflow
    Qsim = flux_qr + flux_qd
    
    # Total Evaporation (Physical)
    E_physical = flux_ef + flux_es

    # Total Exchange (Physical Gain/Loss)
    # 必须使用 *Actual* S2 Exchange + *Actual* Direct Exchange
    total_exchange = flux_fr_actual + actual_exchange_direct

    # Unified "E" for Water Balance Check
    # P = Q + (E_phys - Total_Exchange) + dS
    Ea = E_physical - total_exchange

    return Qsim, Ea, S1_new, S2_new