import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.saturation import saturation_7, saturation_1
from ..flux.evap import evap_3
from ..flux.interflow import interflow_10
from ..flux.baseflow import baseflow_4

# Parameter range dictionary (based on MARRMoT m_14_topmodel_7p_2s)
TOPMODEL_PARAMS_BOUNDS = {
    "suzmax": [1.0, 2000.0],  # Max soil moisture storage in unsatured zone [mm]
    "st": [0.05, 0.95],  # Threshold fraction [-]
    "kd": [0.0, 1.0],  # Leakage coefficient [mm/d]
    "q0": [0.1, 200.0],  # Zero deficit base flow speed [mm/d]
    "f": [0.0, 1.0],  # Baseflow scaling coefficient [mm-1]
    "chi": [1.0, 7.5],  # Gamma distribution parameter [-]
    "phi": [0.1, 5.0],  # Gamma distribution parameter [-]
}

# Parameter description dictionary
TOPMODEL_PARAMS_DESC = {
    "suzmax": "Maximum soil moisture storage in unsaturated zone [mm]",
    "st": "Threshold for flow generation and evap change as fraction of suzmax [-]",
    "kd": "Leakage to saturated zone flow coefficient [mm/d]",
    "q0": "Zero deficit base flow speed [mm/d]",
    "f": "Baseflow scaling coefficient [mm-1]",
    "chi": "Gamma distribution parameter [-]",
    "phi": "Gamma distribution parameter [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for TOPMODEL.
    S1: Unsaturated storage
    S2: Saturated zone deficit (0 = fully saturated)
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def topmodel_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TOPMODEL_PARAMS_BOUNDS keys
    suzmax: torch.Tensor,
    st: torch.Tensor,
    kd: torch.Tensor,
    q0: torch.Tensor,
    f: torch.Tensor,
    chi: torch.Tensor,
    phi: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TOPMODEL single-step calculation.
    [Gradient Safe] Maintains nearzero floor for stability.
    [Mass Conservative] Fluxes are limited strictly to prevent clamp leakage.
    [Deficit] S2 is treated as Deficit Store.
    """

    # ==========================================================================
    # 1. Surface Runoff (S2 Control)
    # ==========================================================================
    mu_fixed = torch.tensor(3.0, device=P.device)
    lambda_para = chi * phi + mu_fixed
    
    # flux_qof: 饱和坡面流
    # 这里不需要改，clamp(min=0) 对 flux 是安全的，物理上 flux 就是非负的
    flux_qof = saturation_7(
        chi, phi, mu_fixed, lambda_para, f, S2, P, nearzero=nearzero
    )
    flux_qof = torch.clamp(flux_qof, min=torch.zeros_like(P), max=P)

    # Peff: 进入非饱和带的水
    flux_peff = P - flux_qof

    # ==========================================================================
    # 2. Unsaturated Zone (S1: Positive Storage)
    # ==========================================================================
    
    # 2.1 蒸发 (Evaporation)
    # 计算潜在蒸发
    flux_ea_pot = evap_3(st, S1, suzmax, PET, nearzero=nearzero)
    
    # [梯度安全修正]
    # 计算 S1 中“自由可用”的水量 (保留 nearzero 作为底)
    # 使用 ReLU 保证梯度在 S1 > nearzero 时为 1，否则为 0
    s1_free = F.relu(S1 - nearzero)
    
    # 限制实际蒸发不能超过自由水量
    flux_ea = torch.minimum(flux_ea_pot, s1_free)
    
    # 2.2 S1 接收雨水并扣除蒸发
    # 此时 S1 肯定 >= nearzero
    S1_tmp = S1 + flux_peff - flux_ea
    
    # 2.3 饱和溢出 (Saturation Excess)
    # 当 S1 > suzmax 时产生溢流，使用 smooth saturation_1 保持梯度连续
    # MATLAB: saturation_1(flux_peff, S1, suzmax)
    flux_qex = saturation_1(flux_peff, S1_tmp, suzmax, nearzero=nearzero)
    flux_qex = torch.clamp(flux_qex, min=torch.zeros_like(flux_peff), max=flux_peff)

    # 扣除溢流
    S1_tmp = S1_tmp - flux_qex

    # 2.4 壤中流/补给 (Recharge / Interflow to S2)
    threshold_s1 = st * suzmax
    capacity_s1 = suzmax - threshold_s1

    flux_qv_pot = interflow_10(
        S1_tmp, kd, threshold_s1, capacity_s1, nearzero=nearzero
    )
    
    # [关键点：双向限制]
    # 限制 1: 不能把 S1 抽干 (保留 nearzero)
    s1_free_now = F.relu(S1_tmp - nearzero)
    
    # 限制 2 (防止 S2 溢出):
    # S2 是亏缺，如果 Recharge(qv) > Deficit(S2)，那么 S2 就会变成负数（过饱和）。
    # 在标准 TOPMODEL 中，这意味着地下水完全饱和，qv 应该停止或者变成 qof。
    # 为了守恒，我们限制 qv 不能超过 S2 当前的亏缺空间。
    # 同时考虑到 Baseflow (qb) 会腾出空间： Max_In = Current_Deficit + Outflow - Safety
    flux_qb_pot = baseflow_4(q0, f, S2, nearzero=nearzero)
    
    # S2 的可用接收空间 = (S2 - nearzero) + qb
    # 如果 S2 已经很小(接近 nearzero)，说明饱和了，接不进水了
    s2_space = F.relu(S2 - nearzero) + flux_qb_pot
    
    # qv 取三者最小值：潜在量、S1供水量、S2接收空间
    flux_qv = torch.minimum(flux_qv_pot, s1_free_now)
    flux_qv = torch.minimum(flux_qv, s2_space)

    # 更新 S1
    S1_new = S1_tmp - flux_qv
    # 这里的 clamp 只是为了消除 1e-18 这种计算浮点误差，不会截断物理水量
    # 因为我们上面已经保证了 flux_qv <= S1_tmp - nearzero
    S1_new = torch.clamp(S1_new, min=nearzero)

    # ==========================================================================
    # 3. Saturated Zone (S2: Deficit Store)
    # ==========================================================================
    # S2: 亏缺层。
    # 增加亏缺(变干): flux_qb
    # 减少亏缺(变湿): flux_qv
    
    # 重新计算 qb (虽然上面算过 pot，但因为 S2 没变，是一样的)
    flux_qb = baseflow_4(q0, f, S2, nearzero=nearzero)
    
    # 更新 S2
    S2_new = S2 + flux_qb - flux_qv
    
    # 这里的 clamp 同样是安全的，因为我们上面限制了 flux_qv <= S2 + qb - nearzero
    # 所以 S2 + qb - qv 必然 >= nearzero
    S2_new = torch.clamp(S2_new, min=nearzero)

    # ==========================================================================
    # 4. Output Aggregation
    # ==========================================================================
    Qsim = flux_qof + flux_qex + flux_qb
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new
