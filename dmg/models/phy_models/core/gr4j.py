import torch
import torch.nn.functional as F
from typing import Tuple

# ==========================================
# 辅助计算函数 (保持 Tanh 和 解析解 不变)
# ==========================================

def _calc_production_store_tanh(
    S: torch.Tensor, x1: torch.Tensor, Pn: torch.Tensor, En: torch.Tensor, nearzero: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tanh 产流计算"""
    ratio_s_x1 = S / (x1 + nearzero)
    
    # Ps
    tanh_pn_x1 = torch.tanh(Pn / (x1 + nearzero))
    ps_num = x1 * (1.0 - ratio_s_x1.pow(2)) * tanh_pn_x1
    ps_den = 1.0 + ratio_s_x1 * tanh_pn_x1
    ps = ps_num / (ps_den + nearzero)
    
    # Es
    tanh_en_x1 = torch.tanh(En / (x1 + nearzero))
    es_num = S * (2.0 - ratio_s_x1) * tanh_en_x1
    es_den = 1.0 + (1.0 - ratio_s_x1) * tanh_en_x1
    es = es_num / (es_den + nearzero)
    
    return ps, es

def _calc_percolation_analytical(
    S: torch.Tensor, x1: torch.Tensor, nearzero: float
) -> torch.Tensor:
    """解析解下渗"""
    ratio_perc = (4.0 / 9.0) * (S / (x1 + nearzero))
    term_perc = (1.0 + ratio_perc.pow(4)).pow(-0.25)
    perc = S * (1.0 - term_perc)
    return perc

def _calc_routing_outflow_analytical(
    S2: torch.Tensor, x3: torch.Tensor, nearzero: float
) -> torch.Tensor:
    """解析解汇流流出"""
    ratio_s2_x3 = S2 / (x3 + nearzero)
    term_qr = (1.0 + ratio_s2_x3.pow(4)).pow(-0.25)
    qr = S2 * (1.0 - term_qr)
    return qr

# ==========================================
# 主计算步骤 (修正水量平衡)
# ==========================================

def gr4j_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GR4J Step with CLOSED LOOP Water Balance.
    Ensures: P = Qsim + Ea + dS1 + dS2
    Method: The 'Exchange' (F) term is subtracted from Ea.
    """

    # 1. 气象强迫分量
    diff = P - PET
    flux_pn = F.relu(diff)      # 净降雨
    flux_en = F.relu(-diff)     # 净蒸发能力
    flux_ei = P - flux_pn       # 截留蒸发 (Interception)

    # 记录初始状态用于计算 delta S
    S1_init = S1.clone()
    S2_init = S2.clone()
    nearzero_tensor = torch.zeros_like(flux_pn) + nearzero
    # 安全截断参数
    S1 = torch.clamp(S1, min=nearzero_tensor, max=x1)
    S2 = torch.clamp(S2, min=nearzero)

    # ==========================================
    # 2. 产流库 (S1) - 内部平衡
    # ==========================================
    flux_ps, flux_es = _calc_production_store_tanh(S1, x1, flux_pn, flux_en, nearzero)
    
    # 中间更新
    S1_mid = S1 - flux_es + flux_ps
    S1_mid = torch.clamp(S1_mid, min=nearzero_tensor, max=x1)
    
    # 下渗
    flux_perc = _calc_percolation_analytical(S1_mid, x1, nearzero)
    S1_new = S1_mid - flux_perc
    S1_new = torch.clamp(S1_new, min=nearzero_tensor, max=x1)

    # 产流库没有外部交换，只有蒸发
    # delta_S1 = Ps - Es - Perc
    # 物理蒸发 (Physical Evap from S1) = flux_es

    # ==========================================
    # 3. 路径分配
    # ==========================================
    # 有效降雨 Pr
    flux_pr = flux_perc + (flux_pn - flux_ps)
    
    # 分配
    flux_q9 = 0.9 * flux_pr
    flux_q1 = 0.1 * flux_pr

    # ==========================================
    # 4. 汇流库 (S2) - 包含外部交换 F
    # ==========================================
    # 计算理论交换量 F
    flux_f_theoretical = x2 * (S2 / (x3 + nearzero)).pow(3.5)
    
    # 更新 S2 (流入 Q9 + 交换 F)
    # 使用 clamp 保证非负，这可能会改变实际发生的 F
    # S2_temp = S2 + Q9 + F
    S2_before_exchange = S2
    S2_integrated = S2 + flux_q9 + flux_f_theoretical
    S2_integrated = torch.clamp(S2_integrated, min=nearzero)
    
    # 【关键】计算 S2 实际发生的净得失水 (Net Gain/Loss of S2)
    # 这个 Net Change 包含了 Q9 (内部转移) 和 F_actual (外部交换)
    # Actual_Inflow_Total = S2_integrated - S2_before_exchange
    # 其中 F_actual_s2 = Actual_Inflow_Total - flux_q9
    f_actual_s2 = (S2_integrated - S2_before_exchange) - flux_q9

    # 计算流出 Qr
    flux_qr = _calc_routing_outflow_analytical(S2_integrated, x3, nearzero)
    S2_new = S2_integrated - flux_qr

    # ==========================================
    # 5. 直接流 (Qd) - 包含外部交换 F
    # ==========================================
    # Qd = max(0, Q1 + F)
    # 理论上 Direct Branch 的 F 和 S2 的 F 是一样的 (flux_f_theoretical)
    flux_qd_potential = flux_q1 + flux_f_theoretical
    flux_qd = F.relu(flux_qd_potential)
    
    # 【关键】计算 Direct Branch 实际发生的外部交换
    # Qd = Q1 + F_actual_q1
    # F_actual_q1 = Qd - Q1
    f_actual_q1 = flux_qd - flux_q1

    # ==========================================
    # 6. 输出与平衡整合
    # ==========================================
    Qsim = flux_qr + flux_qd
    
    # 总物理蒸发 (Loss)
    E_physical = flux_ei + flux_es 
    
    # 总外部交换 (Gain 为正, Loss 为负)
    F_total_actual = f_actual_s2 + f_actual_q1
    
    # 【核心修正】: 为了满足 P = Q + Ea + dS
    # 我们定义 Ea 为 "Net Water Export to Atmosphere/Groundwater"
    # Ea = E_physical - F_total_actual
    # 解释: 
    #   如果 F > 0 (进水), 相当于 Ea 变小 (甚至为负，说明总进水 > 总蒸发)
    #   如果 F < 0 (出水), 相当于 Ea 变大 (水不仅蒸发了，还漏走了)
    Ea_balanced = E_physical - F_total_actual

    # 验证逻辑 (仅供理解，无需写入代码):
    # dS1 = S1_new - S1_init = flux_ps - flux_es - flux_perc
    # dS2 = S2_new - S2_init = flux_q9 + flux_f_actual_s2 - flux_qr
    # P = flux_pn + flux_ei
    # Q = flux_qr + flux_qd
    # ... 代入后会发现恒等
    
    return Qsim, Ea_balanced, S1_new, S2_new


# Parameter range dictionary (based on MARRMoT m_07_gr4j_4p_2s)
GR4J_PARAMS_BOUNDS = {
    "x1": [1.0, 2000.0],  # Max soil moisture storage [mm]
    "x2": [-20.0, 20.0],  # Water exchange coefficient [mm/d]
    "x3": [1.0, 300.0],   # Max routing store storage [mm]
    "x4": [0.5, 15.0],    # Flow delay [d]
}



def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor
]:
    """
    Create initial states for Flex-IS model.
    S1: Snow store
    S2: Interception store
    S3: Soil moisture store
    S4: Fast routing store
    S5: Slow routing store
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2
