import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v2 import UnifyV2

# 引入通量计算函数
from dmg.models.phy_models.flux.evap import evap_11
from dmg.models.phy_models.flux.saturation import saturation_4
# from dmg.models.phy_models.flux.percolation import percolation_3
from dmg.models.phy_models.flux.recharge import recharge_2
# from dmg.models.phy_models.flux.baseflow import baseflow_3

# 引入 GR4J 单位线
from dmg.models.phy_models.unithydro.uh_half_1 import DplHalf1
from dmg.models.phy_models.unithydro.uh_full_2 import DplFull2

# Parameter range dictionary (based on MARRMoT m_07_gr4j_4p_2s)
GR4J_PARAMS_BOUNDS = {
    "x1": [1.0, 2000.0],  # Max soil moisture storage [mm]
    "x2": [-20.0, 20.0],  # Water exchange coefficient [mm/d]
    "x3": [1.0, 300.0],   # Max routing store storage [mm]
    "x4": [0.5, 15.0],    # Flow delay [d] (align with core version)
}

def percolation_3(
    S: torch.Tensor, 
    Smax: torch.Tensor, 
    nearzero: float = 1e-6
) -> torch.Tensor:
    """
    [Safe Version] Non-linear percolation (empirical).
    Original: Smax^(-4) / 4 * (4/9)^4 * S^5
    Refactored: S * (4/9)^4 / 4 * (S/Smax)^4
    """
    # 1. 安全比率计算
    # 将 S^5 拆解为 S * (S/Smax)^4，这样我们可以控制底数
    denom = Smax + nearzero
    ratio = S / denom
    
    # 2. 【关键】截断比率 (Ratio Clamping)
    # 限制 S/Smax 最大为 1.5。
    # 如果 S 很大，(1.5)^4 = 5.06，这是一个常数。
    # 此时函数变为 Linear: Out = k * S * 5.06，梯度恒定，不会 NaN。
    ratio_safe = torch.clamp(ratio, max=1.5)
    
    # 3. 计算常数项 (4/9)^4 / 4
    # (4/9)^4 approx 0.039, divided by 4 approx 0.01
    const_term = (4.0 / 9.0) ** 4.0 / 4.0
    
    # 4. 组合公式
    # Out = Const * S * (Ratio_Safe)^4
    return const_term * S * ratio_safe.pow(4.0)


def baseflow_3(
    S: torch.Tensor, 
    Smax: torch.Tensor, 
    nearzero: float = 1e-6
) -> torch.Tensor:
    """
    [Safe Version] Baseflow 3: Empirical non-linear outflow
    Original: Smax^(-4) / 4 * S^5
    Refactored: S / 4 * (S/Smax)^4
    """
    # 1. 安全比率计算
    denom = Smax + nearzero
    ratio = S / denom
    
    # 2. 【关键】截断比率
    # 同样限制底数最大 1.5，防止 5 次幂爆炸
    ratio_safe = torch.clamp(ratio, max=1.5)
    
    # 3. 组合公式
    # Out = 1/4 * S * (Ratio_Safe)^4
    return 0.25 * S * ratio_safe.pow(4.0)

# ==============================================================================
# 1. 定义单步计算函数 (Static Functions)
# ==============================================================================


def _gr4j_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,
    x1: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    单步产流计算 (Single Step Production)
    """
    # 1. Net precipitation and evaporation
    flux_pn = F.relu(P - PET)
    flux_en = F.relu(PET - P)
    flux_ef = P - flux_pn

    # 2. Production store (S1) process
    flux_ps = saturation_4(S1, x1, flux_pn, nearzero=nearzero)
    zeros = torch.zeros_like(flux_ps)
    flux_ps = torch.clamp(flux_ps, min=zeros, max=flux_pn)

    flux_es = evap_11(S1, x1, flux_en, nearzero=nearzero)
    flux_es = torch.minimum(flux_es, S1)

    # Update S1 for percolation
    S1_tmp = S1 + flux_ps - flux_es
    nearzero_tensor = torch.ones_like(x1) * nearzero
    S1_tmp = torch.clamp(S1_tmp, min=nearzero_tensor, max=x1)

    # Percolation
    flux_perc = percolation_3(S1_tmp, x1, nearzero=nearzero)
    flux_perc = torch.minimum(flux_perc, S1_tmp)

    # Final S1 update
    S1_new = torch.clamp(S1_tmp - flux_perc, min=nearzero_tensor, max=x1)

    # 3. Calculate Effective Rainfall (Pr)
    flux_pr = (flux_pn - flux_ps) + flux_perc

    # Total Evap (Optional, kept for completeness)
    flux_ea = flux_ef + flux_es

    return flux_pr, flux_ea, S1_new


def _gr4j_routing_step_impl(
    flux_q9: torch.Tensor,  # 当前时刻已经卷积滞后过的 Q9
    flux_q1: torch.Tensor,  # 当前时刻已经卷积滞后过的 Q1
    S2: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    单步汇流计算 (Single Step Routing)
    """
    # Constant for recharge (create inside or pass in, tensor scalar is cheap)
    p1_recharge = torch.tensor(3.5, device=S2.device, dtype=S2.dtype)

    # 1. Groundwater exchange (potential)
    flux_fr_potential = recharge_2(p1_recharge, S2, x3, x2, nearzero=nearzero)

    # 2. Apply exchange with clamp to avoid over-draw
    S2_before_exchange = S2
    S2_temp = S2 + flux_fr_potential
    S2_after_exchange = torch.clamp(S2_temp, min=nearzero)
    flux_fr_actual = S2_after_exchange - S2_before_exchange
    S2 = S2_after_exchange

    # 3. Routing store outflow (Qr)
    flux_qr = baseflow_3(S2, x3, nearzero=nearzero)
    flux_qr = torch.minimum(flux_qr, S2)
    S2_new = torch.clamp(S2 - flux_qr, min=nearzero)

    # 4. Direct branch uses potential exchange (consistent with GR4J spec)
    flux_qd_potential = flux_q1 + flux_fr_potential
    flux_qd = F.relu(flux_qd_potential)

    # Total Flow
    Q_total = flux_qr + flux_qd

    return Q_total, S2_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. 模型类 (GR4JModel)
# ==============================================================================


class Gr4j(UnifyV2):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "gr4j")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrographs
        # UH1 (Half Bell): 需要覆盖 x4 的范围
        # UH2 (Full Bell): 需要覆盖 2*x4 的范围，但传入参数已经是 2*x4
        # 所以 UH2 的 max_lag 需要是 2 * (2 * x4_max) = 4 * x4_max
        max_lag_val = GR4J_PARAMS_BOUNDS["x4"][1]
        # UH1: max_lag 需要 >= x4_max，取 2 倍确保足够
        self.uh_1 = DplHalf1(max_lag=int(max_lag_val)+1)  # For Q9 (x4)
        # UH2: 传入参数是 2*x4，S-Curve 需要覆盖到 2*d = 4*x4
        self.uh_2 = DplFull2(max_lag=int(max_lag_val)*2+2)  # For Q1 (2*x4)
        self.production_step = _maybe_compile(_gr4j_production_step_impl, self.backend)
        self.routing_step = _maybe_compile(_gr4j_routing_step_impl, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        """S1: Production, S2: Routing"""
        S1 = (
            torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
        )
        S2 = (
            torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
        )
        return (S1, S2)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul
        nearzero = self.nearzero

        # --- A. Data Prep ---
        # 使用 unbind 将 Tensor 拆解为 Tuple[Tensor, ...]
        # 这样在 Python 循环中 zip 迭代比 tensor[t] 索引快
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters (B, nmul)
        x1 = static_params["x1"]
        x2 = static_params["x2"]
        x3 = static_params["x3"]
        x4 = static_params["x4"]
        
        S1, S2 = states

        track_balance = self.check_water_balance
        if track_balance:
            Et_out = torch.empty(
                (n_steps, n_grid, nmul), device=self.device, dtype=torch.float32
            )
            state_series: Optional[List[torch.Tensor]] = [
                torch.empty(
                    (n_steps + 1, n_grid, nmul),
                    device=self.device,
                    dtype=torch.float32,
                )
                for _ in range(2)
            ]
            state_series[0][0] = S1
            state_series[1][0] = S2
            S_init_sum = torch.stack([s.clone() for s in states]).sum(dim=0)
        else:
            Et_out = None
            state_series = None
            S_init_sum = None

        # ==========================================================
        # Phase 1: Production Loop (Python Loop + Compiled Step)
        # ==========================================================
        flux_pr_list = []
        # flux_ea_list = [] # 如果不需要输出 Ea 可注释掉以加速

        # Python Loop
        for t in range(n_steps):
            # 调用编译好的单步函数
            flux_pr, flux_ea, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1, x1, nearzero
            )
            flux_pr_list.append(flux_pr)
            if track_balance:
                Et_out[t] = flux_ea
                state_series[0][t + 1] = S1

        # Stack outputs: (T, B, M)
        flux_pr_stack = torch.stack(flux_pr_list, dim=0)

        # ==========================================================
        # Phase 2: Parallel Convolution (Sandwich Middle)
        # ==========================================================
        # 1. Split Pr (90% / 10%)
        flux_pr90 = flux_pr_stack * 0.9
        flux_pr10 = flux_pr_stack * 0.1

        # 2. Flatten for Conv1d: (T, B, M) -> (B*M, T)
        B_total = n_grid * nmul
        # permute(1,2,0) -> (B, M, T), reshape -> (B*M, T)
        pr90_flat = flux_pr90.permute(1, 2, 0).reshape(B_total, n_steps)
        pr10_flat = flux_pr10.permute(1, 2, 0).reshape(B_total, n_steps)

        # 3. Prepare UH Params: (B*M, 1)
        x4_flat = x4.reshape(B_total, 1)
        # 4. Apply Convolution (PyTorch Native, fast)
        routed_q9_flat = self.uh_1(pr90_flat, x4_flat)
        routed_q1_flat = self.uh_2(pr10_flat, x4_flat)

        # 5. Reshape back and Unbind for Routing Loop
        # (B*M, T) -> (B, M, T) -> (T, B, M) -> List[Tensor]
        # unbind(2) 对应 permute 后的 Time 维度
        q9_seq = (
            routed_q9_flat.view(n_grid, nmul, n_steps)
            .permute(2, 0, 1)
            .unbind(0)
        )
        q1_seq = (
            routed_q1_flat.view(n_grid, nmul, n_steps)
            .permute(2, 0, 1)
            .unbind(0)
        )

        # ==========================================================
        # Phase 3: Routing Loop (Python Loop + Compiled Step)
        # ==========================================================
        Qsim_list = []

        # Python Loop
        for t in range(n_steps):
            q_total, S2 = self.routing_step(
                q9_seq[t], q1_seq[t], S2, x2, x3, nearzero
            )
            Qsim_list.append(q_total)
            if track_balance and state_series is not None:
                state_series[1][t + 1] = S2

        Qsim_out = torch.stack(Qsim_list, dim=0)
        final_states = (S1, S2)

        if track_balance:
            return self._finalize_output(
                Qsim_out,
                Et_out,
                S_init_sum,
                final_states,
                state_series,
            )

        return self._finalize_output(Qsim_out)