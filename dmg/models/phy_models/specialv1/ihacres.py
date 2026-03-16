import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

# 引入通量计算函数
from dmg.models.phy_models.flux.evap import evap_12
from dmg.models.phy_models.flux.saturation import saturation_5
from dmg.models.phy_models.flux.split import split_1

# 引入单位线
# 1. Exponential Decay UH (for fast/slow routing) -> uh_5_half
from dmg.models.phy_models.unithydro.uh_exp_5 import DplExp5

# 2. Pure Delay UH (for total flow) -> uh_8_delay
from dmg.models.phy_models.unithydro.uh_delay_8 import DplDelay8


# ==============================================================================
# 1. Parameter Bounds (Updated to 7 parameters)
# ==============================================================================
IHACRES_PARAMS_BOUNDS = {
    "lp": [1.0, 2000.0],  # Wilting point [mm]
    "d": [1.0, 2000.0],  # Threshold for flow generation [mm]
    "p": [0.0, 10.0],  # Flow response non-linearity [-]
    "alpha": [0.0, 1.0],  # Fast/slow flow division [-]
    "tau_q": [1.0, 5.0],  # Fast flow routing delay [d]
    "tau_s": [1.0, 30.0],  # Slow flow routing delay [d]
    "tau_d": [1.0, 10.0],  # Pure time delay of total flow [d] (New)
}

IHACRES_PARAMS_DESC = {
    "lp": "Wilting point [mm]",
    "d": "Threshold for flow generation [mm]",
    "p": "Flow response non-linearity [-]",
    "alpha": "Fast/slow flow division [-]",
    "tau_q": "Fast flow routing delay [d]",
    "tau_s": "Slow flow routing delay [d]",
    "tau_d": "Pure time delay of total flow [d]",
}

def evap_linear_deficit(
    S: torch.Tensor, 
    lp: torch.Tensor, 
    Ep: torch.Tensor, 
    nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Linear evaporation decline based on Moisture Deficit.
    
    Physics:
    - When S = 0 (Saturated):  Ea = Ep
    - When S = lp (Empty/Dry): Ea = 0
    - When S > lp (Over Dry):  Ea = 0
    
    Formula:
    Ea = Ep * max(0, 1 - S / lp)
    
    Parameters:
    - S:  Current Moisture Deficit [mm]
    - lp: Wilting Point / Max Deficit Capacity [mm]
    - Ep: Potential Evapotranspiration [mm/d]
    """
    # 1. 计算水分胁迫因子 (1.0 = 无胁迫, 0.0 = 停止蒸发)
    # 使用 clamp(min=0.0) 确保当 S > lp 时蒸发为 0，而不是负数
    stress_factor = torch.clamp(1.0 - S / (lp + nearzero), min=0.0, max=1.0)
    
    # 2. 计算实际蒸发
    return stress_factor * Ep

# ==============================================================================
# 2. Static Step Function (Compiled)
# ==============================================================================
# 产流部分的逻辑不变，因为它只负责生成 uq 和 us，不负责汇流
def _ihacres_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,  # Deficit Store
    lp: torch.Tensor,
    d: torch.Tensor,
    p: torch.Tensor,
    alpha: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 1: Production Step
    Calculates Moisture Deficit (S1) and Effective Rainfall splitting.
    """
    # 1. Evapotranspiration (increases deficit)
    flux_ea = evap_linear_deficit(S1, lp, PET, nearzero=nearzero)
    flux_ea = F.relu(flux_ea)

    # 2. Parameter-based effective rainfall (bounded by P)
    flux_u_calc = saturation_5(S1, d, p, P, nearzero=nearzero)
    flux_u_calc = torch.clamp(flux_u_calc, min=torch.zeros_like(P), max=P)

    # 3. Compute provisional deficit update
    S1_temp = S1 - P + flux_ea + flux_u_calc

    # 4. Capture overflow (saturation excess) when S1_temp < 0
    flux_overflow = F.relu(-S1_temp)

    # 5. Total effective rainfall includes overflow
    flux_u_total = flux_u_calc + flux_overflow

    # 6. Update state (cannot go below nearzero)
    S1_new = torch.clamp(S1_temp, min=nearzero)

    # 7. Split fast/slow branches using total effective rainfall
    flux_uq = split_1(alpha, flux_u_total, nearzero=nearzero)
    flux_us = split_1(1.0 - alpha, flux_u_total, nearzero=nearzero)

    return flux_uq, flux_us, flux_ea, S1_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (IhacresModel)
# ==============================================================================


class Ihacres(UnifyV1):
    """
    IHACRES Hydrological Model (7 Parameters)

    Architecture:
    1. Production Loop: Splits u -> uq, us.
    2. Parallel Conv: uq -> Q_fast, us -> Q_slow (Exp Decay).
    3. Summation: Q_temp = Q_fast + Q_slow.
    4. Serial Conv: Q_temp -> Q_total (Pure Delay).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "ihacres")
        super().__init__(config, device, backend)
        self.parameter_bounds = IHACRES_PARAMS_BOUNDS
        # Initialize Unit Hydrographs

        # 1. Parallel Branches (Exp Decay)
        self.uh_fast = DplExp5(max_lag=int(IHACRES_PARAMS_BOUNDS["tau_q"][1]))
        self.uh_slow = DplExp5(max_lag=int(IHACRES_PARAMS_BOUNDS["tau_s"][1]))

        # 2. Final Series Delay (Pure Delay)
        # uh_8_delay uses tau_d
        self.uh_delay = DplDelay8(
            max_lag=int(IHACRES_PARAMS_BOUNDS["tau_d"][1])
        )
        self.production_step = _maybe_compile(_ihacres_production_step_impl, self.backend)

    def _init_states(self, n_grid: int, nmul: int = None) -> Tuple[torch.Tensor, ...]:
        """S1: Deficit Store"""
        nmul = nmul or self.nmul
        S1 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        return (S1,)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        nmul: int = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = nmul or self.nmul
        nearzero = self.nearzero

        # --- A. Data Prep ---
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters (including tau_d)
        lp = static_params["lp"]
        d = static_params["d"]
        p = static_params["p"]
        alpha = static_params["alpha"]
        tau_q = static_params["tau_q"]
        tau_s = static_params["tau_s"]
        tau_d = static_params["tau_d"]  # New parameter

        (S1,) = states

        # ==========================================================
        # Phase 1: Production Loop
        # ==========================================================
        raw_uq_list = []
        raw_us_list = []

        for t in range(n_steps):
            flux_uq, flux_us, flux_ea, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1, lp, d, p, alpha, nearzero
            )
            raw_uq_list.append(flux_uq)
            raw_us_list.append(flux_us)

        # Stack outputs: (T, B, M)
        uq_stack = torch.stack(raw_uq_list, dim=0)
        us_stack = torch.stack(raw_us_list, dim=0)
        
        # ==========================================================
        # Phase 2: Parallel Convolution (Fast & Slow)
        # ==========================================================

        # 1. Flatten for Conv1d: (B*M, T)
        B_total = n_grid * nmul
        uq_flat = uq_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        us_flat = us_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        # 2. UH Params: (B*M, 1)
        tau_q_flat = tau_q.reshape(B_total, 1)
        tau_s_flat = tau_s.reshape(B_total, 1)

        # 3. Apply Parallel Convolution (Exp Decay)
        routed_uq_flat = self.uh_fast(uq_flat, tau_q_flat)
        routed_us_flat = self.uh_slow(us_flat, tau_s_flat)

        # 4. Summation (Intermediate Q)
        # Q_sum = Q_fast + Q_slow
        q_sum_flat = routed_uq_flat + routed_us_flat

        # 1. Param for delay
        tau_d_flat = tau_d.reshape(B_total, 1)

        # 2. Apply Serial Convolution (Pure Delay)
        routed_total_flat = self.uh_delay(q_sum_flat, tau_d_flat)

        # 3. Reshape Final Output: (T, B, M)
        # Note: DplDelay8 output matches input shape (Batch, Time)
        # Qsim_out = routed_total_flat.view(n_grid, nmul, n_steps).permute(
        #     2, 0, 1
        # )        
        Qsim_out = routed_total_flat.view(n_grid, nmul, n_steps).permute(
            2, 0, 1
        )

        warm_up = min(self.warm_up, n_steps)
        return {"streamflow": Qsim_out[warm_up:].flatten(start_dim=1)}
