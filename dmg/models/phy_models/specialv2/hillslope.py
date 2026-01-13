import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v2 import UnifyV2

# 引入通量计算函数
from dmg.models.phy_models.flux.interception import interception_2
from dmg.models.phy_models.flux.evap import evap_1
from dmg.models.phy_models.flux.saturation import saturation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.capillary import capillary_2
from dmg.models.phy_models.flux.baseflow import baseflow_1

# 引入单位线 (Triangular UH)
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3


# ==============================================================================
# 1. Parameter Bounds & Description
# ==============================================================================
HILLSLOPE_PARAMS_BOUNDS = {
    "dw": [0.0, 5.0],        # Interception capacity [mm]
    "betaw": [0.0, 10.0],    # Soil moisture distribution parameter [-]
    "swmax": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "a": [0.0, 1.0],         # Surface/groundwater split fraction [-]
    "th": [1.0, 120.0],      # Routing delay [d]
    "c_rad": [0.0, 4.0],     # Rate of capillary rise [mm/d]
    "kh": [0.0, 1.0],        # Groundwater runoff coefficient [d-1]
}

HILLSLOPE_PARAMS_DESC = {
    "dw": "Daily interception capacity [mm]",
    "betaw": "Soil moisture storage distribution parameter [-]",
    "swmax": "Maximum soil moisture storage [mm]",
    "a": "Division parameter for surface and groundwater flow [-]",
    "th": "Time delay for routing [d]",
    "c_rad": "Rate of capillary rise [mm/d]",
    "kh": "Groundwater runoff coefficient [d-1]",
}


# ==============================================================================
# 2. Static Step Functions (Compiled)
# ==============================================================================

def _hillslope_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor, # Soil
    S2: torch.Tensor, # Groundwater
    dw: torch.Tensor,
    betaw: torch.Tensor,
    swmax: torch.Tensor,
    a: torch.Tensor,
    c_rad: torch.Tensor,
    kh: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 1: Combined Production & Groundwater Step
    Calculates S1 (Soil) and S2 (Groundwater) dynamics together.
    
    Returns:
    - flux_qses: Surface runoff (Needs Routing)
    - flux_qhgw: Groundwater flow (Direct Output)
    - flux_ea: Actual Evap
    - S1_new: Updated Soil state
    - S2_new: Updated Groundwater state
    """
    # 1. Inflow + Interception
    flux_pe = interception_2(P, dw, nearzero=nearzero)
    # flux_ei (intercepted evap) is implicitly P - flux_pe, stored nowhere? 
    # Original code: flux_ei = F.relu(P - flux_pe)
    # But note: The original code logic doesn't store Interception Water (Si). 
    # It assumes interception evaporates immediately or is lost.
    flux_ei = F.relu(P - flux_pe)

    # 2. Fast Process (Saturation Excess)
    flux_qse = saturation_2(S1, swmax, betaw, flux_pe, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=flux_pe)

    # 3. Flow splitting
    flux_qses = split_1(a, flux_qse, nearzero=nearzero)
    flux_qseg = F.relu(flux_qse - flux_qses)

    # 4. Sequential state updates
    # S1 interim
    S1_tmp = S1 + flux_pe - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # S2 interim (Groundwater receives recharge immediately)
    S2_tmp = S2 + flux_qseg
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # 5. Evaporation (from S1)
    flux_ea_soil = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ea_soil = torch.minimum(flux_ea_soil, S1_tmp - nearzero)
    flux_ea_soil = torch.minimum(flux_ea_soil, PET) # Cap at PET
    flux_ea_soil = F.relu(flux_ea_soil)

    S1_tmp2 = torch.clamp(S1_tmp - flux_ea_soil, min=nearzero)

    # 6. Slow Processes (Capillary Rise & Baseflow)
    # flux_c: Capillary rise S2 -> S1
    flux_c = capillary_2(c_rad, S2_tmp, nearzero=nearzero)
    flux_c = torch.minimum(flux_c, S2_tmp - nearzero)
    flux_c = F.relu(flux_c)

    # Update S2 after capillary loss
    S2_tmp2 = torch.clamp(S2_tmp - flux_c, min=nearzero)

    # flux_qhgw: Baseflow from S2
    flux_qhgw = baseflow_1(kh, S2_tmp2, nearzero=nearzero)
    flux_qhgw = torch.minimum(flux_qhgw, S2_tmp2 - nearzero)
    flux_qhgw = F.relu(flux_qhgw)

    # 7. Final State Updates
    S1_new = torch.clamp(S1_tmp2 + flux_c, min=nearzero)
    S2_new = torch.clamp(S2_tmp2 - flux_qhgw, min=nearzero)

    # Total Evap
    flux_ea_total = flux_ei + flux_ea_soil

    # Output: qses (fast flow to be routed), qhgw (slow flow ready), ea, states
    return flux_qses, flux_qhgw, flux_ea_total, S1_new, S2_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (HillslopeModel)
# ==============================================================================

class Hillslope(UnifyV2):
    """
    Hillslope (FLEX-Topo) Hydrological Model
    
    Architecture: Hybrid Sandwich
    1. Production Loop: Calculates Soil (S1) & Groundwater (S2). 
       Outputs Surface Runoff (Fast) and Baseflow (Slow).
    2. Convolution: Delays Surface Runoff using Triangular UH.
    3. Summation: Q = Delayed_Surface + Baseflow.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None: config = {}
        config.setdefault("model_name", "hillslope")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrograph for Surface Runoff (th)
        # Assuming th is Triangular
        self.uh_surface = DplTri3(max_lag=int(HILLSLOPE_PARAMS_BOUNDS["th"][1]))
        self.production_step = _maybe_compile(_hillslope_production_step_impl, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        """S1: Soil, S2: Groundwater"""
        S1 = torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
        S2 = torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
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
        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        # T_seq unused
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        dw = static_params["dw"]
        betaw = static_params["betaw"]
        swmax = static_params["swmax"]
        a = static_params["a"]
        th = static_params["th"]
        c_rad = static_params["c_rad"]
        kh = static_params["kh"]

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
        # Phase 1: Production & Groundwater Loop
        # ==========================================================
        raw_qses_list = [] # Surface runoff (needs routing)
        raw_qhgw_list = [] # Baseflow (ready)
        # ea_list = []

        for t in range(n_steps):
            flux_qses, flux_qhgw, flux_ea, S1, S2 = self.production_step(
                P_seq[t], PET_seq[t], S1, S2,
                dw, betaw, swmax, a, c_rad, kh,
                nearzero
            )
            raw_qses_list.append(flux_qses)
            raw_qhgw_list.append(flux_qhgw)
            if track_balance:
                Et_out[t] = flux_ea
                state_series[0][t + 1] = S1
                state_series[1][t + 1] = S2

        # Stack: (T, B, M)
        qses_stack = torch.stack(raw_qses_list, dim=0)
        qhgw_stack = torch.stack(raw_qhgw_list, dim=0)

        # ==========================================================
        # Phase 2: Parallel Convolution (Surface Runoff Only)
        # ==========================================================
        # 
        
        # 1. Flatten for Conv1d: (T, B, M) -> (B*M, T)
        B_total = n_grid * nmul
        qses_flat = qses_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        
        # 2. UH Params: (B*M, 1)
        th_flat = th.reshape(B_total, 1)

        # 3. Apply Convolution
        routed_qses_flat = self.uh_surface(qses_flat, th_flat)

        # 4. Reshape back: (B*M, T) -> (T, B, M)
        routed_qses = routed_qses_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1)

        # ==========================================================
        # Phase 3: Aggregation
        # ==========================================================
        # Total Q = Routed Surface Runoff + Baseflow
        Qsim_out = routed_qses + qhgw_stack
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