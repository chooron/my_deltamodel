import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v1 import UnifyV1

# 引入通量计算函数
from dmg.models.phy_models.flux.evap import evap_1, evap_6, evap_5
from dmg.models.phy_models.flux.interception import interception_1
from dmg.models.phy_models.flux.saturation import saturation_1
from dmg.models.phy_models.flux.interflow import interflow_9
from dmg.models.phy_models.flux.baseflow import baseflow_1

# 引入单位线: uh_4_full (Full Triangular)
from dmg.models.phy_models.unithydro.uh_tri_4 import DplTri4


# ==============================================================================
# 1. Parameter Definitions
# ==============================================================================
NEWZEALAND2_PARAMS_BOUNDS = {
    "s1max": [0.0, 5.0],  # Maximum interception storage [mm]
    "s2max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "sfc_frac": [0.05, 0.95],  # Field capacity fraction [-]
    "m": [0.05, 0.95],  # Fraction forest [-]
    "a": [0.0, 1.0],  # Subsurface runoff coefficient [d-1]
    "b": [1.0, 5.0],  # Runoff non-linearity [-]
    "tcbf": [0.0, 1.0],  # Baseflow runoff coefficient [d-1]
    "d_delay": [1.0, 30.0],  # Routing time delay [d] (UH base)
}

NEWZEALAND2_PARAMS_DESC = {
    "s1max": "Maximum interception storage [mm]",
    "s2max": "Maximum soil moisture storage [mm]",
    "sfc_frac": "Field capacity as fraction of maximum soil moisture [-]",
    "m": "Fraction forest [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "b": "Runoff non-linearity [-]",
    "tcbf": "Baseflow runoff coefficient [d-1]",
    "d_delay": "Routing time delay [d]",
}


# ==============================================================================
# 2. Static Step Function (Compiled)
# ==============================================================================


def _newzealand2_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,  # Interception
    S2: torch.Tensor,  # Soil Moisture
    s1max: torch.Tensor,
    s2max: torch.Tensor,
    sfc_frac: torch.Tensor,
    m: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tcbf: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 1: Production Step
    Calculates Interception and Soil Moisture Dynamics.
    Generates total instantaneous runoff (q_total) to be routed.
    """

    # --- 1. Interception process (S1) ---
    # Throughfall
    flux_qtf = interception_1(P, S1, s1max, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qtf)
    flux_qtf = torch.clamp(flux_qtf, min=zeros, max=P)

    # S1 update (Interim)
    S1_tmp = S1 + P - flux_qtf
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # Evaporation from Interception
    flux_eint = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_eint = torch.minimum(flux_eint, S1_tmp - nearzero)
    flux_eint = torch.minimum(flux_eint, PET)
    flux_eint = F.relu(flux_eint)

    S1_new = torch.clamp(S1_tmp - flux_eint, min=nearzero)

    # --- 2. Soil moisture process (S2) ---
    # Saturation Excess (Fast Runoff)
    flux_qse = saturation_1(flux_qtf, S2, s2max, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=flux_qtf)

    S2_tmp = S2 + flux_qtf - flux_qse
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Evaporation from Soil
    pet_rem = F.relu(PET - flux_eint)

    # Weighted evap (Vegetated + Bare Soil)
    flux_veg = evap_6(m, sfc_frac, S2_tmp, s2max, pet_rem, nearzero=nearzero)
    flux_ebs = evap_5(m, S2_tmp, s2max, pet_rem, nearzero=nearzero)

    flux_ea_s2 = flux_veg + flux_ebs

    # Constraints
    flux_ea_s2 = torch.minimum(flux_ea_s2, S2_tmp - nearzero)
    flux_ea_s2 = torch.minimum(flux_ea_s2, pet_rem)
    flux_ea_s2 = F.relu(flux_ea_s2)

    S2_tmp2 = torch.clamp(S2_tmp - flux_ea_s2, min=nearzero)

    # Subsurface Runoff (Interflow)
    # Note: sfc_frac is fraction, need threshold amount
    sfc_threshold = sfc_frac * s2max
    flux_qss = interflow_9(S2_tmp2, a, sfc_threshold, b, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S2_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    S2_tmp3 = torch.clamp(S2_tmp2 - flux_qss, min=nearzero)

    # Baseflow
    flux_qbf = baseflow_1(tcbf, S2_tmp3, nearzero=nearzero)
    flux_qbf = torch.minimum(flux_qbf, S2_tmp3 - nearzero)
    flux_qbf = F.relu(flux_qbf)

    S2_new = torch.clamp(S2_tmp3 - flux_qbf, min=nearzero)

    # --- 3. Total Instantaneous Runoff ---
    flux_q_total = flux_qse + flux_qss + flux_qbf

    # Total Evap
    flux_ea_total = flux_eint + flux_ea_s2

    return flux_q_total, flux_ea_total, S1_new, S2_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (Newzealand2)
# ==============================================================================


class Newzealand2(UnifyV1):
    """
    New Zealand Model v2 (MARRMoT m_16)

    Architecture:
    1. Production: Generates total runoff (q_total).
    2. Convolution: Routes q_total using a Full Triangular Unit Hydrograph (uh_4_full).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "newzealand2")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrograph (Full Triangle)
        # Parameter: d_delay
        self.uh = DplTri4(max_lag=int(NEWZEALAND2_PARAMS_BOUNDS["d_delay"][1]))
        self.production_step = _maybe_compile(_newzealand2_production_step_impl, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        """S1: Interception, S2: Soil"""
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

        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        # T_seq unused
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        s1max = static_params["s1max"]
        s2max = static_params["s2max"]
        sfc_frac = static_params["sfc_frac"]
        m = static_params["m"]
        a = static_params["a"]
        b = static_params["b"]
        tcbf = static_params["tcbf"]
        d_delay = static_params["d_delay"]

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
        # Phase 1: Production Loop
        # ==========================================================
        raw_q_total_list = []
        # ea_list = []

        for t in range(n_steps):
            flux_q_total, flux_ea, S1, S2 = self.production_step(
                P_seq[t],
                PET_seq[t],
                S1,
                S2,
                s1max,
                s2max,
                sfc_frac,
                m,
                a,
                b,
                tcbf,
                nearzero,
            )
            raw_q_total_list.append(flux_q_total)
            if track_balance:
                Et_out[t] = flux_ea
                state_series[0][t + 1] = S1
                state_series[1][t + 1] = S2

        # Stack: (T, B, M)
        q_total_stack = torch.stack(raw_q_total_list, dim=0)
        
        # ==========================================================
        # Phase 2: Convolution (Full Triangle)
        # ==========================================================
        # 1. Flatten for Conv1d: (B*M, T)
        B_total = n_grid * nmul
        q_total_flat = q_total_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        # 2. UH Params: (B*M, 1)
        d_delay_flat = d_delay.reshape(B_total, 1)

        # 3. Apply Convolution
        routed_q_flat = self.uh(q_total_flat, d_delay_flat)

        # 4. Reshape Final Output: (T, B, M)
        Qsim_out = routed_q_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1)
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
