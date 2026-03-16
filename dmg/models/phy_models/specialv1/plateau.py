import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

# 引入通量计算函数
from dmg.models.phy_models.flux.interception import interception_2
from dmg.models.phy_models.flux.infiltration import infiltration_4
from dmg.models.phy_models.flux.evap import evap_4
from dmg.models.phy_models.flux.capillary import capillary_2
from dmg.models.phy_models.flux.saturation import saturation_1
from dmg.models.phy_models.flux.baseflow import baseflow_1

# 引入单位线 (Triangular UH, assuming tp uses a triangular response like Hillslope)
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3


# ==============================================================================
# 1. Parameter Bounds
# ==============================================================================
PLATEAU_PARAMS_BOUNDS = {
    "fmax": [0.0, 200.0],  # Max infiltration rate [mm/d]
    "dp": [0.0, 5.0],  # Interception capacity [mm]
    "sumax": [1.0, 2000.0],  # Soil moisture depth [mm]
    "lp": [0.05, 0.95],  # Wilting point fraction [-]
    "p_coeff": [0.0, 1.0],  # Evap coefficient [-]
    "tp": [1.0, 120.0],  # Routing delay [d]
    "c_rise": [0.0, 4.0],  # Capillary rise [mm/d]
    "kp": [0.0, 1.0],  # Base flow time parameter [d-1]
}

PLATEAU_PARAMS_DESC = {
    "fmax": "Maximum infiltration rate [mm/d]",
    "dp": "Interception capacity [mm]",
    "sumax": "Soil moisture depth [mm]",
    "lp": "Wilting point as fraction of Sumax [-]",
    "p_coeff": "Coefficient for moisture constrained evaporation [-]",
    "tp": "Time delay for routing [d]",
    "c_rise": "Rate of capillary rise [mm/d]",
    "kp": "Base flow time parameter [d-1]",
}


# ==============================================================================
# 2. Static Step Function (Compiled)
# ==============================================================================


def _plateau_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,  # Unsaturated
    S2: torch.Tensor,  # Saturated
    fmax: torch.Tensor,
    dp: torch.Tensor,
    sumax: torch.Tensor,
    lp: torch.Tensor,
    p_coeff: torch.Tensor,
    c_rise: torch.Tensor,
    kp: torch.Tensor,
    nearzero: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Phase 1: Production Step
    Calculates coupled S1/S2 dynamics.

    Returns:
    - flux_pie: Surface Runoff Excess (Needs Routing)
    - flux_qpgw: Baseflow (Direct)
    - flux_ea: Actual Evap
    - S1_new, S2_new
    """

    # 1. Precipitation and Interception
    flux_pe = interception_2(P, dp, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pe)

    # 2. Infiltration and Surface Runoff
    # flux_pi: infiltration into S1
    # flux_pie: surface runoff (overland flow) -> needs routing
    flux_pi = infiltration_4(flux_pe, fmax, nearzero=nearzero)
    flux_pi = torch.minimum(flux_pi, flux_pe)
    flux_pie = F.relu(flux_pe - flux_pi)

    # 3. Capillary Rise (from S2 to S1)
    flux_c = capillary_2(c_rise, S2, nearzero=nearzero)
    flux_c = torch.minimum(flux_c, S2 - nearzero)
    flux_c = F.relu(flux_c)

    # 4. Evapotranspiration from S1
    # Update S1 interim for ET
    S1_tmp = S1 + flux_pi + flux_c
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    flux_et = evap_4(PET, p_coeff, S1_tmp, lp, sumax, nearzero=nearzero)
    flux_et = torch.minimum(flux_et, S1_tmp - nearzero)
    flux_et = torch.minimum(flux_et, PET)
    flux_et = F.relu(flux_et)

    S1_tmp2 = torch.clamp(S1_tmp - flux_et, min=nearzero)

    # 5. Percolation / Saturation Excess (S1 -> S2)
    # flux_r: saturation excess driven by inflows (pi + c)
    inflow_s1 = flux_pi + flux_c
    flux_r = saturation_1(inflow_s1, S1_tmp2, sumax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_r)
    flux_r = torch.clamp(flux_r, min=zeros, max=inflow_s1)

    # Final S1 update
    S1_new = torch.clamp(S1_tmp2 - flux_r, min=nearzero)

    # 6. Saturated Store process (S2)
    # Update S2 (in: r, out: c)
    S2_tmp = S2 + flux_r - flux_c
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Baseflow
    flux_qpgw = baseflow_1(kp, S2_tmp, nearzero=nearzero)
    flux_qpgw = torch.minimum(flux_qpgw, S2_tmp - nearzero)
    flux_qpgw = F.relu(flux_qpgw)

    S2_new = torch.clamp(S2_tmp - flux_qpgw, min=nearzero)

    # Total Evap
    flux_ea = flux_ei + flux_et

    return flux_pie, flux_qpgw, flux_ea, S1_new, S2_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (PlateauModel)
# ==============================================================================


class Plateau(UnifyV1):
    """
    Plateau (FLEX-Topo) Model

    Architecture:
    1. Production: Computes Surface Runoff (pie) and Baseflow (qpgw).
    2. Convolution: Routes pie using Triangular UH (tp).
    3. Summation: Q = routed_pie + qpgw.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "plateau")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrograph for Surface Runoff (tp)
        self.uh_surface = DplTri3(max_lag=int(PLATEAU_PARAMS_BOUNDS["tp"][1]))
        self.production_step = _maybe_compile(_plateau_production_step_impl, self.backend)

    def _init_states(self, n_grid: int, nmul: int = None) -> Tuple[torch.Tensor, ...]:
        """S1: Unsaturated, S2: Saturated"""
        nmul = nmul or self.nmul
        S1 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S2 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        return (S1, S2)

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

        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        # T_seq unused
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        fmax = static_params["fmax"]
        dp = static_params["dp"]
        sumax = static_params["sumax"]
        lp = static_params["lp"]
        p_coeff = static_params["p_coeff"]
        tp = static_params["tp"]
        c_rise = static_params["c_rise"]
        kp = static_params["kp"]

        S1, S2 = states

        # ==========================================================
        # Phase 1: Production Loop
        # ==========================================================
        raw_pie_list = []  # Surface runoff (needs routing)
        raw_qpgw_list = []  # Baseflow (direct)
        # ea_list = []

        for t in range(n_steps):
            flux_pie, flux_qpgw, flux_ea, S1, S2 = self.production_step(
                P_seq[t],
                PET_seq[t],
                S1,
                S2,
                fmax,
                dp,
                sumax,
                lp,
                p_coeff,
                c_rise,
                kp,
                nearzero,
            )
            raw_pie_list.append(flux_pie)
            raw_qpgw_list.append(flux_qpgw)

        # Stack: (T, B, M)
        pie_stack = torch.stack(raw_pie_list, dim=0)
        qpgw_stack = torch.stack(raw_qpgw_list, dim=0)

        # ==========================================================
        # Phase 2: Convolution (Surface Runoff Only)
        # ==========================================================
        # 1. Flatten for Conv1d: (B*M, T)
        B_total = n_grid * nmul
        pie_flat = pie_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        # 2. UH Params: (B*M, 1)
        tp_flat = tp.reshape(B_total, 1)

        # 3. Apply Convolution
        routed_pie_flat = self.uh_surface(pie_flat, tp_flat)

        # 4. Reshape back: (T, B, M)
        routed_pie = routed_pie_flat.view(n_grid, nmul, n_steps).permute(
            2, 0, 1
        )

        # ==========================================================
        # Phase 3: Aggregation
        # ==========================================================
        # Q = Routed Surface + Baseflow
        Qsim_out = routed_pie + qpgw_stack

        return {"streamflow": Qsim_out.flatten(start_dim=1)}
