import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

# 引入通量计算函数
from dmg.models.phy_models.flux.saturation import saturation_3
from dmg.models.phy_models.flux.evap import evap_3
from dmg.models.phy_models.flux.percolation import percolation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.baseflow import baseflow_1

# 引入单位线 (Triangular UH)
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3


# ==============================================================================
# 1. Parameter Definitions
# ==============================================================================
FLEXB_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "beta": [0.0, 10.0],     # Unsaturated zone shape parameter [-]
    "d_split": [0.0, 1.0],   # Fast/slow runoff distribution parameter [-]
    "percmax": [0.0, 20.0],  # Maximum percolation rate [mm/d]
    "lp": [0.05, 0.95],      # Wilting point as fraction of s1max [-]
    "nlagf": [1.0, 5.0],     # Flow delay before fast runoff [d]
    "nlags": [1.0, 15.0],    # Flow delay before slow runoff [d]
    "kf": [0.0, 1.0],        # Fast runoff coefficient [d-1]
    "ks": [0.0, 1.0],        # Slow runoff coefficient [d-1]
}

FLEXB_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "beta": "Unsaturated zone shape parameter [-]",
    "d_split": "Fast/slow runoff distribution parameter [-]",
    "percmax": "Maximum percolation rate [mm/d]",
    "lp": "Wilting point as fraction of s1max [-]",
    "nlagf": "Flow delay before fast runoff [d]",
    "nlags": "Flow delay before slow runoff [d]",
    "kf": "Fast runoff coefficient [d-1]",
    "ks": "Slow runoff coefficient [d-1]",
}


# ==============================================================================
# 2. Static Step Functions (Compiled)
# ==============================================================================

def _flexb_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor, # Unsaturated Soil
    s1max: torch.Tensor,
    beta: torch.Tensor,
    d_split: torch.Tensor,
    percmax: torch.Tensor,
    lp: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 1: Production Step
    Calculates S1 dynamics (Infiltration, Evap, Percolation).
    
    Returns:
    - flux_rf: Fast flow component (to be delayed)
    - flux_slow_in: Slow flow component (rs + ps) (to be delayed)
    - flux_eur: Actual Evap
    - S1_new: Updated State
    """
    # --- 1. Infiltration ---
    flux_ru = saturation_3(S1, s1max, beta, P, nearzero=nearzero)
    zeros = torch.zeros_like(flux_ru)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=P)

    # Surface Excess
    p_excess = F.relu(P - flux_ru)

    # Split Excess
    flux_rf = split_1(1.0 - d_split, p_excess, nearzero=nearzero)
    flux_rs = F.relu(p_excess - flux_rf)

    # --- 2. State Update S1 ---
    S1_tmp = torch.clamp(S1 + flux_ru, min=nearzero)

    # Evaporation
    flux_eur = evap_3(lp, S1_tmp, s1max, PET, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S1_tmp - nearzero)
    flux_eur = torch.minimum(flux_eur, PET)
    flux_eur = F.relu(flux_eur)

    S1_tmp2 = torch.clamp(S1_tmp - flux_eur, min=nearzero)

    # Percolation
    flux_ps = percolation_2(percmax, S1_tmp2, s1max, nearzero=nearzero)
    flux_ps = torch.minimum(flux_ps, S1_tmp2 - nearzero)
    flux_ps = F.relu(flux_ps)

    S1_new = torch.clamp(S1_tmp2 - flux_ps, min=nearzero)

    # --- 3. Outputs for Routing ---
    # Slow branch receives both Split Excess (rs) and Percolation (ps)
    flux_slow_in = flux_rs + flux_ps

    return flux_rf, flux_slow_in, flux_eur, S1_new


def _flexb_routing_step_impl(
    flux_rfl: torch.Tensor, # Routed Fast Inflow
    flux_rsl: torch.Tensor, # Routed Slow Inflow
    S2: torch.Tensor,       # Fast Reservoir
    S3: torch.Tensor,       # Slow Reservoir
    kf: torch.Tensor,
    ks: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 3: Routing Step (Linear Reservoirs S2 & S3)
    """
    # --- Fast Store (S2) ---
    S2_tmp = torch.clamp(S2 + flux_rfl, min=nearzero)
    
    flux_qf = baseflow_1(kf, S2_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S2_tmp - nearzero)
    flux_qf = F.relu(flux_qf)
    
    S2_new = torch.clamp(S2_tmp - flux_qf, min=nearzero)

    # --- Slow Store (S3) ---
    S3_tmp = torch.clamp(S3 + flux_rsl, min=nearzero)
    
    flux_qs = baseflow_1(ks, S3_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S3_tmp - nearzero)
    flux_qs = F.relu(flux_qs)
    
    S3_new = torch.clamp(S3_tmp - flux_qs, min=nearzero)

    # Total Q
    Qsim = flux_qf + flux_qs

    return Qsim, S2_new, S3_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (FlexbModel)
# ==============================================================================

class Flexb(UnifyV1):
    """
    Flex-B Hydrological Model (MARRMoT m_21)
    
    Architecture: Sandwich (Loop -> Conv -> Loop)
    1. Production: Generates fast (rf) and slow (rs+ps) inflows.
    2. Convolution: Delays flows using Triangular UH (nlagf, nlags).
    3. Routing: Routes delayed flows through Linear Reservoirs (S2, S3).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None: config = {}
        config.setdefault("model_name", "flexb")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrographs (Half Triangle)
        self.uh_fast = DplTri3(max_lag=int(FLEXB_PARAMS_BOUNDS["nlagf"][1]))
        self.uh_slow = DplTri3(max_lag=int(FLEXB_PARAMS_BOUNDS["nlags"][1]))
        self.production_step = _maybe_compile(_flexb_production_step_impl, self.backend)
        self.routing_step = _maybe_compile(_flexb_routing_step_impl, self.backend)

    def _init_states(self, n_grid: int, nmul: int = None) -> Tuple[torch.Tensor, ...]:
        """S1: Unsaturated, S2: Fast, S3: Slow"""
        nmul = nmul or self.nmul
        S1 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S2 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S3 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        return (S1, S2, S3)

    def _run_model(
        self,
        x: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        nmul: int = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x['x_phy']
        n_steps, n_grid = forcing.shape[:2]
        nmul = nmul or self.nmul
        nearzero = self.nearzero

        # --- A. Data Prep ---
        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        # T_seq unused
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        s1max = static_params["s1max"]
        beta = static_params["beta"]
        d_split = static_params["d_split"]
        percmax = static_params["percmax"]
        lp = static_params["lp"]
        nlagf = static_params["nlagf"]
        nlags = static_params["nlags"]
        kf = static_params["kf"]
        ks = static_params["ks"]

        S1, S2, S3 = states
        warm_up = min(self.warm_up, n_steps)

        with torch.no_grad():
            for t in range(warm_up):
                _, _, _, S1 = self.production_step(
                    P_seq[t], PET_seq[t], S1,
                    s1max, beta, d_split, percmax, lp, nearzero)
        S1 = S1.detach()

        # ==========================================================
        # Phase 1: Production Loop
        # ==========================================================
        raw_rf_list = []
        raw_slow_list = []

        for t in range(n_steps):
            flux_rf, flux_slow_in, flux_eur, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1,
                s1max, beta, d_split, percmax, lp,
                nearzero
            )
            raw_rf_list.append(flux_rf)
            raw_slow_list.append(flux_slow_in)

        # Stack: (T, B, M)
        rf_stack = torch.stack(raw_rf_list, dim=0)
        slow_stack = torch.stack(raw_slow_list, dim=0)

        # ==========================================================
        # Phase 2: Parallel Convolution (Fast & Slow)
        # ==========================================================
        # 1. Flatten for Conv1d: (B*M, T)
        B_total = n_grid * nmul
        rf_flat = rf_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        slow_flat = slow_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        
        # 2. UH Params: (B*M, 1)
        nlagf_flat = nlagf.reshape(B_total, 1)
        nlags_flat = nlags.reshape(B_total, 1)

        # 3. Apply Convolution
        routed_rf_flat = self.uh_fast(rf_flat, nlagf_flat)
        routed_slow_flat = self.uh_slow(slow_flat, nlags_flat)

        # 4. Reshape back & Unbind: List[Tensor]
        rfl_seq = routed_rf_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)
        rsl_seq = routed_slow_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)

        # ==========================================================
        # Phase 3: Routing Loop (S2, S3)
        # ==========================================================
        Qsim_list = []

        for t in range(n_steps):
            # Pass convolved fluxes to reservoirs
            Qsim, S2, S3 = self.routing_step(
                rfl_seq[t], rsl_seq[t], S2, S3,
                kf, ks,
                nearzero
            )
            Qsim_list.append(Qsim)

        Qsim_out = torch.stack(Qsim_list, dim=0)

        warm_up = min(self.warm_up, n_steps)
        return {"streamflow": Qsim_out[warm_up:].flatten(start_dim=1)}