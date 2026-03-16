import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

# 引入通量计算函数
from dmg.models.phy_models.flux.interception import interception_1
from dmg.models.phy_models.flux.evap import evap_1, evap_3
from dmg.models.phy_models.flux.saturation import saturation_3
from dmg.models.phy_models.flux.percolation import percolation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.baseflow import baseflow_1

# 引入单位线 (Triangular UH)
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3


# ==============================================================================
# 1. Parameter Bounds & Description
# ==============================================================================
FLEXI_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "beta": [0.0, 10.0],  # Unsaturated zone shape parameter [-]
    "d_split": [0.0, 1.0],  # Fast/slow runoff distribution parameter [-]
    "percmax": [0.0, 20.0],  # Maximum percolation rate [mm/d]
    "lp": [0.05, 0.95],  # Wilting point as fraction of smax [-]
    "nlagf": [1.0, 5.0],  # Flow delay before fast runoff [d]
    "nlags": [1.0, 15.0],  # Flow delay before slow runoff [d]
    "kf": [0.0, 1.0],  # Fast runoff coefficient [d-1]
    "ks": [0.0, 1.0],  # Slow runoff coefficient [d-1]
    "imax": [0.0, 5.0],  # Maximum interception storage [mm]
}

FLEXI_PARAMS_DESC = {
    "smax": "Maximum soil moisture storage [mm]",
    "beta": "Unsaturated zone shape parameter [-]",
    "d_split": "Fast/slow runoff distribution parameter [-]",
    "percmax": "Maximum percolation rate [mm/d]",
    "lp": "Wilting point as fraction of smax [-]",
    "nlagf": "Flow delay before fast runoff [d]",
    "nlags": "Flow delay before slow runoff [d]",
    "kf": "Fast runoff coefficient [d-1]",
    "ks": "Slow runoff coefficient [d-1]",
    "imax": "Maximum interception storage [mm]",
}


# ==============================================================================
# 2. Static Step Functions (Compiled)
# ==============================================================================


def _flexi_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    smax: torch.Tensor,
    beta: torch.Tensor,
    d_split: torch.Tensor,
    percmax: torch.Tensor,
    lp: torch.Tensor,
    imax: torch.Tensor,
    nearzero: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Phase 1: Production Step
    Calculates fluxes BEFORE routing (S1, S2 dynamics).
    Returns: flux_rf (fast in), flux_rs_total (slow in), flux_ea, S1_new, S2_new
    """
    # --- 1. Interception Process (S1) ---
    flux_peff = interception_1(P, S1, imax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_peff)
    flux_peff = torch.clamp(flux_peff, min=zeros, max=P)

    S1_tmp = S1 + P - flux_peff
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    flux_ei = evap_1(S1_tmp, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S1_tmp - nearzero)
    flux_ei = F.relu(flux_ei)

    S1_new = torch.clamp(S1_tmp - flux_ei, min=nearzero)

    # --- 2. Soil Moisture Process (S2) ---
    flux_ru = saturation_3(S2, smax, beta, flux_peff, nearzero=nearzero)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=flux_peff)

    rem_peff = F.relu(flux_peff - flux_ru)

    # Split excess -> Fast vs Slow
    flux_rf = split_1(1.0 - d_split, rem_peff, nearzero=nearzero)
    flux_rs = F.relu(rem_peff - flux_rf)

    S2_tmp = torch.clamp(S2 + flux_ru, min=nearzero)

    PET_rem = F.relu(PET - flux_ei)
    flux_eur = evap_3(lp, S2_tmp, smax, PET_rem, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S2_tmp - nearzero)
    flux_eur = F.relu(flux_eur)

    S2_tmp2 = torch.clamp(S2_tmp - flux_eur, min=nearzero)

    flux_ps = percolation_2(percmax, S2_tmp2, smax, nearzero=nearzero)
    flux_ps = torch.minimum(flux_ps, S2_tmp2 - nearzero)
    flux_ps = F.relu(flux_ps)

    S2_new = torch.clamp(S2_tmp2 - flux_ps, min=nearzero)

    # Total flux entering slow reservoir = percolation + slow split
    flux_rs_total = flux_ps + flux_rs

    # Total Evaporation
    flux_ea = flux_ei + flux_eur

    return flux_rf, flux_rs_total, flux_ea, S1_new, S2_new


def _flexi_routing_step_impl(
    flux_rfl: torch.Tensor,  # Convolved Fast Flow
    flux_rsl: torch.Tensor,  # Convolved Slow Flow
    S3: torch.Tensor,
    S4: torch.Tensor,
    kf: torch.Tensor,
    ks: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 3: Routing Step
    Linear Reservoir Routing (S3, S4 dynamics).
    """
    # --- S3: Fast Routing Store ---
    S3_tmp = torch.clamp(S3 + flux_rfl, min=nearzero)

    flux_qf = baseflow_1(kf, S3_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S3_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S3_new = torch.clamp(S3_tmp - flux_qf, min=nearzero)

    # --- S4: Slow Routing Store ---
    S4_tmp = torch.clamp(S4 + flux_rsl, min=nearzero)

    flux_qs = baseflow_1(ks, S4_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S4_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    S4_new = torch.clamp(S4_tmp - flux_qs, min=nearzero)

    # Total Streamflow
    Qsim = flux_qf + flux_qs

    return Qsim, S3_new, S4_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (FlexiModel)
# ==============================================================================


class Flexi(UnifyV1):
    """
    Flex-I Hydrological Model

    Architecture: Sandwich (Loop -> Conv -> Loop)
    Optimization: Single-step compilation + Python unbind loop
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "flexi")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrographs (DplTri3)
        # Using bounds to define max lag size
        self.uh_fast = DplTri3(max_lag=int(FLEXI_PARAMS_BOUNDS["nlagf"][1]))
        self.uh_slow = DplTri3(max_lag=int(FLEXI_PARAMS_BOUNDS["nlags"][1]))
        self.production_step = _maybe_compile(_flexi_production_step_impl, self.backend)
        self.routing_step = _maybe_compile(_flexi_routing_step_impl, self.backend)

    def _init_states(self, n_grid: int, nmul: int = None) -> Tuple[torch.Tensor, ...]:
        """
        Initialize 4 states: S1(Interception), S2(Soil), S3(Fast), S4(Slow)
        """
        nmul = nmul or self.nmul
        S1 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S2 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S3 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S4 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        return (S1, S2, S3, S4)

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
        # Unbind forcing for faster iteration
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        # T_seq unused in step function
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        smax = static_params["smax"]
        beta = static_params["beta"]
        d_split = static_params["d_split"]
        percmax = static_params["percmax"]
        lp = static_params["lp"]
        nlagf = static_params["nlagf"]
        nlags = static_params["nlags"]
        kf = static_params["kf"]
        ks = static_params["ks"]
        imax = static_params["imax"]

        # Unpack States
        S1, S2, S3, S4 = states

        # ==========================================================
        # Phase 1: Production Loop (Python Loop + Compiled Step)
        # ==========================================================
        raw_fast_list = []
        raw_slow_list = []
        # ea_list = [] # Uncomment if evaporation output is needed

        for t in range(n_steps):
            flux_rf, flux_rs_total, flux_ea, S1, S2 = self.production_step(
                P_seq[t],
                PET_seq[t],
                S1,
                S2,
                smax,
                beta,
                d_split,
                percmax,
                lp,
                imax,
                nearzero,
            )
            raw_fast_list.append(flux_rf)
            raw_slow_list.append(flux_rs_total)

        # Stack outputs: (T, B, M)
        fast_in_stack = torch.stack(raw_fast_list, dim=0)
        slow_in_stack = torch.stack(raw_slow_list, dim=0)

        # ==========================================================
        # Phase 2: Parallel Convolution (Sandwich Middle)
        # ==========================================================
        # 1. Flatten for Conv1d: (T, B, M) -> (B*M, T)
        B_total = n_grid * nmul
        fast_in_flat = fast_in_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        slow_in_flat = slow_in_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        # 2. Prepare UH Params: (B*M, 1)
        nlagf_flat = nlagf.reshape(B_total, 1)
        nlags_flat = nlags.reshape(B_total, 1)

        # 3. Apply Convolution (PyTorch Native)

        routed_fast_flat = self.uh_fast(fast_in_flat, nlagf_flat)
        routed_slow_flat = self.uh_slow(slow_in_flat, nlags_flat)

        # 4. Reshape back and Unbind for Routing Loop
        # (B*M, T) -> (B, M, T) -> (T, B, M) -> List[Tensor]
        rfl_seq = (
            routed_fast_flat.view(n_grid, nmul, n_steps)
            .permute(2, 0, 1)
            .unbind(0)
        )
        rsl_seq = (
            routed_slow_flat.view(n_grid, nmul, n_steps)
            .permute(2, 0, 1)
            .unbind(0)
        )

        # ==========================================================
        # Phase 3: Routing Loop (Python Loop + Compiled Step)
        # ==========================================================
        Qsim_list = []

        for t in range(n_steps):
            Qsim, S3, S4 = self.routing_step(
                rfl_seq[t], rsl_seq[t], S3, S4, kf, ks, nearzero
            )
            Qsim_list.append(Qsim)

        Qsim_out = torch.stack(Qsim_list, dim=0)

        warm_up = min(self.warm_up, n_steps)
        return {"streamflow": Qsim_out[warm_up:].flatten(start_dim=1)}
