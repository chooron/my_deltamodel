import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

# 引入通量计算函数
from dmg.models.phy_models.flux.snowfall import snowfall_1
from dmg.models.phy_models.flux.rainfall import rainfall_1
from dmg.models.phy_models.flux.melt import melt_1
from dmg.models.phy_models.flux.evap import evap_1, evap_3
from dmg.models.phy_models.flux.saturation import saturation_3
from dmg.models.phy_models.flux.percolation import percolation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.baseflow import baseflow_1

# 引入单位线 (使用三角形单位线，与 Flex-I 保持一致)
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3


# ==============================================================================
# 1. Parameter Definitions
# ==============================================================================
FLEXIS_PARAMS_BOUNDS = {
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
    "tt": [-3.0, 5.0],  # Threshold temperature [oC]
    "ddf": [0.0, 20.0],  # Degree-day factor [mm/d/oC]
}

FLEXIS_PARAMS_DESC = {
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
    "tt": "Threshold temperature for snowfall/snowmelt [oC]",
    "ddf": "Degree-day factor for snowmelt [mm/d/oC]",
}


# ==============================================================================
# 2. Static Step Functions (Compiled)
# ==============================================================================


def _flexis_production_step_impl(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,  # Snow
    S2: torch.Tensor,  # Interception
    S3: torch.Tensor,  # Soil
    smax: torch.Tensor,
    beta: torch.Tensor,
    d_split: torch.Tensor,
    percmax: torch.Tensor,
    lp: torch.Tensor,
    imax: torch.Tensor,
    tt: torch.Tensor,
    ddf: torch.Tensor,
    nearzero: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Phase 1: Production Step (Snow -> Interception -> Soil)
    Calculates fluxes BEFORE routing.
    Returns: flux_rf, flux_rs_total, flux_ea, S1_new, S2_new, S3_new
    """
    # --- 1. Snow Process (S1) ---
    flux_ps = snowfall_1(P, T, tt, nearzero=nearzero)
    flux_pi = rainfall_1(P, T, tt, nearzero=nearzero)

    flux_m = melt_1(ddf, tt, T, S1, nearzero=nearzero)
    flux_m = torch.minimum(flux_m, S1 - nearzero)
    flux_m = F.relu(flux_m)

    S1_new = torch.clamp(S1 + flux_ps - flux_m, min=nearzero)

    # --- 2. Interception Process (S2) ---
    # Inflow to S2 is melt + rainfall
    inflow_S2 = flux_m + flux_pi

    # MATLAB: interception_1 uses smoothThreshold_storage_logistic(S2, imax, r=0.01, e=5.0)
    # = sigmoid((S2 - imax*(1-r)) / (r*e*imax)) = sigmoid(20*(S2/imax - 0.99))
    _r, _e = 0.01, 5.0
    _imax_safe = torch.abs(imax) + nearzero
    _sf_ic = torch.sigmoid((S2 - imax + _r * _imax_safe) / (_r * _e * _imax_safe))
    zeros = torch.zeros_like(inflow_S2)
    flux_peff = torch.clamp(inflow_S2 * (1.0 - _sf_ic), min=zeros, max=inflow_S2)

    S2_tmp = S2 + inflow_S2 - flux_peff
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    flux_ei = evap_1(S2_tmp, PET, nearzero=nearzero)
    flux_ei = torch.minimum(flux_ei, S2_tmp - nearzero)
    flux_ei = F.relu(flux_ei)

    S2_new = torch.clamp(S2_tmp - flux_ei, min=nearzero)

    # --- 3. Soil Moisture Process (S3) ---
    flux_ru = saturation_3(S3, smax, beta, flux_peff, nearzero=nearzero)
    flux_ru = torch.clamp(flux_ru, min=zeros, max=flux_peff)

    rem_peff = F.relu(flux_peff - flux_ru)

    # Split excess -> Fast vs Slow
    flux_rf = split_1(1.0 - d_split, rem_peff, nearzero=nearzero)
    flux_rs = F.relu(rem_peff - flux_rf)

    S3_tmp = torch.clamp(S3 + flux_ru, min=nearzero)

    flux_eur = evap_3(lp, S3_tmp, smax, PET, nearzero=nearzero)
    flux_eur = torch.minimum(flux_eur, S3_tmp - nearzero)
    flux_eur = F.relu(flux_eur)

    S3_tmp2 = torch.clamp(S3_tmp - flux_eur, min=nearzero)

    flux_rp = percolation_2(percmax, S3_tmp2, smax, nearzero=nearzero)
    flux_rp = torch.minimum(flux_rp, S3_tmp2 - nearzero)
    flux_rp = F.relu(flux_rp)

    S3_new = torch.clamp(S3_tmp2 - flux_rp, min=nearzero)

    # Total flux to slow reservoir
    flux_rs_total = flux_rs + flux_rp

    # Total Evaporation
    flux_ea = flux_ei + flux_eur

    return flux_rf, flux_rs_total, flux_ea, S1_new, S2_new, S3_new


def _flexis_routing_step_impl(
    flux_rfl: torch.Tensor,  # Convolved Fast Flow
    flux_rsl: torch.Tensor,  # Convolved Slow Flow
    S4: torch.Tensor,  # Fast Store
    S5: torch.Tensor,  # Slow Store
    kf: torch.Tensor,
    ks: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 3: Routing Step (Linear Reservoirs S4 & S5)
    """
    # --- S4: Fast Routing Store ---
    S4_tmp = torch.clamp(S4 + flux_rfl, min=nearzero)

    flux_qf = baseflow_1(kf, S4_tmp, nearzero=nearzero)
    flux_qf = torch.minimum(flux_qf, S4_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S4_new = torch.clamp(S4_tmp - flux_qf, min=nearzero)

    # --- S5: Slow Routing Store ---
    S5_tmp = torch.clamp(S5 + flux_rsl, min=nearzero)

    flux_qs = baseflow_1(ks, S5_tmp, nearzero=nearzero)
    flux_qs = torch.minimum(flux_qs, S5_tmp - nearzero)
    flux_qs = F.relu(flux_qs)

    S5_new = torch.clamp(S5_tmp - flux_qs, min=nearzero)

    # Total Flow
    Qsim = flux_qf + flux_qs

    return Qsim, S4_new, S5_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (FlexisModel)
# ==============================================================================


class Flexis(UnifyV1):
    """
    Flex-IS Hydrological Model (Flex-I with Snow)

    Architecture: Sandwich (Loop -> Conv -> Loop)
    States: 5 (S1:Snow, S2:Int, S3:Soil, S4:Fast, S5:Slow)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "flexis")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrographs
        # Using bounds to define max lag size
        self.uh_fast = DplTri3(max_lag=int(FLEXIS_PARAMS_BOUNDS["nlagf"][1]))
        self.uh_slow = DplTri3(max_lag=int(FLEXIS_PARAMS_BOUNDS["nlags"][1]))
        self.production_step = _maybe_compile(_flexis_production_step_impl, self.backend)
        self.routing_step = _maybe_compile(_flexis_routing_step_impl, self.backend)

    def _init_states(self, n_grid: int, nmul: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """Initialize 5 states"""
        nmul = nmul or self.nmul
        S1 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S2 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S3 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S4 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        S5 = torch.zeros((n_grid, nmul), device=self.device) + self.nearzero
        return (S1, S2, S3, S4, S5)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        nmul: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = nmul or self.nmul
        nearzero = self.nearzero

        # --- A. Data Prep ---
        # Unbind forcing for faster iteration
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
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
        tt = static_params["tt"]
        ddf = static_params["ddf"]

        # Unpack States
        S1, S2, S3, S4, S5 = states
        warm_up = min(self.warm_up, n_steps)
        B_total = n_grid * nmul
        nlagf_flat = nlagf.reshape(B_total, 1)
        nlags_flat = nlags.reshape(B_total, 1)

        # ==========================================================
        # Phase 1: Production Loop (full n_steps, all with grad)
        # ==========================================================
        raw_fast_list = []
        raw_slow_list = []
        for t in range(n_steps):
            flux_rf, flux_rs_total, _, S1, S2, S3 = self.production_step(
                P_seq[t], T_seq[t], PET_seq[t], S1, S2, S3,
                smax, beta, d_split, percmax, lp, imax, tt, ddf, nearzero)
            raw_fast_list.append(flux_rf)
            raw_slow_list.append(flux_rs_total)

        fast_stack = torch.stack(raw_fast_list, dim=0)
        slow_stack = torch.stack(raw_slow_list, dim=0)

        # ==========================================================
        # Phase 2: Parallel Convolution over full n_steps
        # ==========================================================
        fast_flat = fast_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        slow_flat = slow_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        routed_fast_flat = self.uh_fast(fast_flat, nlagf_flat)
        routed_slow_flat = self.uh_slow(slow_flat, nlags_flat)

        rfl_seq = routed_fast_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)
        rsl_seq = routed_slow_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)

        # ==========================================================
        # Phase 3: Routing Loop (full n_steps)
        # ==========================================================
        Qsim_list = []
        for t in range(n_steps):
            Qsim, S4, S5 = self.routing_step(
                rfl_seq[t], rsl_seq[t], S4, S5, kf, ks, nearzero)
            Qsim_list.append(Qsim)

        Qsim_out = torch.stack(Qsim_list, dim=0)
        return {"streamflow": Qsim_out[warm_up:].flatten(start_dim=1)}
