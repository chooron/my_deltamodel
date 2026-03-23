import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

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
    # interception_1: peff = P * (1 - sf), where sf = 1/(1+exp((S-Smax+r*e*Smax)/(r*Smax)))
    # sf -> 1 when S << Smax (store empty) -> peff -> 0 (all intercepted)
    # sf -> 0 when S >> Smax (store full)  -> peff -> P (all passes through)
    # Using MARRMoT defaults: r=0.01, e=5.0
    _r, _e = 0.01, 5.0
    _imax_safe = torch.abs(imax) + nearzero
    _sf = torch.sigmoid((S1 - imax + _r * _e * _imax_safe) / (_r * _imax_safe))
    flux_peff = torch.clamp(P * (1.0 - _sf), min=torch.zeros_like(P), max=P)

    # Sequential update: S1 receives P, loses peff
    S1_tmp = torch.clamp(S1 + P - flux_peff, min=nearzero)

    # evap_1: min(S1, PET)  [dt=1]
    flux_ei = torch.minimum(S1_tmp, PET)
    flux_ei = torch.minimum(flux_ei, S1_tmp - nearzero)
    flux_ei = F.relu(flux_ei)

    S1_new = torch.clamp(S1_tmp - flux_ei, min=nearzero)

    # --- 2. Soil Moisture Process (S2) ---
    # saturation_3: (1 - 1/(1+exp((S/Smax + 0.5)/beta))) * peff  [MARRMoT sigmoid form]
    _ratio = S2 / (smax + nearzero)
    _sat_frac = 1.0 - 1.0 / (1.0 + torch.exp((_ratio + 0.5) / (beta + nearzero)))
    flux_ru = torch.clamp(_sat_frac * flux_peff, min=torch.zeros_like(flux_peff), max=flux_peff)

    rem_peff = F.relu(flux_peff - flux_ru)

    # split_1: fast = (1-d)*excess, slow = d*excess
    flux_rf = (1.0 - d_split) * rem_peff
    flux_rs = d_split * rem_peff

    S2_tmp = torch.clamp(S2 + flux_ru, min=nearzero)

    # evap_3: min(S2/(lp*smax)*PET, PET, S2)  [dt=1, uses full PET per MARRMoT]
    flux_eur = torch.minimum(
        torch.minimum(S2_tmp / (lp * smax + nearzero) * PET, PET),
        S2_tmp,
    )
    flux_eur = torch.minimum(flux_eur, S2_tmp - nearzero)
    flux_eur = F.relu(flux_eur)

    S2_tmp2 = torch.clamp(S2_tmp - flux_eur, min=nearzero)

    # percolation_2: min(S2, percmax*S2/smax)  [dt=1]
    flux_ps = torch.minimum(S2_tmp2, percmax * S2_tmp2 / (smax + nearzero))
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

    # baseflow_1: kf * S3  [linear reservoir, rate param d-1]
    flux_qf = torch.minimum(kf * S3_tmp, S3_tmp - nearzero)
    flux_qf = F.relu(flux_qf)

    S3_new = torch.clamp(S3_tmp - flux_qf, min=nearzero)

    # --- S4: Slow Routing Store ---
    S4_tmp = torch.clamp(S4 + flux_rsl, min=nearzero)

    # baseflow_1: ks * S4  [linear reservoir, rate param d-1]
    flux_qs = torch.minimum(ks * S4_tmp, S4_tmp - nearzero)
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

    def _init_states(self, n_grid: int, nmul: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
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
        nmul: Optional[int] = None,
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
        warm_up = min(self.warm_up, n_steps)
        B_total = n_grid * nmul
        nlagf_flat = nlagf.reshape(B_total, 1)
        nlags_flat = nlags.reshape(B_total, 1)

        # ==========================================================
        # Phase 1: Production Loop (warm_up + training, full sequence)
        # ==========================================================
        all_fast_list = []
        all_slow_list = []

        with torch.no_grad():
            for t in range(warm_up):
                flux_rf_wu, flux_rs_wu, _, S1, S2 = self.production_step(
                    P_seq[t], PET_seq[t], S1, S2,
                    smax, beta, d_split, percmax, lp, imax, nearzero)
                all_fast_list.append(flux_rf_wu)
                all_slow_list.append(flux_rs_wu)

        S1, S2 = S1.detach(), S2.detach()

        for t in range(n_steps):
            flux_rf, flux_rs_total, _, S1, S2 = self.production_step(
                P_seq[t], PET_seq[t], S1, S2,
                smax, beta, d_split, percmax, lp, imax, nearzero)
            all_fast_list.append(flux_rf)
            all_slow_list.append(flux_rs_total)

        T_full = warm_up + n_steps
        fast_stack = torch.stack(all_fast_list, dim=0)
        slow_stack = torch.stack(all_slow_list, dim=0)

        # ==========================================================
        # Phase 2: Parallel Convolution over full sequence
        # UH sees continuous flux history — no carry-over loss
        # ==========================================================
        fast_flat = fast_stack.permute(1, 2, 0).reshape(B_total, T_full)
        slow_flat = slow_stack.permute(1, 2, 0).reshape(B_total, T_full)

        routed_fast_flat = self.uh_fast(fast_flat, nlagf_flat)
        routed_slow_flat = self.uh_slow(slow_flat, nlags_flat)

        # Slice off warm_up portion, keep only training steps
        rfl_seq = routed_fast_flat[:, warm_up:].view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)
        rsl_seq = routed_slow_flat[:, warm_up:].view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)

        # Warm up routing stores using warm_up portion of routed fluxes
        with torch.no_grad():
            rfl_wu = routed_fast_flat[:, :warm_up].view(n_grid, nmul, warm_up).permute(2, 0, 1).unbind(0)
            rsl_wu = routed_slow_flat[:, :warm_up].view(n_grid, nmul, warm_up).permute(2, 0, 1).unbind(0)
            for t in range(warm_up):
                _, S3, S4 = self.routing_step(rfl_wu[t], rsl_wu[t], S3, S4, kf, ks, nearzero)

        S3, S4 = S3.detach(), S4.detach()

        # ==========================================================
        # Phase 3: Routing Loop — training steps only
        # ==========================================================
        Qsim_list = []
        for t in range(n_steps):
            Qsim, S3, S4 = self.routing_step(
                rfl_seq[t], rsl_seq[t], S3, S4, kf, ks, nearzero)
            Qsim_list.append(Qsim)

        Qsim_out = torch.stack(Qsim_list, dim=0)
        return {"streamflow": Qsim_out.flatten(start_dim=1)}
