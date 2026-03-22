import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1

from dmg.models.phy_models.flux.evap import evap_12
from dmg.models.phy_models.flux.saturation import saturation_5
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.unithydro.uh_exp_5 import DplExp5
from dmg.models.phy_models.unithydro.uh_delay_8 import DplDelay8


IHACRES_PARAMS_BOUNDS = {
    "lp": [1.0, 2000.0],
    "d": [1.0, 2000.0],
    "p": [0.0, 10.0],
    "alpha": [0.0, 1.0],
    "tau_q": [1.0, 100.0],
    "tau_s": [1.0, 300.0],
    "tau_d": [0.0, 30.0],
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


def _ihacres_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,
    lp: torch.Tensor,
    d: torch.Tensor,
    p: torch.Tensor,
    alpha: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sequential-explicit IHACRES production step.

    This keeps the MARRMoT flux semantics but uses a discrete update that is
    state-safe in PyTorch:
    1. Compute effective rainfall from the current deficit.
    2. Use the remaining rainfall to fill the deficit store, with overflow
       routed back to streamflow to preserve mass.
    3. Apply evaporation on the post-rainfall deficit state.
    """
    P = torch.clamp(P, min=0.0)
    PET = torch.clamp(PET, min=0.0)
    zeros = torch.zeros_like(P)
    S1 = F.relu(S1)

    flux_u_pot = saturation_5(S1, d, p, P, nearzero=nearzero)
    flux_u_pot = torch.clamp(flux_u_pot, min=zeros, max=P)

    rain_to_store = P - flux_u_pot
    rain_fill = torch.minimum(rain_to_store, S1)
    flux_overflow = rain_to_store - rain_fill
    S1_wet = S1 - rain_fill

    flux_ea = evap_12(S1_wet, lp, PET, nearzero=nearzero)
    flux_ea = torch.clamp(flux_ea, min=zeros, max=PET)

    flux_u_total = flux_u_pot + flux_overflow
    flux_uq = split_1(alpha, flux_u_total, nearzero=nearzero)
    flux_us = flux_u_total - flux_uq
    S1_new = S1_wet + flux_ea

    return flux_uq, flux_us, flux_ea, S1_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


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

        self.uh_fast = DplExp5(max_lag=int(IHACRES_PARAMS_BOUNDS["tau_q"][1]))
        self.uh_slow = DplExp5(max_lag=int(IHACRES_PARAMS_BOUNDS["tau_s"][1]))
        self.uh_delay = DplDelay8(max_lag=int(IHACRES_PARAMS_BOUNDS["tau_d"][1]) + 1)
        self.production_step = _maybe_compile(_ihacres_production_step_impl, self.backend)

    def _init_states(self, n_grid: int, nmul: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        nmul = nmul or self.nmul
        S1 = torch.zeros((n_grid, nmul), device=self.device)
        return (S1,)

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

        if n_steps == 0:
            return {"streamflow": forcing.new_empty((0, n_grid * nmul))}

        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        lp = static_params["lp"]
        d = static_params["d"]
        p = static_params["p"]
        alpha = static_params["alpha"]
        tau_q = static_params["tau_q"]
        tau_s = static_params["tau_s"]
        tau_d = static_params["tau_d"]

        (S1,) = states
        warm_up = min(self.warm_up, n_steps)
        raw_uq_list = []
        raw_us_list = []

        with torch.no_grad():
            for t in range(warm_up):
                flux_uq, flux_us, _, S1 = self.production_step(
                    P_seq[t], PET_seq[t], S1, lp, d, p, alpha, nearzero
                )
                raw_uq_list.append(flux_uq)
                raw_us_list.append(flux_us)
        S1 = S1.detach()

        for t in range(warm_up, n_steps):
            flux_uq, flux_us, _, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1, lp, d, p, alpha, nearzero
            )
            raw_uq_list.append(flux_uq)
            raw_us_list.append(flux_us)

        uq_stack = torch.stack(raw_uq_list, dim=0)
        us_stack = torch.stack(raw_us_list, dim=0)

        B_total = n_grid * nmul
        uq_flat = uq_stack.permute(1, 2, 0).reshape(B_total, n_steps)
        us_flat = us_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        tau_q_flat = tau_q.reshape(B_total, 1)
        tau_s_flat = tau_s.reshape(B_total, 1)
        routed_uq_flat = self.uh_fast(uq_flat, tau_q_flat)
        routed_us_flat = self.uh_slow(us_flat, tau_s_flat)

        q_sum_flat = routed_uq_flat + routed_us_flat
        tau_d_flat = tau_d.reshape(B_total, 1)
        routed_total_flat = self.uh_delay(q_sum_flat, tau_d_flat)

        Qsim_out = routed_total_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1)
        return {"streamflow": Qsim_out[warm_up:].flatten(start_dim=1)}
