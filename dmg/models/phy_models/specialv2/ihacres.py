import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.saturation import saturation_5
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.unithydro.uh_exp_5 import DplExp5
from dmg.models.phy_models.unithydro.uh_delay_8 import DplDelay8

_TAU_Q_MAX = 5
_TAU_S_MAX = 30
_TAU_D_MAX = 10


def _evap_linear_deficit(S, lp, Ep, nearzero=1e-6):
    return torch.clamp(1.0 - S / (lp + nearzero), min=0.0, max=1.0) * Ep


def _ihacres_production_step(P, PET, S1, lp, d, p, alpha, nearzero):
    flux_ea = F.relu(_evap_linear_deficit(S1, lp, PET, nearzero))
    flux_u_calc = torch.clamp(saturation_5(S1, d, p, P, nearzero=nearzero),
                              min=torch.zeros_like(P), max=P)
    S1_temp = S1 - P + flux_ea + flux_u_calc
    flux_u_total = flux_u_calc + F.relu(-S1_temp)
    S1_new = torch.clamp(S1_temp, min=nearzero)
    flux_uq = split_1(alpha, flux_u_total, nearzero=nearzero)
    flux_us = split_1(1.0 - alpha, flux_u_total, nearzero=nearzero)
    return flux_uq, flux_us, flux_ea, S1_new


class Ihacres(UnifyV2):
    """IHACRES: Production -> Parallel Exp Conv -> Sum -> Delay Conv."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "ihacres")
        super().__init__(config, device, backend)
        self.uh_fast  = DplExp5(max_lag=_TAU_Q_MAX)
        self.uh_slow  = DplExp5(max_lag=_TAU_S_MAX)
        self.uh_delay = DplDelay8(max_lag=_TAU_D_MAX)
        self.production_step = _maybe_compile(_ihacres_production_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        return (torch.zeros((n_grid, 1), device=self.device) + self.nearzero,)

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        lp    = params_dict["lp"]
        d     = params_dict["d"]
        p     = params_dict["p"]
        alpha = params_dict["alpha"]
        tau_q = params_dict["tau_q"]
        tau_s = params_dict["tau_s"]
        tau_d = params_dict["tau_d"]

        def warmup_step(t, curr):
            (S1,) = curr
            _, _, _, S1_new = self.production_step(
                P_seq[t], PET_seq[t], S1, lp, d, p, alpha, nearzero)
            return (S1_new,)

        (S1,) = self._run_warmup(warmup_step, n_steps, states)

        uq_list, us_list = [], []
        for t in range(n_steps):
            flux_uq, flux_us, _, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1, lp, d, p, alpha, nearzero)
            uq_list.append(flux_uq)
            us_list.append(flux_us)

        B = n_grid
        uq_flat = torch.stack(uq_list, 0).permute(1, 2, 0).reshape(B, n_steps)
        us_flat = torch.stack(us_list, 0).permute(1, 2, 0).reshape(B, n_steps)

        q_sum_flat = (
            self.uh_fast(uq_flat, tau_q.reshape(B, 1)) +
            self.uh_slow(us_flat, tau_s.reshape(B, 1))
        )
        routed = self.uh_delay(q_sum_flat, tau_d.reshape(B, 1))
        Qsim_out = routed.view(n_grid, 1, n_steps).permute(2, 0, 1)
        return self._finalize_output(Qsim_out, params_dict)
