import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.saturation import saturation_3
from dmg.models.phy_models.flux.evap import evap_3
from dmg.models.phy_models.flux.percolation import percolation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.baseflow import baseflow_1
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3

_NLAGF_MAX = 5
_NLAGS_MAX = 15


def _production_step(P, PET, S1, s1max, beta, d_split, percmax, lp, nearzero):
    flux_ru = torch.clamp(saturation_3(S1, s1max, beta, P, nearzero=nearzero),
                          min=torch.zeros_like(P), max=P)
    p_excess = F.relu(P - flux_ru)
    flux_rf = split_1(1.0 - d_split, p_excess, nearzero=nearzero)
    flux_slow = F.relu(p_excess - flux_rf)
    S1_tmp = torch.clamp(S1 + flux_ru, min=nearzero)
    flux_eur = torch.clamp(
        torch.minimum(torch.minimum(evap_3(lp, S1_tmp, s1max, PET, nearzero=nearzero), S1_tmp - nearzero), PET),
        min=0.0)
    S1_tmp2 = torch.clamp(S1_tmp - flux_eur, min=nearzero)
    flux_ps = torch.clamp(torch.minimum(percolation_2(percmax, S1_tmp2, s1max, nearzero=nearzero), S1_tmp2 - nearzero), min=0.0)
    S1_new = torch.clamp(S1_tmp2 - flux_ps, min=nearzero)
    return flux_rf, flux_slow + flux_ps, flux_eur, S1_new


def _routing_step(rfl, rsl, S2, S3, kf, ks, nearzero):
    S2_tmp = torch.clamp(S2 + rfl, min=nearzero)
    qf = torch.clamp(torch.minimum(baseflow_1(kf, S2_tmp, nearzero=nearzero), S2_tmp - nearzero), min=0.0)
    S2_new = torch.clamp(S2_tmp - qf, min=nearzero)
    S3_tmp = torch.clamp(S3 + rsl, min=nearzero)
    qs = torch.clamp(torch.minimum(baseflow_1(ks, S3_tmp, nearzero=nearzero), S3_tmp - nearzero), min=0.0)
    S3_new = torch.clamp(S3_tmp - qs, min=nearzero)
    return qf + qs, S2_new, S3_new


class Flexb(UnifyV2):
    """Flex-B (MARRMoT m_21): Production -> Conv -> Routing."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "flexb")
        super().__init__(config, device, backend)
        self.uh_fast = DplTri3(max_lag=_NLAGF_MAX)
        self.uh_slow = DplTri3(max_lag=_NLAGS_MAX)
        self.production_step = _maybe_compile(_production_step, self.backend)
        self.routing_step = _maybe_compile(_routing_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return (z.clone(), z.clone(), z.clone())

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        s1max   = params_dict["s1max"]
        beta    = params_dict["beta"]
        d_split = params_dict["d_split"]
        percmax = params_dict["percmax"]
        lp      = params_dict["lp"]
        nlagf   = params_dict["nlagf"]
        nlags   = params_dict["nlags"]
        kf      = params_dict["kf"]
        ks      = params_dict["ks"]

        def warmup_step(t, curr):
            S1, S2, S3 = curr
            _, _, _, S1_new = self.production_step(
                P_seq[t], PET_seq[t], S1, s1max, beta, d_split, percmax, lp, nearzero)
            return (S1_new, S2, S3)

        S1, S2, S3 = self._run_warmup(warmup_step, n_steps, states)

        rf_list, slow_list = [], []
        for t in range(n_steps):
            flux_rf, flux_slow, _, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1, s1max, beta, d_split, percmax, lp, nearzero)
            rf_list.append(flux_rf)
            slow_list.append(flux_slow)

        B = n_grid
        rfl_seq = self.uh_fast(
            torch.stack(rf_list, 0).permute(1, 2, 0).reshape(B, n_steps), nlagf.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1).unbind(0)
        rsl_seq = self.uh_slow(
            torch.stack(slow_list, 0).permute(1, 2, 0).reshape(B, n_steps), nlags.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1).unbind(0)

        q_list = []
        for t in range(n_steps):
            Qsim, S2, S3 = self.routing_step(rfl_seq[t], rsl_seq[t], S2, S3, kf, ks, nearzero)
            q_list.append(Qsim)

        return self._finalize_output(torch.stack(q_list, dim=0), params_dict)
