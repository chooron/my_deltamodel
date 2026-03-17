import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.snowfall import snowfall_1
from dmg.models.phy_models.flux.rainfall import rainfall_1
from dmg.models.phy_models.flux.melt import melt_1
from dmg.models.phy_models.flux.interception import interception_1
from dmg.models.phy_models.flux.evap import evap_1, evap_3
from dmg.models.phy_models.flux.saturation import saturation_3
from dmg.models.phy_models.flux.percolation import percolation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.baseflow import baseflow_1
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3

_NLAGF_MAX = 5
_NLAGS_MAX = 15


def _flexis_production_step(P, T, PET, S1, S2, S3, smax, beta, d_split, percmax, lp, imax, tt, ddf, nearzero):
    flux_ps = snowfall_1(P, T, tt, nearzero=nearzero)
    flux_pi = rainfall_1(P, T, tt, nearzero=nearzero)
    flux_m = torch.clamp(torch.minimum(melt_1(ddf, tt, T, S1, nearzero=nearzero), S1 - nearzero), min=0.0)
    S1_new = torch.clamp(S1 + flux_ps - flux_m, min=nearzero)
    inflow_S2 = flux_m + flux_pi
    flux_peff = torch.clamp(interception_1(inflow_S2, S2, imax, nearzero=nearzero),
                            min=torch.zeros_like(inflow_S2), max=inflow_S2)
    S2_tmp = torch.clamp(S2 + inflow_S2 - flux_peff, min=nearzero)
    flux_ei = torch.clamp(torch.minimum(evap_1(S2_tmp, PET, nearzero=nearzero), S2_tmp - nearzero), min=0.0)
    S2_new = torch.clamp(S2_tmp - flux_ei, min=nearzero)
    flux_ru = torch.clamp(saturation_3(S3, smax, beta, flux_peff, nearzero=nearzero),
                          min=torch.zeros_like(flux_peff), max=flux_peff)
    rem = F.relu(flux_peff - flux_ru)
    flux_rf = split_1(1.0 - d_split, rem, nearzero=nearzero)
    flux_rs = F.relu(rem - flux_rf)
    S3_tmp = torch.clamp(S3 + flux_ru, min=nearzero)
    PET_rem = F.relu(PET - flux_ei)
    flux_eur = torch.clamp(
        torch.minimum(torch.minimum(evap_3(lp, S3_tmp, smax, PET_rem, nearzero=nearzero), S3_tmp - nearzero), PET_rem),
        min=0.0)
    S3_tmp2 = torch.clamp(S3_tmp - flux_eur, min=nearzero)
    flux_rp = torch.clamp(torch.minimum(percolation_2(percmax, S3_tmp2, smax, nearzero=nearzero), S3_tmp2 - nearzero), min=0.0)
    S3_new = torch.clamp(S3_tmp2 - flux_rp, min=nearzero)
    return flux_rf, flux_rs + flux_rp, flux_ei + flux_eur, S1_new, S2_new, S3_new


def _flexis_routing_step(rfl, rsl, S4, S5, kf, ks, nearzero):
    S4_tmp = torch.clamp(S4 + rfl, min=nearzero)
    qf = torch.clamp(torch.minimum(baseflow_1(kf, S4_tmp, nearzero=nearzero), S4_tmp - nearzero), min=0.0)
    S4_new = torch.clamp(S4_tmp - qf, min=nearzero)
    S5_tmp = torch.clamp(S5 + rsl, min=nearzero)
    qs = torch.clamp(torch.minimum(baseflow_1(ks, S5_tmp, nearzero=nearzero), S5_tmp - nearzero), min=0.0)
    S5_new = torch.clamp(S5_tmp - qs, min=nearzero)
    return qf + qs, S4_new, S5_new


class Flexis(UnifyV2):
    """Flex-IS: Snow + Interception + Soil + Conv + Routing."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "flexis")
        super().__init__(config, device, backend)
        self.uh_fast = DplTri3(max_lag=_NLAGF_MAX)
        self.uh_slow = DplTri3(max_lag=_NLAGS_MAX)
        self.production_step = _maybe_compile(_flexis_production_step, self.backend)
        self.routing_step = _maybe_compile(_flexis_routing_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return tuple(z.clone() for _ in range(5))

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        T_seq   = forcing[..., 1:2].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        smax    = params_dict["smax"]
        beta    = params_dict["beta"]
        d_split = params_dict["d_split"]
        percmax = params_dict["percmax"]
        lp      = params_dict["lp"]
        nlagf   = params_dict["nlagf"]
        nlags   = params_dict["nlags"]
        kf      = params_dict["kf"]
        ks      = params_dict["ks"]
        imax    = params_dict["imax"]
        tt      = params_dict["tt"]
        ddf     = params_dict["ddf"]

        def warmup_step(t, curr):
            S1, S2, S3, S4, S5 = curr
            _, _, _, S1n, S2n, S3n = self.production_step(
                P_seq[t], T_seq[t], PET_seq[t], S1, S2, S3,
                smax, beta, d_split, percmax, lp, imax, tt, ddf, nearzero)
            return (S1n, S2n, S3n, S4, S5)

        S1, S2, S3, S4, S5 = self._run_warmup(warmup_step, n_steps, states)

        fast_list, slow_list = [], []
        for t in range(n_steps):
            flux_rf, flux_rs, _, S1, S2, S3 = self.production_step(
                P_seq[t], T_seq[t], PET_seq[t], S1, S2, S3,
                smax, beta, d_split, percmax, lp, imax, tt, ddf, nearzero)
            fast_list.append(flux_rf)
            slow_list.append(flux_rs)

        B = n_grid
        rfl_seq = self.uh_fast(
            torch.stack(fast_list, 0).permute(1, 2, 0).reshape(B, n_steps), nlagf.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1).unbind(0)
        rsl_seq = self.uh_slow(
            torch.stack(slow_list, 0).permute(1, 2, 0).reshape(B, n_steps), nlags.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1).unbind(0)

        q_list = []
        for t in range(n_steps):
            Qsim, S4, S5 = self.routing_step(rfl_seq[t], rsl_seq[t], S4, S5, kf, ks, nearzero)
            q_list.append(Qsim)

        return self._finalize_output(torch.stack(q_list, dim=0), params_dict)
