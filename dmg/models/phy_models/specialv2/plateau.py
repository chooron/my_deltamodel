import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.interception import interception_2
from dmg.models.phy_models.flux.infiltration import infiltration_4
from dmg.models.phy_models.flux.evap import evap_4
from dmg.models.phy_models.flux.capillary import capillary_2
from dmg.models.phy_models.flux.saturation import saturation_1
from dmg.models.phy_models.flux.baseflow import baseflow_1
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3

_TP_MAX = 120


def _plateau_production_step(P, PET, S1, S2, fmax, dp, sumax, lp, p_coeff, c_rise, kp, nearzero):
    flux_pe = interception_2(P, dp, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pe)
    flux_pi = torch.minimum(infiltration_4(flux_pe, fmax, nearzero=nearzero), flux_pe)
    flux_pie = F.relu(flux_pe - flux_pi)
    flux_c = torch.clamp(torch.minimum(capillary_2(c_rise, S2, nearzero=nearzero), S2 - nearzero), min=0.0)
    S1_tmp = torch.clamp(S1 + flux_pi + flux_c, min=nearzero)
    flux_et = torch.clamp(
        torch.minimum(torch.minimum(evap_4(PET, p_coeff, S1_tmp, lp, sumax, nearzero=nearzero), S1_tmp - nearzero), PET),
        min=0.0)
    S1_tmp2 = torch.clamp(S1_tmp - flux_et, min=nearzero)
    inflow_s1 = flux_pi + flux_c
    flux_r = torch.clamp(saturation_1(inflow_s1, S1_tmp2, sumax, nearzero=nearzero),
                         min=torch.zeros_like(inflow_s1), max=inflow_s1)
    S1_new = torch.clamp(S1_tmp2 - flux_r, min=nearzero)
    S2_tmp = torch.clamp(S2 + flux_r - flux_c, min=nearzero)
    flux_qpgw = torch.clamp(torch.minimum(baseflow_1(kp, S2_tmp, nearzero=nearzero), S2_tmp - nearzero), min=0.0)
    S2_new = torch.clamp(S2_tmp - flux_qpgw, min=nearzero)
    return flux_pie, flux_qpgw, flux_ei + flux_et, S1_new, S2_new


class Plateau(UnifyV2):
    """Plateau (FLEX-Topo): Production -> Conv(surface) + Baseflow."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "plateau")
        super().__init__(config, device, backend)
        self.uh_surface = DplTri3(max_lag=_TP_MAX)
        self.production_step = _maybe_compile(_plateau_production_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return (z.clone(), z.clone())

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        fmax    = params_dict["fmax"]
        dp      = params_dict["dp"]
        sumax   = params_dict["sumax"]
        lp      = params_dict["lp"]
        p_coeff = params_dict["p_coeff"]
        tp      = params_dict["tp"]
        c_rise  = params_dict["c_rise"]
        kp      = params_dict["kp"]

        def warmup_step(t, curr):
            S1, S2 = curr
            _, _, _, S1_new, S2_new = self.production_step(
                P_seq[t], PET_seq[t], S1, S2,
                fmax, dp, sumax, lp, p_coeff, c_rise, kp, nearzero)
            return (S1_new, S2_new)

        S1, S2 = self._run_warmup(warmup_step, n_steps, states)

        pie_list, qpgw_list = [], []
        for t in range(n_steps):
            flux_pie, flux_qpgw, _, S1, S2 = self.production_step(
                P_seq[t], PET_seq[t], S1, S2,
                fmax, dp, sumax, lp, p_coeff, c_rise, kp, nearzero)
            pie_list.append(flux_pie)
            qpgw_list.append(flux_qpgw)

        B = n_grid
        routed_pie = self.uh_surface(
            torch.stack(pie_list, 0).permute(1, 2, 0).reshape(B, n_steps), tp.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1)

        return self._finalize_output(routed_pie + torch.stack(qpgw_list, 0), params_dict)
