import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.interception import interception_2
from dmg.models.phy_models.flux.evap import evap_1
from dmg.models.phy_models.flux.saturation import saturation_2
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.capillary import capillary_2
from dmg.models.phy_models.flux.baseflow import baseflow_1
from dmg.models.phy_models.unithydro.uh_tri_3 import DplTri3

_TH_MAX = 120


def _hillslope_production_step(P, PET, S1, S2, dw, betaw, swmax, a, c_rad, kh, nearzero):
    flux_pe = interception_2(P, dw, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pe)
    flux_qse = torch.clamp(saturation_2(S1, swmax, betaw, flux_pe, nearzero=nearzero),
                           min=torch.zeros_like(flux_pe), max=flux_pe)
    flux_qses = split_1(a, flux_qse, nearzero=nearzero)
    flux_qseg = F.relu(flux_qse - flux_qses)
    S1_tmp = torch.clamp(S1 + flux_pe - flux_qse, min=nearzero)
    S2_tmp = torch.clamp(S2 + flux_qseg, min=nearzero)
    flux_ea_soil = torch.clamp(
        torch.minimum(torch.minimum(evap_1(S1_tmp, PET, nearzero=nearzero), S1_tmp - nearzero), PET),
        min=0.0)
    S1_tmp2 = torch.clamp(S1_tmp - flux_ea_soil, min=nearzero)
    flux_c = torch.clamp(torch.minimum(capillary_2(c_rad, S2_tmp, nearzero=nearzero), S2_tmp - nearzero), min=0.0)
    S2_tmp2 = torch.clamp(S2_tmp - flux_c, min=nearzero)
    flux_qhgw = torch.clamp(torch.minimum(baseflow_1(kh, S2_tmp2, nearzero=nearzero), S2_tmp2 - nearzero), min=0.0)
    S1_new = torch.clamp(S1_tmp2 + flux_c, min=nearzero)
    S2_new = torch.clamp(S2_tmp2 - flux_qhgw, min=nearzero)
    return flux_qses, flux_qhgw, flux_ei + flux_ea_soil, S1_new, S2_new


class Hillslope(UnifyV2):
    """Hillslope (FLEX-Topo): Production -> Conv(surface) + Baseflow."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "hillslope")
        super().__init__(config, device, backend)
        self.uh_surface = DplTri3(max_lag=_TH_MAX)
        self.production_step = _maybe_compile(_hillslope_production_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return (z.clone(), z.clone())

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        dw    = params_dict["dw"]
        betaw = params_dict["betaw"]
        swmax = params_dict["swmax"]
        a     = params_dict["a"]
        th    = params_dict["th"]
        c_rad = params_dict["c_rad"]
        kh    = params_dict["kh"]

        def warmup_step(t, curr):
            S1, S2 = curr
            _, _, _, S1_new, S2_new = self.production_step(
                P_seq[t], PET_seq[t], S1, S2, dw, betaw, swmax, a, c_rad, kh, nearzero)
            return (S1_new, S2_new)

        S1, S2 = self._run_warmup(warmup_step, n_steps, states)

        qses_list, qhgw_list = [], []
        for t in range(n_steps):
            flux_qses, flux_qhgw, _, S1, S2 = self.production_step(
                P_seq[t], PET_seq[t], S1, S2, dw, betaw, swmax, a, c_rad, kh, nearzero)
            qses_list.append(flux_qses)
            qhgw_list.append(flux_qhgw)

        B = n_grid
        routed_qses = self.uh_surface(
            torch.stack(qses_list, 0).permute(1, 2, 0).reshape(B, n_steps), th.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1)

        return self._finalize_output(routed_qses + torch.stack(qhgw_list, 0), params_dict)
