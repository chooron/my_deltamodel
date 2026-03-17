import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.evap import evap_1, evap_6, evap_5
from dmg.models.phy_models.flux.interception import interception_1
from dmg.models.phy_models.flux.saturation import saturation_1
from dmg.models.phy_models.flux.interflow import interflow_9
from dmg.models.phy_models.flux.baseflow import baseflow_1
from dmg.models.phy_models.unithydro.uh_tri_4 import DplTri4

_D_DELAY_MAX = 30


def _newzealand2_production_step(P, PET, S1, S2, s1max, s2max, sfc_frac, m, a, b, tcbf, nearzero):
    flux_qtf = torch.clamp(interception_1(P, S1, s1max, nearzero=nearzero),
                           min=torch.zeros_like(P), max=P)
    S1_tmp = torch.clamp(S1 + P - flux_qtf, min=nearzero)
    flux_eint = torch.clamp(
        torch.minimum(torch.minimum(evap_1(S1_tmp, PET, nearzero=nearzero), S1_tmp - nearzero), PET),
        min=0.0)
    S1_new = torch.clamp(S1_tmp - flux_eint, min=nearzero)
    flux_qse = torch.clamp(saturation_1(flux_qtf, S2, s2max, nearzero=nearzero),
                           min=torch.zeros_like(flux_qtf), max=flux_qtf)
    S2_tmp = torch.clamp(S2 + flux_qtf - flux_qse, min=nearzero)
    pet_rem = F.relu(PET - flux_eint)
    flux_ea_s2 = torch.clamp(
        torch.minimum(torch.minimum(
            evap_6(m, sfc_frac, S2_tmp, s2max, pet_rem, nearzero=nearzero) +
            evap_5(m, S2_tmp, s2max, pet_rem, nearzero=nearzero),
            S2_tmp - nearzero), pet_rem),
        min=0.0)
    S2_tmp2 = torch.clamp(S2_tmp - flux_ea_s2, min=nearzero)
    flux_qss = torch.clamp(
        torch.minimum(interflow_9(S2_tmp2, a, sfc_frac * s2max, b, nearzero=nearzero), S2_tmp2 - nearzero),
        min=0.0)
    S2_tmp3 = torch.clamp(S2_tmp2 - flux_qss, min=nearzero)
    flux_qbf = torch.clamp(
        torch.minimum(baseflow_1(tcbf, S2_tmp3, nearzero=nearzero), S2_tmp3 - nearzero),
        min=0.0)
    S2_new = torch.clamp(S2_tmp3 - flux_qbf, min=nearzero)
    return flux_qse + flux_qss + flux_qbf, flux_eint + flux_ea_s2, S1_new, S2_new


class Newzealand2(UnifyV2):
    """New Zealand Model v2 (MARRMoT m_16): Production -> Full Triangle Conv."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "newzealand2")
        super().__init__(config, device, backend)
        self.uh = DplTri4(max_lag=_D_DELAY_MAX)
        self.production_step = _maybe_compile(_newzealand2_production_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return (z.clone(), z.clone())

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        s1max    = params_dict["s1max"]
        s2max    = params_dict["s2max"]
        sfc_frac = params_dict["sfc_frac"]
        m        = params_dict["m"]
        a        = params_dict["a"]
        b        = params_dict["b"]
        tcbf     = params_dict["tcbf"]
        d_delay  = params_dict["d_delay"]

        def warmup_step(t, curr):
            S1, S2 = curr
            _, _, S1_new, S2_new = self.production_step(
                P_seq[t], PET_seq[t], S1, S2,
                s1max, s2max, sfc_frac, m, a, b, tcbf, nearzero)
            return (S1_new, S2_new)

        S1, S2 = self._run_warmup(warmup_step, n_steps, states)

        q_list = []
        for t in range(n_steps):
            flux_q, _, S1, S2 = self.production_step(
                P_seq[t], PET_seq[t], S1, S2,
                s1max, s2max, sfc_frac, m, a, b, tcbf, nearzero)
            q_list.append(flux_q)

        B = n_grid
        Qsim_out = self.uh(
            torch.stack(q_list, 0).permute(1, 2, 0).reshape(B, n_steps), d_delay.reshape(B, 1)
        ).view(n_grid, 1, n_steps).permute(2, 0, 1)
        return self._finalize_output(Qsim_out, params_dict)
