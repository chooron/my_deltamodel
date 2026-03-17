import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.flux.effective import effective_1
from dmg.models.phy_models.flux.saturation import saturation_1, saturation_6
from dmg.models.phy_models.flux.infiltration import infiltration_4
from dmg.models.phy_models.flux.evap import evap_13, evap_14
from dmg.models.phy_models.flux.split import split_1
from dmg.models.phy_models.flux.baseflow import baseflow_1
from dmg.models.phy_models.unithydro.base import DplUHBase

_NK_MAX = 120


class DplGamma6(DplUHBase):
    """Gamma (Nash Cascade) Unit Hydrograph."""

    def get_weights(self, params):
        n = params[:, 0:1].unsqueeze(-1)
        k = params[:, 1:2].unsqueeze(-1)
        alpha = F.relu(n) + 0.1
        theta = F.relu(k) + 0.5
        t = self.t_idx.to(alpha.device)
        log_w = (
            (alpha - 1) * torch.log(t) - t / theta
            - torch.lgamma(alpha) - alpha * torch.log(theta)
        )
        return torch.exp(log_w)


def _smar_production_step(
    P, PET, S1, S2, S3, S4, S5, S6,
    h_runoff, y_inf, smax, c_evap, g_rech, kg, nearzero,
):
    flux_pstar     = effective_1(P, PET, nearzero=nearzero)
    flux_estar     = effective_1(PET, P, nearzero=nearzero)
    flux_evap_base = torch.minimum(PET, P)
    S_tot   = S1 + S2 + S3 + S4 + S5
    flux_r1 = saturation_6(h_runoff, S_tot, smax, flux_pstar, nearzero=nearzero)
    inflow_i = flux_pstar - flux_r1
    flux_i  = infiltration_4(inflow_i, y_inf, nearzero=nearzero)
    flux_r2 = effective_1(inflow_i, flux_i, nearzero=nearzero)
    dev = P.device
    flux_e1 = evap_13(c_evap, torch.tensor(0.0, device=dev), flux_estar, S1, nearzero=nearzero)
    flux_e2 = evap_14(c_evap, torch.tensor(1.0, device=dev), flux_estar, S2, S1, torch.tensor(0.1, device=dev), nearzero=nearzero)
    flux_e3 = evap_14(c_evap, torch.tensor(2.0, device=dev), flux_estar, S3, S2, torch.tensor(0.1, device=dev), nearzero=nearzero)
    flux_e4 = evap_14(c_evap, torch.tensor(3.0, device=dev), flux_estar, S4, S3, torch.tensor(0.1, device=dev), nearzero=nearzero)
    flux_e5 = evap_14(c_evap, torch.tensor(4.0, device=dev), flux_estar, S5, S4, torch.tensor(0.1, device=dev), nearzero=nearzero)
    layer_cap = smax / 5.0
    flux_q1  = saturation_1(flux_i,  S1, layer_cap, nearzero=nearzero)
    flux_q2  = saturation_1(flux_q1, S2, layer_cap, nearzero=nearzero)
    flux_q3  = saturation_1(flux_q2, S3, layer_cap, nearzero=nearzero)
    flux_q4  = saturation_1(flux_q3, S4, layer_cap, nearzero=nearzero)
    flux_r3  = saturation_1(flux_q4, S5, layer_cap, nearzero=nearzero)
    flux_rg     = split_1(g_rech,       flux_r3, nearzero=nearzero)
    flux_r3star = split_1(1.0 - g_rech, flux_r3, nearzero=nearzero)
    flux_qg     = baseflow_1(kg, S6, nearzero=nearzero)
    S1_new = torch.clamp(S1 + flux_i  - flux_e1 - flux_q1, min=nearzero)
    S2_new = torch.clamp(S2 + flux_q1 - flux_e2 - flux_q2, min=nearzero)
    S3_new = torch.clamp(S3 + flux_q2 - flux_e3 - flux_q3, min=nearzero)
    S4_new = torch.clamp(S4 + flux_q3 - flux_e4 - flux_q4, min=nearzero)
    S5_new = torch.clamp(S5 + flux_q4 - flux_e5 - flux_r3, min=nearzero)
    S6_new = torch.clamp(S6 + flux_rg - flux_qg,           min=nearzero)
    return flux_r1 + flux_r2 + flux_r3star, flux_qg, S1_new, S2_new, S3_new, S4_new, S5_new, S6_new


class Smar(UnifyV2):
    """SMAR (MARRMoT m_40): 6-store production -> Gamma Conv + Baseflow."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "smar")
        super().__init__(config, device, backend)
        self.uh = DplGamma6(max_lag=_NK_MAX)
        self.production_step = _maybe_compile(_smar_production_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return tuple(z.clone() for _ in range(6))

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        h_runoff = params_dict["h_runoff"]
        y_inf    = params_dict["y_inf"]
        smax     = params_dict["smax"]
        c_evap   = params_dict["c_evap"]
        g_rech   = params_dict["g_rech"]
        kg       = params_dict["kg"]
        n_res    = params_dict["n_res"]
        nk_delay = params_dict["nk_delay"]

        def warmup_step(t, curr):
            S1, S2, S3, S4, S5, S6 = curr
            _, _, S1n, S2n, S3n, S4n, S5n, S6n = self.production_step(
                P_seq[t], PET_seq[t], S1, S2, S3, S4, S5, S6,
                h_runoff, y_inf, smax, c_evap, g_rech, kg, nearzero)
            return (S1n, S2n, S3n, S4n, S5n, S6n)

        S1, S2, S3, S4, S5, S6 = self._run_warmup(warmup_step, n_steps, states)

        qr_list, qg_list = [], []
        for t in range(n_steps):
            flux_qr, flux_qg, S1, S2, S3, S4, S5, S6 = self.production_step(
                P_seq[t], PET_seq[t], S1, S2, S3, S4, S5, S6,
                h_runoff, y_inf, smax, c_evap, g_rech, kg, nearzero)
            qr_list.append(flux_qr)
            qg_list.append(flux_qg)

        B = n_grid
        n_flat    = n_res.reshape(B, 1)
        k_flat    = nk_delay.reshape(B, 1) / (n_flat + nearzero)
        uh_params = torch.cat([n_flat, k_flat], dim=1)

        routed_qr = self.uh(
            torch.stack(qr_list, 0).permute(1, 2, 0).reshape(B, n_steps), uh_params
        ).view(n_grid, 1, n_steps).permute(2, 0, 1)

        return self._finalize_output(routed_qr + torch.stack(qg_list, 0), params_dict)
