import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.unithydro.uh_half_1 import DplHalf1
from dmg.models.phy_models.unithydro.uh_full_2 import DplFull2

_X4_MAX = 15


def _calc_production_store_tanh(S, x1, Pn, En, nearzero):
    r = S / (x1 + nearzero)
    ps = x1 * (1.0 - r**2) * torch.tanh(Pn / (x1 + nearzero)) / (
        1.0 + r * torch.tanh(Pn / (x1 + nearzero)) + nearzero)
    es = S * (2.0 - r) * torch.tanh(En / (x1 + nearzero)) / (
        1.0 + (1.0 - r) * torch.tanh(En / (x1 + nearzero)) + nearzero)
    return ps, es


def _gr4j_production_step(P, PET, S1, x1, nearzero):
    diff = P - PET
    Pn = F.relu(diff)
    En = F.relu(-diff)
    nz = torch.zeros_like(Pn) + nearzero
    S1 = torch.clamp(S1, min=nz, max=x1)
    ps, es = _calc_production_store_tanh(S1, x1, Pn, En, nearzero)
    S1_mid = torch.clamp(S1 - es + ps, min=nz, max=x1)
    perc = S1_mid * (1.0 - (1.0 + (4.0 / 9.0 * S1_mid / (x1 + nearzero))**4)**(-0.25))
    S1_new = torch.clamp(S1_mid - perc, min=nz, max=x1)
    return perc + (Pn - ps), P - Pn + es, S1_new


def _gr4j_routing_step(q9, q1, S2, x2, x3, nearzero):
    nz = torch.zeros_like(q9) + nearzero
    S2 = torch.clamp(S2, min=nz)
    flux_f = x2 * (S2 / (x3 + nearzero))**3.5
    S2_int = torch.clamp(S2 + q9 + flux_f, min=nz)
    qr = S2_int * (1.0 - (1.0 + (S2_int / (x3 + nearzero))**4)**(-0.25))
    S2_new = S2_int - qr
    qd = F.relu(q1 + flux_f)
    return qr + qd, S2_new


class Gr4j(UnifyV2):
    """GR4J: Production -> Split Conv -> Routing."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "gr4j")
        super().__init__(config, device, backend)
        self.uh_1 = DplHalf1(max_lag=_X4_MAX + 1)
        self.uh_2 = DplFull2(max_lag=_X4_MAX * 2 + 2)
        self.production_step = _maybe_compile(_gr4j_production_step, self.backend)
        self.routing_step = _maybe_compile(_gr4j_routing_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        z = torch.zeros((n_grid, 1), device=self.device) + self.nearzero
        return (z.clone(), z.clone())

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        pet_idx = 2 if forcing.shape[-1] > 2 else 1
        PET_seq = forcing[..., pet_idx:pet_idx + 1].unbind(0)

        x1 = params_dict["x1"]
        x2 = params_dict["x2"]
        x3 = params_dict["x3"]
        x4 = params_dict["x4"]

        def warmup_step(t, curr):
            S1, S2 = curr
            _, _, S1_new = self.production_step(P_seq[t], PET_seq[t], S1, x1, nearzero)
            return (S1_new, S2)

        S1, S2 = self._run_warmup(warmup_step, n_steps, states)

        pr_list = []
        for t in range(n_steps):
            flux_pr, _, S1 = self.production_step(P_seq[t], PET_seq[t], S1, x1, nearzero)
            pr_list.append(flux_pr)

        pr_stack = torch.stack(pr_list, dim=0)
        B = n_grid
        x4_flat = x4.reshape(B, 1)

        q9_seq = self.uh_1(
            (pr_stack * 0.9).permute(1, 2, 0).reshape(B, n_steps), x4_flat
        ).view(n_grid, 1, n_steps).permute(2, 0, 1).unbind(0)
        q1_seq = self.uh_2(
            (pr_stack * 0.1).permute(1, 2, 0).reshape(B, n_steps), x4_flat * 2.0
        ).view(n_grid, 1, n_steps).permute(2, 0, 1).unbind(0)

        q_list = []
        for t in range(n_steps):
            q_total, S2 = self.routing_step(q9_seq[t], q1_seq[t], S2, x2, x3, nearzero)
            q_list.append(q_total)

        return self._finalize_output(torch.stack(q_list, dim=0), params_dict)
