import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v1 import UnifyV1
from dmg.models.phy_models.unithydro.uh_half_1 import DplHalf1
from dmg.models.phy_models.unithydro.uh_full_2 import DplFull2

GR4J_PARAMS_BOUNDS = {
    "x1": [1.0, 2000.0],
    "x2": [-20.0, 20.0],
    "x3": [1.0, 300.0],
    "x4": [0.5, 15.0],
}


def _calc_production_store_tanh(
    S: torch.Tensor, x1: torch.Tensor, Pn: torch.Tensor, En: torch.Tensor, nearzero: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    ratio_s_x1 = S / (x1 + nearzero)
    tanh_pn_x1 = torch.tanh(Pn / (x1 + nearzero))
    ps_num = x1 * (1.0 - ratio_s_x1.pow(2)) * tanh_pn_x1
    ps_den = 1.0 + ratio_s_x1 * tanh_pn_x1
    ps = ps_num / (ps_den + nearzero)

    tanh_en_x1 = torch.tanh(En / (x1 + nearzero))
    es_num = S * (2.0 - ratio_s_x1) * tanh_en_x1
    es_den = 1.0 + (1.0 - ratio_s_x1) * tanh_en_x1
    es = es_num / (es_den + nearzero)
    return ps, es


def _calc_percolation_analytical(
    S: torch.Tensor, x1: torch.Tensor, nearzero: float
) -> torch.Tensor:
    ratio_perc = (4.0 / 9.0) * (S / (x1 + nearzero))
    term_perc = (1.0 + ratio_perc.pow(4)).pow(-0.25)
    return S * (1.0 - term_perc)


def _calc_routing_outflow_analytical(
    S2: torch.Tensor, x3: torch.Tensor, nearzero: float
) -> torch.Tensor:
    ratio_s2_x3 = S2 / (x3 + nearzero)
    term_qr = (1.0 + ratio_s2_x3.pow(4)).pow(-0.25)
    return S2 * (1.0 - term_qr)


def _gr4j_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,
    x1: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = P - PET
    flux_pn = F.relu(diff)
    flux_en = F.relu(-diff)
    flux_ei = P - flux_pn

    nearzero_tensor = torch.zeros_like(flux_pn) + nearzero
    S1 = torch.clamp(S1, min=nearzero_tensor, max=x1)

    flux_ps, flux_es = _calc_production_store_tanh(S1, x1, flux_pn, flux_en, nearzero)

    S1_mid = S1 - flux_es + flux_ps
    S1_mid = torch.clamp(S1_mid, min=nearzero_tensor, max=x1)

    flux_perc = _calc_percolation_analytical(S1_mid, x1, nearzero)
    S1_new = S1_mid - flux_perc
    S1_new = torch.clamp(S1_new, min=nearzero_tensor, max=x1)

    flux_pr = flux_perc + (flux_pn - flux_ps)
    e_physical = flux_ei + flux_es

    return flux_pr, e_physical, S1_new


def _gr4j_routing_step_impl(
    flux_q9: torch.Tensor,
    flux_q1: torch.Tensor,
    S2: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    e_physical: torch.Tensor,
    nearzero: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nearzero_tensor = torch.zeros_like(flux_q9) + nearzero
    S2 = torch.clamp(S2, min=nearzero_tensor)

    flux_f_theoretical = x2 * (S2 / (x3 + nearzero)).pow(3.5)

    S2_before = S2
    S2_integrated = S2 + flux_q9 + flux_f_theoretical
    S2_integrated = torch.clamp(S2_integrated, min=nearzero_tensor)

    f_actual_s2 = (S2_integrated - S2_before) - flux_q9

    flux_qr = _calc_routing_outflow_analytical(S2_integrated, x3, nearzero)
    S2_new = S2_integrated - flux_qr

    flux_qd_potential = flux_q1 + flux_f_theoretical
    flux_qd = F.relu(flux_qd_potential)
    f_actual_q1 = flux_qd - flux_q1

    q_total = flux_qr + flux_qd

    F_total_actual = f_actual_s2 + f_actual_q1
    ea_balanced = e_physical - F_total_actual

    return q_total, ea_balanced, S2_new


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


class Gr4j(UnifyV1):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "gr4j")
        super().__init__(config, device, backend)

        max_lag_val = GR4J_PARAMS_BOUNDS["x4"][1]
        self.uh_1 = DplHalf1(max_lag=int(max_lag_val) + 1)
        self.uh_2 = DplFull2(max_lag=int(max_lag_val) * 2 + 2)

        self.production_step = _maybe_compile(_gr4j_production_step_impl, self.backend)
        self.routing_step = _maybe_compile(_gr4j_routing_step_impl, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        S1 = torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
        S2 = torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
        return (S1, S2)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul
        nearzero = self.nearzero

        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        pet_idx = 2 if forcing.shape[-1] > 2 else 1
        PET_seq = forcing[..., pet_idx : pet_idx + 1].expand(-1, -1, nmul).unbind(0)

        x1 = static_params["x1"]
        x2 = static_params["x2"]
        x3 = static_params["x3"]
        x4 = static_params["x4"]

        S1, S2 = states

        track_balance = self.check_water_balance
        Et_out: Optional[torch.Tensor] = None
        state_series: Optional[List[torch.Tensor]] = None
        S_init_sum: Optional[torch.Tensor] = None

        if track_balance:
            Et_out = torch.empty((n_steps, n_grid, nmul), device=self.device)
            state_series = [
                torch.empty((n_steps + 1, n_grid, nmul), device=self.device)
                for _ in range(2)
            ]
            state_series[0][0] = S1
            state_series[1][0] = S2
            S_init_sum = torch.stack([s.clone() for s in states]).sum(dim=0)

        flux_pr_list = []
        e_phys_list = []
        for t in range(n_steps):
            flux_pr, e_phys, S1 = self.production_step(
                P_seq[t], PET_seq[t], S1, x1, nearzero
            )
            flux_pr_list.append(flux_pr)
            e_phys_list.append(e_phys)
            if track_balance and state_series is not None:
                state_series[0][t + 1] = S1

        flux_pr_stack = torch.stack(flux_pr_list, dim=0)
        e_phys_stack = torch.stack(e_phys_list, dim=0)

        flux_pr90 = flux_pr_stack * 0.9
        flux_pr10 = flux_pr_stack * 0.1

        B_total = n_grid * nmul
        pr90_flat = flux_pr90.permute(1, 2, 0).reshape(B_total, n_steps)
        pr10_flat = flux_pr10.permute(1, 2, 0).reshape(B_total, n_steps)

        x4_flat = x4.reshape(B_total, 1)
        x4_flat_uh2 = x4_flat * 2.0

        routed_q9_flat = self.uh_1(pr90_flat, x4_flat)
        routed_q1_flat = self.uh_2(pr10_flat, x4_flat_uh2)

        q9_seq = (
            routed_q9_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)
        )
        q1_seq = (
            routed_q1_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1).unbind(0)
        )
        e_phys_seq = e_phys_stack.unbind(0)

        Qsim_list = []
        for t in range(n_steps):
            q_total, ea_balanced, S2 = self.routing_step(
                q9_seq[t], q1_seq[t], S2, x2, x3, e_phys_seq[t], nearzero
            )
            Qsim_list.append(q_total)
            if track_balance and state_series is not None and Et_out is not None:
                Et_out[t] = ea_balanced
                state_series[1][t + 1] = S2

        Qsim_out = torch.stack(Qsim_list, dim=0)
        final_states = (S1, S2)

        if track_balance:
            return self._finalize_output(
                Qsim_out,
                Et_out,
                S_init_sum,
                final_states,
                state_series,
            )

        return self._finalize_output(Qsim_out)