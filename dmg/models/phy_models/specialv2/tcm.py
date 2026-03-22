import torch
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.core.tcm import tcm_step, create_initial_state


class Tcm(UnifyV2):
    """Thames Catchment Model (TCM) - 6 parameters, 4 stores."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "tcm")
        super().__init__(config, device, backend)
        self.model_step = _maybe_compile(tcm_step, self.backend)

        # Store mean_P for abstraction calculation
        self.mean_P = None

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        return create_initial_state(n_grid, 1, self.device, self.nearzero)

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        # Compute mean_P from the entire precipitation time series
        # This matches MATLAB's init() function: ca = fa * mean(P)
        P_all = forcing[..., 0]  # shape: (n_steps, n_grid)
        mean_P = P_all.mean(dim=0, keepdim=True)  # shape: (1, n_grid)

        # Expand mean_P to match the model shape
        mean_P_expanded = mean_P.expand(n_grid, 1)  # shape: (n_grid, 1)

        P_seq   = forcing[..., 0:1].unbind(0)
        T_seq   = forcing[..., 1:2].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        phi = params_dict["phi"]
        rc = params_dict["rc"]
        gam = params_dict["gam"]
        k1 = params_dict["k1"]
        fa = params_dict["fa"]
        k2 = params_dict["k2"]

        def warmup_step(t, curr):
            S1, S2, S3, S4 = curr
            _, _, S1n, S2n, S3n, S4n = self.model_step(
                P_seq[t], T_seq[t], PET_seq[t],
                phi, rc, gam, k1, fa, k2,
                S1, S2, S3, S4, mean_P_expanded, nearzero=nearzero)
            return (S1n, S2n, S3n, S4n)

        S1, S2, S3, S4 = self._run_warmup(warmup_step, n_steps, states)

        q_list = []
        for t in range(n_steps):
            Qsim, _, S1, S2, S3, S4 = self.model_step(
                P_seq[t], T_seq[t], PET_seq[t],
                phi, rc, gam, k1, fa, k2,
                S1, S2, S3, S4, mean_P_expanded, nearzero=nearzero)
            q_list.append(Qsim)

        return self._finalize_output(torch.stack(q_list, dim=0), params_dict)
