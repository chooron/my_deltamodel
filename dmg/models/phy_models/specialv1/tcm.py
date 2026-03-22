import torch
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1
from dmg.models.phy_models.core.tcm import tcm_step, create_initial_state


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


class Tcm(UnifyV1):
    """Thames Catchment Model (TCM) - 6 parameters, 4 stores."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "tcm")
        super().__init__(config, device, backend)
        self.model_step = _maybe_compile(tcm_step, self.backend)

    def _init_states(self, n_grid: int, nmul: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        # States: S1, S2, S3, S4
        return create_initial_state(n_grid, nmul or self.nmul, self.device, self.nearzero)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        nmul: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = nmul or self.nmul
        nearzero = self.nearzero

        # Compute mean_P from the entire precipitation time series
        # This matches MATLAB's init() function: ca = fa * mean(P)
        P_all = forcing[..., 0]  # shape: (n_steps, n_grid)
        mean_P = P_all.mean(dim=0, keepdim=True)  # shape: (1, n_grid)

        # Expand mean_P to match nmul
        mean_P_expanded = mean_P.expand(n_grid, nmul)  # shape: (n_grid, nmul)

        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack parameters
        phi = static_params["phi"]
        rc = static_params["rc"]
        gam = static_params["gam"]
        k1 = static_params["k1"]
        fa = static_params["fa"]
        k2 = static_params["k2"]

        S1, S2, S3, S4 = states
        warm_up = min(self.warm_up, n_steps)

        with torch.no_grad():
            for t in range(warm_up):
                _, _, S1, S2, S3, S4 = self.model_step(
                    P_seq[t], T_seq[t], PET_seq[t],
                    phi, rc, gam, k1, fa, k2,
                    S1, S2, S3, S4, mean_P_expanded, nearzero=nearzero)
        S1, S2, S3, S4 = (s.detach() for s in (S1, S2, S3, S4))

        q_list = []
        for t in range(warm_up, n_steps):
            Qsim, _, S1, S2, S3, S4 = self.model_step(
                P_seq[t], T_seq[t], PET_seq[t],
                phi, rc, gam, k1, fa, k2,
                S1, S2, S3, S4, mean_P_expanded, nearzero=nearzero)
            q_list.append(Qsim)

        return {"streamflow": torch.stack(q_list, dim=0).flatten(start_dim=1)}
