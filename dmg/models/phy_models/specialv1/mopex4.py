import torch
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v1 import UnifyV1
from dmg.models.phy_models.core.mopex4 import mopex4_step, create_initial_state


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


# ==============================================================================
# 3. Model Class (Mopex4)
# ==============================================================================

class Mopex4(UnifyV1):
    """
    Mopex4 Hydrological Model (seasonal interception + snow + two-bucket routing).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "mopex4")
        super().__init__(config, device, backend)
        self.model_step = _maybe_compile(mopex4_step, self.backend)

    def _init_states(self, n_grid: int, nmul: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        # S1, S2, Sc1, Sc2, Sn
        return create_initial_state(n_grid, nmul or self.nmul, self.device, self.nearzero)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        nmul: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        doy_raw = x_dict["doy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = nmul or self.nmul
        nearzero = self.nearzero

        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Day of year -> expand to match nmul
        if doy_raw.ndim == 2:
            doy_raw = doy_raw.unsqueeze(-1)
        doy_seq = doy_raw.expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        tcrit = static_params["tcrit"]
        ddf = static_params["ddf"]
        Sb1 = static_params["s2max"]
        tw = static_params["tw"]
        alpha = static_params["alpha"]
        is_time = static_params["is_time"]
        tu = static_params["tu"]
        Se = static_params["se"]
        Sb2 = static_params["s3max"]
        tc = static_params["tc"]

        S1, S2, Sc1, Sc2, Sn = states
        warm_up = min(self.warm_up, n_steps)

        # ── Warmup: no_grad ──────────────────────────────────────────
        with torch.no_grad():
            for t in range(warm_up):
                _, _, S1, S2, Sc1, Sc2, Sn = self.model_step(
                    P_seq[t], T_seq[t], PET_seq[t], doy_seq[t],
                    tcrit, ddf, Sb1, tw, alpha, is_time, tu, Se, Sb2, tc,
                    S1, S2, Sc1, Sc2, Sn, nearzero=nearzero)
        S1, S2, Sc1, Sc2, Sn = (s.detach() for s in (S1, S2, Sc1, Sc2, Sn))

        # ── Train: normal graph ──────────────────────────────────────
        q_list = []
        for t in range(warm_up, n_steps):
            Qsim, _, S1, S2, Sc1, Sc2, Sn = self.model_step(
                P_seq[t], T_seq[t], PET_seq[t], doy_seq[t],
                tcrit, ddf, Sb1, tw, alpha, is_time, tu, Se, Sb2, tc,
                S1, S2, Sc1, Sc2, Sn, nearzero=nearzero)
            q_list.append(Qsim)

        return {"streamflow": torch.stack(q_list, dim=0).flatten(start_dim=1)}