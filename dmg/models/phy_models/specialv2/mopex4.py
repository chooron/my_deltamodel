import torch
from typing import Dict, Tuple, Optional, Any

from dmg.models.phy_models.unify_v2 import UnifyV2, _maybe_compile
from dmg.models.phy_models.core.mopex4 import mopex4_step, create_initial_state


class Mopex4(UnifyV2):
    """Mopex4: seasonal interception + snow + two-bucket routing."""

    def __init__(self, config=None, device=None, backend="compile"):
        if config is None:
            config = {}
        config.setdefault("model_name", "mopex4")
        super().__init__(config, device, backend)
        self.model_step = _maybe_compile(mopex4_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        return create_initial_state(n_grid, 1, self.device, self.nearzero)

    def _run_model(self, x_dict, states, params_dict):
        forcing = x_dict["x_phy"]
        doy_raw = x_dict["doy"]
        n_steps, n_grid = forcing.shape[:2]
        nearzero = self.nearzero

        P_seq   = forcing[..., 0:1].unbind(0)
        T_seq   = forcing[..., 1:2].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        if doy_raw.ndim == 2:
            doy_raw = doy_raw.unsqueeze(-1)
        doy_seq = doy_raw.unbind(0)

        tcrit   = params_dict["tcrit"]
        ddf     = params_dict["ddf"]
        Sb1     = params_dict["s2max"]
        tw      = params_dict["tw"]
        alpha   = params_dict["alpha"]
        is_time = params_dict["is_time"]
        tu      = params_dict["tu"]
        Se      = params_dict["se"]
        Sb2     = params_dict["s3max"]
        tc      = params_dict["tc"]

        def warmup_step(t, curr):
            S1, S2, Sc1, Sc2, Sn = curr
            _, _, S1n, S2n, Sc1n, Sc2n, Snn = self.model_step(
                P_seq[t], T_seq[t], PET_seq[t], doy_seq[t],
                tcrit, ddf, Sb1, tw, alpha, is_time, tu, Se, Sb2, tc,
                S1, S2, Sc1, Sc2, Sn, nearzero=nearzero)
            return (S1n, S2n, Sc1n, Sc2n, Snn)

        S1, S2, Sc1, Sc2, Sn = self._run_warmup(warmup_step, n_steps, states)

        q_list = []
        for t in range(n_steps):
            Qsim, _, S1, S2, Sc1, Sc2, Sn = self.model_step(
                P_seq[t], T_seq[t], PET_seq[t], doy_seq[t],
                tcrit, ddf, Sb1, tw, alpha, is_time, tu, Se, Sb2, tc,
                S1, S2, Sc1, Sc2, Sn, nearzero=nearzero)
            q_list.append(Qsim)

        return self._finalize_output(torch.stack(q_list, dim=0), params_dict)
