import torch
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v1 import UnifyV1
from dmg.models.phy_models.core.mopex5 import mopex5_step, create_initial_state


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


class Mopex5(UnifyV1):
    """Mopex5 hydrological model (phenology-aware interception, snow, two-bucket routing)."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "mopex5")
        super().__init__(config, device, backend)
        self.model_step = _maybe_compile(mopex5_step, self.backend)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        # States: S1, S2, Sc1, Sc2, Sn
        return create_initial_state(n_grid, self.nmul, self.device, self.nearzero)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        doy_raw = x_dict["doy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul
        nearzero = self.nearzero

        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Day of year expand to match nmul
        if doy_raw.ndim == 2:
            doy_raw = doy_raw.unsqueeze(-1)
        doy_seq = doy_raw.expand(-1, -1, nmul).unbind(0)

        # Unpack parameters
        Sb1 = static_params["Sb1"]
        tw = static_params["tw"]
        tu = static_params["tu"]
        Se = static_params["Se"]
        tc = static_params["tc"]
        ddf = static_params["ddf"]
        tcrit = static_params["tcrit"]
        Sb2 = static_params["Sb2"]
        alpha = static_params["alpha"]
        is_time = static_params["is_time"]
        tmin = static_params["tmin"]
        tmax = static_params["tmax"]

        S1, S2, Sc1, Sc2, Sn = states

        track_balance = self.check_water_balance
        if track_balance:
            Et_out = torch.empty(
                (n_steps, n_grid, nmul), device=self.device, dtype=torch.float32
            )
            state_series: Optional[List[torch.Tensor]] = [
                torch.empty(
                    (n_steps + 1, n_grid, nmul),
                    device=self.device,
                    dtype=torch.float32,
                )
                for _ in range(len(states))
            ]
            for idx, state in enumerate(states):
                state_series[idx][0] = state
            S_init_sum = torch.stack([s.clone() for s in states]).sum(dim=0)
        else:
            Et_out = None
            state_series = None
            S_init_sum = None

        q_list = []

        for t in range(n_steps):
            Qsim, flux_ea, S1, S2, Sc1, Sc2, Sn = self.model_step(
                P_seq[t],
                T_seq[t],
                PET_seq[t],
                doy_seq[t],
                Sb1,
                tw,
                tu,
                Se,
                tc,
                ddf,
                tcrit,
                Sb2,
                alpha,
                is_time,
                tmin,
                tmax,
                S1,
                S2,
                Sc1,
                Sc2,
                Sn,
                nearzero=nearzero,
            )
            q_list.append(Qsim)
            if track_balance:
                Et_out[t] = flux_ea
                if state_series is not None:
                    state_series[0][t + 1] = S1
                    state_series[1][t + 1] = S2
                    state_series[2][t + 1] = Sc1
                    state_series[3][t + 1] = Sc2
                    state_series[4][t + 1] = Sn

        Qsim_out = torch.stack(q_list, dim=0)
        final_states = (S1, S2, Sc1, Sc2, Sn)

        if track_balance:
            return self._finalize_output(
                Qsim_out,
                Et_out,
                S_init_sum,
                final_states,
                state_series,
            )

        return self._finalize_output(Qsim_out)
