import torch
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v2 import UnifyV2
from dmg.models.hydromodel.mopex5 import mopex5_step, create_initial_state


# Compile step for speed
_mopex5_step = torch.compile(mopex5_step)


class Mopex5(UnifyV2):
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

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        # States: S1, S2, Sc1, Sc2, Sn
        return create_initial_state(n_grid, self.nmul, self.device, self.nearzero)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        dy_params: Dict[str, torch.Tensor],
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

        q_list = []

        for t in range(n_steps):
            Qsim, _, S1, S2, Sc1, Sc2, Sn = _mopex5_step(
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

        if self.initialize:
            return (S1, S2, Sc1, Sc2, Sn)

        Qsim_out = torch.stack(q_list, dim=0)
        return self._finalize_output(Qsim_out)
