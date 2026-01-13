"""
Unified Hydrological Model V2 (Multi-Start Optimizer Calibration)

Designed for parameter calibration using multi-start optimization algorithms.

Key Features:
- Input: Parameters from optimizer with shape (batch, n_params * nmul)
- Computation: Reshaped to (batch, n_params, nmul), then flattened to (batch*nmul,) for expanded batch
- Output: (time, batch*nmul) - flattened output for direct loss computation

Difference from V1:
- V1: NN-predicted params -> mean over nmul -> routed output (time, batch, 1)
- V2: Multi-start optimizer params -> flatten batch*nmul -> raw output (time, batch*nmul)
"""

from typing import Any, Optional, Union, Dict, Tuple, List
import torch
import torch.nn as nn
from dmg.models.phy_models.core import PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO


class UnifyV2(nn.Module):
    """
    Unified Hydrological Model V2 (Multi-Start Optimizer Calibration)
    
    Parameters from optimizer are expanded by flattening batch*nmul dimension,
    output shape is (time, batch*nmul) for direct loss computation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        super().__init__()

        self.config = config or {}
        self.model_name = self.config.get("model_name", "hbv96").lower()
        self.name = f"Unify_{self.model_name}"

        if self.model_name not in PARAM_INFO:
            raise ValueError(
                f"Unknown model_name: {self.model_name}. Available: {list(PARAM_INFO.keys())}"
            )

        self.parameter_bounds = PARAM_INFO[self.model_name]
        self.raw_step_fn = STFN_INFO[self.model_name]
        self.init_fn = INIT_INFO[self.model_name]
        self.n_states = STATE_INFO[self.model_name]

        self.warm_up = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1

        self.backend = self.config.get("backend", backend)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.backend == "compile" and hasattr(torch, "compile"):
            self.step_fn = torch.compile(self.raw_step_fn)
        elif self.backend == "jit":
            self.step_fn = torch.jit.script(self.raw_step_fn)
        else:
            self.step_fn = self.raw_step_fn

        self._load_config(self.config)
        self._set_parameters()

    def _load_config(self, config: Dict) -> None:
        for attr in ["warm_up", "warm_up_states", "variables", "nearzero", "nmul"]:
            if attr in config:
                setattr(self, attr, config[attr])
        self.check_water_balance = config.get("check_water_balance", False)

    def _set_parameters(self) -> None:
        self.phy_param_names = list(self.parameter_bounds.keys())
        self.learnable_param_count = len(self.phy_param_names)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        return self.init_fn(n_grid, self.nmul, self.device, self.nearzero)

    def _descale_params(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        bounds = self.parameter_bounds
        return {
            name: params[:, i, :] * (bounds[name][1] - bounds[name][0]) + bounds[name][0]
            for i, name in enumerate(self.phy_param_names)
        }

    def unpack_parameters(
        self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        _, raw_phy_static = parameters
        static_count = len(self.phy_param_names)
        return raw_phy_static[:, : static_count * self.nmul].view(
            raw_phy_static.shape[0], static_count, self.nmul
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        x_phy = x_dict["x_phy"]
        phy_static = self.unpack_parameters(parameters)
        n_grid = x_phy.size(1)
        states = self._init_states(n_grid)
        phy_static_dict = self._descale_params(phy_static)
        return self._run_model(x_dict, states, phy_static_dict)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul

        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        param_values = [static_params[name] for name in self.phy_param_names]

        Qsim_out = torch.empty(
            (n_steps, n_grid, nmul), device=self.device, dtype=torch.float32
        )

        if self.check_water_balance:
            P_out = forcing[..., 0:1].expand(-1, -1, nmul)
            Et_out = torch.empty(
                (n_steps, n_grid, nmul), device=self.device, dtype=torch.float32
            )
            St_sum_out = torch.empty(
                (n_steps, n_grid, nmul), device=self.device, dtype=torch.float32
            )
        else:
            P_out = None
            Et_out = None
            St_sum_out = None

        curr_states = states
        for t in range(n_steps):
            outputs = self.step_fn(
                P_seq[t], T_seq[t], PET_seq[t],
                *param_values,
                *curr_states,
                self.nearzero,
            )
            Qsim_out[t] = outputs[0]
            curr_states = outputs[2:]

            if self.check_water_balance:
                Et_out[t] = outputs[1]
                if self.model_name == 'penman':
                    s1 = curr_states[0]
                    s2 = curr_states[1]
                    s3 = curr_states[2]
                    St_sum_out[t] = s1 - s2 + s3
                elif self.model_name == 'topmodel':
                    St_sum_out[t] = curr_states[0] - curr_states[1]
                else:
                    St_sum_out[t] = sum(curr_states)

        return self._finalize_output(Qsim_out, P_out, Et_out, St_sum_out)

    def _finalize_output(
        self,
        Qsim_out: torch.Tensor,
        P_out: Optional[torch.Tensor] = None,
        Et_out: Optional[torch.Tensor] = None,
        St_sum_out: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        Qsimavg = Qsim_out.flatten(start_dim=1)
        result = {"streamflow": Qsimavg}

        if self.check_water_balance:
            if P_out is not None:
                result["precipitation"] = P_out.flatten(start_dim=1)
                
            if Et_out is not None:
                result["evaporation"] = Et_out.flatten(start_dim=1)

            if St_sum_out is not None:
                result["storage_sum"] = St_sum_out.flatten(start_dim=1)

        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.warm_up:]

        return result
