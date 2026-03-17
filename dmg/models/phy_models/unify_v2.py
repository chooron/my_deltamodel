"""
Unified Hydrological Model V2 (Warmup-Optimized, nmul=1)

改进点：
- nmul 固定为 1，不再支持 ensemble，简化计算路径
- static_params 与 routing_param_dict 统一为 params_dict
- _finalize_output 直接输出 streamflow，不做 routing（子类可覆盖）
- warmup 段 no_grad + detach，训练段正常建图
- 保留水量平衡检测（默认关闭）
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from dmg.models.hydrodl2 import change_param_range
from dmg.models.phy_models.core import (
    PARAM_INFO,
    STFN_INFO,
    INIT_INFO,
    STATE_INFO,
)


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        return torch.compile(fn)
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


class UnifyV2(nn.Module):
    """
    Unified Hydrological Model V2 — nmul=1, warmup-optimized.

    forward() 接受 (x_dict, parameters)，parameters = (_, raw_static)
    raw_static shape: (batch, n_params)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        super().__init__()

        self.config = config or {}
        self.model_name = self.config.get("model_name", "hbv96")
        self.name = f"UnifyV2_{self.model_name}"

        if self.model_name not in PARAM_INFO:
            raise ValueError(
                f"Unknown model_name: {self.model_name}. "
                f"Available: {list(PARAM_INFO.keys())}"
            )

        self.parameter_bounds = PARAM_INFO[self.model_name]
        self.raw_step_fn = STFN_INFO[self.model_name]
        self.init_fn = INIT_INFO[self.model_name]
        self.n_states = STATE_INFO[self.model_name]

        # nmul is always 1
        self.nmul = 1
        self.warm_up = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.check_water_balance = False

        self.backend = self.config.get("backend", backend)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.step_fn = _maybe_compile(self.raw_step_fn, self.backend)

        self._load_config(self.config)
        self.phy_param_names = list(self.parameter_bounds.keys())
        self.learnable_param_count = len(self.phy_param_names)

    def _load_config(self, config: Dict) -> None:
        for attr in ["warm_up", "warm_up_states", "variables", "nearzero",
                     "check_water_balance"]:
            if attr in config:
                setattr(self, attr, config[attr])
        self.nearzero = float(self.nearzero)

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        return self.init_fn(n_grid, 1, self.device, self.nearzero)

    def _descale_params(
        self, raw: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """(batch, n_params) -> dict of (batch, 1) tensors"""
        return {
            name: change_param_range(
                raw[:, i:i+1], self.parameter_bounds[name]
            )
            for i, name in enumerate(self.phy_param_names)
        }

    def unpack_parameters(
        self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Extract raw static params -> (batch, n_params).
        Handles both (batch, n_params) and (batch, n_params, 1) from Parameterize.
        """
        _, raw = parameters
        if raw.dim() == 3:
            raw = raw.squeeze(-1)   # (batch, n_params, 1) -> (batch, n_params)
        return raw[:, : len(self.phy_param_names)]

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raw = self.unpack_parameters(parameters)
        n_grid = x_dict["x_phy"].size(1)
        states = self._init_states(n_grid)
        params_dict = self._descale_params(raw)
        return self._run_model(x_dict, states, params_dict)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        params_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Warmup 段 no_grad + detach，训练段正常建图。
        nmul=1，forcing shape: (time, batch, 1)
        """
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        effective_warmup = min(self.warm_up, n_steps)

        P_seq   = forcing[..., 0:1].unbind(0)
        T_seq   = forcing[..., 1:2].unbind(0)
        PET_seq = forcing[..., 2:3].unbind(0)

        param_values = [params_dict[name] for name in self.phy_param_names]
        curr_states = states

        # ── Warmup: no_grad ──────────────────────────────────────────
        warmup_outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for t in range(effective_warmup):
                out = self.step_fn(
                    P_seq[t], T_seq[t], PET_seq[t],
                    *param_values, *curr_states, self.nearzero,
                )
                if self.warm_up_states:
                    warmup_outputs.append(out[0])
                curr_states = out[2:]

        curr_states = tuple(s.detach() for s in curr_states)

        # ── Train: normal graph ──────────────────────────────────────
        n_train = n_steps - effective_warmup
        train_out = torch.empty(
            (n_train, n_grid, 1), device=self.device, dtype=torch.float32
        )
        for i in range(n_train):
            t = effective_warmup + i
            out = self.step_fn(
                P_seq[t], T_seq[t], PET_seq[t],
                *param_values, *curr_states, self.nearzero,
            )
            train_out[i] = out[0]
            curr_states = out[2:]

        if self.warm_up_states and warmup_outputs:
            Qsim_out = torch.cat(
                [torch.stack(warmup_outputs, dim=0), train_out], dim=0
            )
        else:
            Qsim_out = train_out

        return self._finalize_output(Qsim_out, params_dict)

    def _run_warmup(
        self,
        step_fn,
        n_steps: int,
        states: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Run warmup steps under no_grad, return warmed+detached states.
        step_fn: callable(t, curr_states) -> new_states
        """
        effective_warmup = min(self.warm_up, n_steps)
        curr_states = states
        with torch.no_grad():
            for t in range(effective_warmup):
                curr_states = step_fn(t, curr_states)
        return tuple(s.detach() for s in curr_states)

    def _finalize_output(
        self,
        Qsim_out: torch.Tensor,
        params_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Default: squeeze nmul dim and return streamflow directly.
        子类（有内置汇流的模型）可覆盖此方法。
        Input: (time, batch, 1) -> Output: (time, batch)
        """
        return {"streamflow": Qsim_out.squeeze(-1)}
