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

[优化] _run_model：将单一时间循环拆分为两段
  - Warmup 段（前 warm_up 步）：在 torch.no_grad() 下运行，不建立计算图，
    结束后对状态执行 .detach()，节省 warmup 期间约 50% 的显存和时间。
  - 训练段（剩余步数）：正常运行，建立完整计算图。
"""

from typing import Any, Optional, Dict, Tuple
import torch
import torch.nn as nn
from dmg.models.phy_models.core import PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO


class UnifyV1(nn.Module):
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
        self.nearzero = float(self.nearzero)

    def _set_parameters(self) -> None:
        self.phy_param_names = list(self.parameter_bounds.keys())
        self.learnable_param_count = len(self.phy_param_names)

    def _init_states(self, n_grid: int, nmul: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        return self.init_fn(n_grid, nmul if nmul is not None else self.nmul, self.device, self.nearzero)

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
        if raw_phy_static.dim() == 3:
            # Already (batch, n_params, num_start) from Calibrate model
            return raw_phy_static
        static_count = len(self.phy_param_names)
        return raw_phy_static[:, : static_count * self.nmul].view(
            raw_phy_static.shape[0], static_count, self.nmul
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x_phy = x_dict["x_phy"]
        phy_static = self.unpack_parameters(parameters)
        # Use actual num_start from tensor, not config nmul
        actual_nmul = phy_static.shape[2]
        n_grid = x_phy.size(1)
        states = self._init_states(n_grid, actual_nmul)
        phy_static_dict = self._descale_params(phy_static)
        return self._run_model(x_dict, states, phy_static_dict, actual_nmul)

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        nmul: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = nmul if nmul is not None else self.nmul
        effective_warmup = min(self.warm_up, n_steps)

        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        param_values = [static_params[name] for name in self.phy_param_names]
        curr_states = states

        # ── Warmup 段：no_grad，只更新状态，不收集输出 ────────────────────────
        with torch.no_grad():
            for t in range(effective_warmup):
                outputs = self.step_fn(
                    P_seq[t], T_seq[t], PET_seq[t],
                    *param_values,
                    *curr_states,
                    nearzero=self.nearzero,
                )
                curr_states = outputs[2:]

        # detach 状态，切断梯度流
        curr_states = tuple(s.detach() for s in curr_states)

        # ── 训练段：正常建图 ──────────────────────────────────────────────────
        n_train = n_steps - effective_warmup
        train_Q = torch.empty(
            (n_train, n_grid, nmul), device=self.device, dtype=torch.float32
        )

        for i in range(n_train):
            t = effective_warmup + i
            outputs = self.step_fn(
                P_seq[t], T_seq[t], PET_seq[t],
                *param_values,
                *curr_states,
                nearzero=self.nearzero,
            )
            train_Q[i] = outputs[0]
            curr_states = outputs[2:]

        return {"streamflow": train_Q.flatten(start_dim=1)}

