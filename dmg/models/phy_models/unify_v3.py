"""
Unified Hydrological Model V3 (Warmup-Optimized)

基于 UnifyV1 的改进版本，针对长序列训练进行了优化：

[优化] _run_model：将单一时间循环拆分为两段
  - Warmup 段（前 warm_up 步）：在 torch.no_grad() 下运行，不建立计算图，
    结束后对状态执行 .detach()，节省 warmup 期间约 50% 的显存和时间。
  - 训练段（剩余步数）：正常运行，建立完整计算图。

Difference from V1/V2:
  - V1: 完整序列建图，包括 warmup 段
  - V3: warmup 段使用 no_grad，仅训练段建图，显著减少显存占用
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from dmg.models.hydrodl2 import change_param_range, uh_gamma, uh_conv
from dmg.models.phy_models.core import PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO


class UnifyV3(nn.Module):
    """
    Unified Hydrological Model V3 — Warmup-Optimized.

    与 UnifyV1 接口完全兼容；forward() 采用 warmup-no_grad 优化，
    减少显存占用和训练时间。
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
        self.name = f"UnifyV3_{self.model_name}"

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
        self.routing = False

        self.routing_parameter_bounds = {
            'rout_a': [0, 2.9],
            'rout_b': [0, 6.5],
        }

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
        for attr in ["warm_up", "warm_up_states", "variables", "nearzero", "nmul", "routing"]:
            if attr in config:
                setattr(self, attr, config[attr])
        self.check_water_balance = config.get("check_water_balance", False)

    def _set_parameters(self) -> None:
        self.phy_param_names = list(self.parameter_bounds.keys())
        if self.routing:
            self.routing_param_names = list(self.routing_parameter_bounds.keys())
        else:
            self.routing_param_names = []
        self.learnable_param_count = (
            len(self.phy_param_names) * self.nmul + len(self.routing_param_names)
        )

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        return self.init_fn(n_grid, self.nmul, self.device, self.nearzero)

    def _descale_params(
        self, params: torch.Tensor, names: List[str], bounds: Dict[str, List[float]]
    ) -> Dict[str, torch.Tensor]:
        return {
            name: change_param_range(params[:, i, :], bounds[name])
            for i, name in enumerate(names)
        }

    def _descale_routing_params(
        self, routing_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        routing_params = torch.sigmoid(routing_params)
        return {
            name: change_param_range(
                routing_params[:, i], self.routing_parameter_bounds[name]
            )
            for i, name in enumerate(self.routing_param_names)
        }

    def unpack_parameters(
        self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(batch, n_params*nmul + n_routing_params) -> (batch, n_params, nmul), routing"""
        _, raw_phy_static = parameters
        param_count = len(self.phy_param_names)
        phy_params = raw_phy_static[:, : param_count * self.nmul].view(
            raw_phy_static.shape[0], param_count, self.nmul
        )

        routing_params = None
        if self.routing:
            routing_params = raw_phy_static[:, param_count * self.nmul :]

        return phy_params, routing_params

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """向后兼容的 forward 接口，内部使用 [改动 1] 的 warmup-no_grad 优化。"""
        x_phy = x_dict["x_phy"]
        phy_static, routing_params = self.unpack_parameters(parameters)
        n_grid = x_phy.size(1)
        states = self._init_states(n_grid)
        phy_static_dict = self._descale_params(
            phy_static, self.phy_param_names, self.parameter_bounds
        )

        routing_param_dict = None
        if self.routing and routing_params is not None:
            routing_param_dict = self._descale_routing_params(routing_params)

        return self._run_model(x_dict, states, phy_static_dict, routing_param_dict)

    # -----------------------------------------------------------------------
    # [改动 1] _run_model：warmup 段 no_grad + 状态 detach，训练段正常建图
    # -----------------------------------------------------------------------
    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        static_params: Dict[str, torch.Tensor],
        routing_param_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        优化的模型运行方法，相较于 UnifyV1：
          - Warmup 段在 torch.no_grad() 下执行，不建立计算图。
          - Warmup 结束后对 curr_states 调用 .detach()，切断梯度流。
          - 训练段正常执行，为后续 backward 保留计算图。
        """
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul
        # warmup 步数上限不超过总步数
        effective_warmup = min(self.warm_up, n_steps)

        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[..., 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        param_values = [static_params[name] for name in self.phy_param_names]
        curr_states = states

        # ── Warmup 段：no_grad，仅在 warm_up_states=True 时收集输出 ──────────
        warmup_outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for t in range(effective_warmup):
                outputs = self.step_fn(
                    P_seq[t], T_seq[t], PET_seq[t],
                    *param_values,
                    *curr_states,
                    self.nearzero,
                )
                if self.warm_up_states:
                    warmup_outputs.append(outputs[0])
                curr_states = outputs[2:]

        # detach 状态，切断梯度流
        curr_states = tuple(s.detach() for s in curr_states)

        # ── 训练段：正常建图，预分配张量避免 list + stack 开销 ─────────────
        n_train = n_steps - effective_warmup
        train_out = torch.empty(
            (n_train, n_grid, nmul), device=self.device, dtype=torch.float32
        )
        for i, t in enumerate(range(effective_warmup, n_steps)):
            outputs = self.step_fn(
                P_seq[t], T_seq[t], PET_seq[t],
                *param_values,
                *curr_states,
                self.nearzero,
            )
            train_out[i] = outputs[0]
            curr_states = outputs[2:]

        # ── 拼合 Qsim_out ─────────────────
        if self.warm_up_states and warmup_outputs:
            Qsim_out = torch.cat(
                [torch.stack(warmup_outputs, dim=0), train_out], dim=0
            )
        else:
            Qsim_out = train_out

        return self._finalize_output(Qsim_out, routing_param_dict)

    def _finalize_output(
        self,
        Qsim_out: torch.Tensor,
        routing_param_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize with ensemble averaging and optional unit hydrograph routing.
        Input: (time, batch, nmul) -> Output: (time, batch)

        注意：当 warm_up_states=False 时，Qsim_out 已经在 _run_model 中排除了 warmup 段，
        因此这里不需要再次截断。
        """
        Qsimavg = Qsim_out.mean(-1)

        if self.routing and routing_param_dict is not None:
            n_steps, n_grid = Qsimavg.shape

            UH = uh_gamma(
                routing_param_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1),
                routing_param_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=15,
            )

            rf = torch.unsqueeze(Qsimavg, -1).permute([1, 2, 0])
            UH = UH.permute([1, 2, 0])

            Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
            streamflow = Qsrout.squeeze(-1)
        else:
            streamflow = Qsimavg

        result = {"streamflow": streamflow}

        # 注意：不需要在这里截断 warmup
        # 当 warm_up_states=False 时，_run_model 已经只返回训练段（不含 warmup）
        # 当 warm_up_states=True 时，_run_model 返回完整序列（含 warmup），此时
        # target 也包含 warmup 段，所以也不需要在这里截断

        return result
