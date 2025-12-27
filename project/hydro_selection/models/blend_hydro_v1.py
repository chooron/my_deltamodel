"""
Blend hydrological model that wraps multiple JIT-accelerated core models
(HBV, SHM, EXP-HYDRO, HyMod) and averages their simulated runoff before
routing.

This follows the same parameter handling pattern as `Hbv` but extends it to
multiple conceptual models. Parameters for every sub-model are provided in a
single concatenated tensor tuple, and each sub-model runs with its own JIT
loop. Their pre-routing discharges are averaged and then passed through a
unit-hydrograph routing module.

Author: GitHub Copilot (GPT-5.1-Codex-Max preview)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from project.hydro_selection.models.layers import hydro_core


class BlendHydroV1(nn.Module):
    """Blend of multiple JIT hydrological cores.

    The model consumes a pair of tensors `parameters = (raw_phy_dy, raw_phy_static)`
    produced by an upstream network, mirroring the interface of `Hbv`. The
    parameters for all sub-models are concatenated in a fixed order, followed by
    routing parameters. Dynamic parameters are supported per-model via the
    `dynamic_params` config field.

    Config keys (all optional):
    - warm_up: int
    - warm_up_states: bool
    - variables: list[str] (default ["prcp", "tmean", "pet"])
    - nearzero: float
    - nmul: int (ensemble members per model)
    - dynamic_params: dict[str, list[str]] keyed by model name
    - selected_models: list[str] (e.g. ["HBV", "EXPHYDRO"])
    - hymod_nq: int (number of Nash cascade reservoirs)
    - routing: bool (kept for API parity; routing is always applied)
    """

    HBV_BOUNDS = hydro_core.HBV_PARAMS_BOUNDS
    SHM_BOUNDS = hydro_core.SHM_PARAMS_BOUNDS
    HYMOD_BOUNDS = hydro_core.HYMOD_PARAMS_BOUNDS
    EXPHYDRO_BOUNDS = hydro_core.EXPHYDRO_PARAMS_BOUNDS

    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.name = "BlendHydro"
        self.config = config or {}
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.dynamic_params: Dict[str, List[str]] = {}

        self.parameter_bounds_by_model = {
            "HBV": self.HBV_BOUNDS,
            "SHM": self.SHM_BOUNDS,
            "EXPHYDRO": self.EXPHYDRO_BOUNDS,
            "HYMOD": self.HYMOD_BOUNDS,
        }
        self.routing_parameter_bounds = self.ROUTING_BOUNDS
        self.all_supported_models = ["HBV", "SHM", "EXPHYDRO", "HYMOD"]
        self.model_order = list(self.all_supported_models)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if config is not None:
            self._load_config(config)

        self._set_parameters()
        self._setup_kernels()

    # ------------------------------------------------------------------
    # Config & parameter bookkeeping
    # ------------------------------------------------------------------
    def _load_config(self, config: Dict[str, Any]) -> None:
        simple_attrs = [
            "warm_up",
            "warm_up_states",
            "variables",
            "nearzero",
            "nmul",
        ]
        for attr in simple_attrs:
            if attr in config:
                setattr(self, attr, config[attr])

        if "dynamic_params" in config:
            # Expecting dict keyed by model name, e.g. {"HBV": ["parK0", ...]}
            self.dynamic_params = config.get("dynamic_params", {})
        
        if "selected_models" in config:
            selected = [m.upper() for m in config["selected_models"]]
            # Filter to only supported ones and maintain internal order if possible
            order = [m for m in self.all_supported_models if m in selected]
            if order:
                self.model_order = order
            print(self.model_order)
        
        # New: compilation setting (jit, torch.compile, none)
        self.compile_type = config.get("compile_type", "jit")

    def _set_parameters(self) -> None:
        self.routing_param_names = list(self.routing_parameter_bounds.keys())

        # Distribute nmul budget across models
        n_models = len(self.model_order)
        # We treat the configured nmul as the total target budget
        self.nmul = max(1, self.nmul // n_models)

        # Build per-model param name lists
        self.phy_param_names_by_model: Dict[str, List[str]] = {}
        dy_total = 0
        static_total = 0
        for name in self.model_order:
            bounds = self.parameter_bounds_by_model[name]
            self.phy_param_names_by_model[name] = list(bounds.keys())
            dy_count = len(self.dynamic_params.get(name, []))
            static_count = len(bounds) - dy_count
            dy_total += dy_count
            static_total += static_count

        self.learnable_param_count1 = dy_total * self.nmul
        self.learnable_param_count2 = static_total * self.nmul + len(
            self.routing_param_names
        )
        self.learnable_param_count = (
            self.learnable_param_count1 + self.learnable_param_count2
        )

    def _setup_kernels(self) -> None:
        """Setup model kernels based on compile_type."""
        # Map model names to their corresponding functions in hydro_core
        kernel_map = {
            "HBV": hydro_core.hbv_timestep_loop,
            "SHM": hydro_core.shm_timestep_loop,
            "EXPHYDRO": hydro_core.exphydro_timestep_loop,
            "HYMOD": hydro_core.hymod_timestep_loop,
        }

        # Apply compilation or keep as is
        self.kernels: Dict[str, Any] = {}
        for model_name in self.model_order:
            kernel_fn = kernel_map[model_name]
            if self.compile_type == "torch.compile":
                self.kernels[model_name] = torch.compile(kernel_fn)
            elif self.compile_type == "jit":
                # If functions in hydro_core are already decorated with @torch.jit.script,
                # they are already JITed. If not, we script them here.
                # Assuming hydro_core functions are NOT pre-decorated for maximum flexibility:
                self.kernels[model_name] = torch.jit.script(kernel_fn)
            else:
                # "none" mode: use raw functions
                self.kernels[model_name] = kernel_fn

        # Setup CUDA streams for parallel execution (strictly CUDA)
        self.streams = {
            m: torch.cuda.Stream(device=self.device) for m in self.model_order
        }

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    def _descale_params(
        self, params: torch.Tensor, names: List[str], bounds: Dict[str, List[float]]
    ) -> Dict[str, torch.Tensor]:
        return {
            name: change_param_range(params[:, i, :], bounds[name])
            for i, name in enumerate(names)
        }

    def _descale_dynamic_params(
        self, params: torch.Tensor, names: List[str], bounds: Dict[str, List[float]]
    ) -> Dict[str, torch.Tensor]:
        n_steps, n_grid = params.shape[:2]
        pmat = torch.ones([1, n_grid, 1], device=self.device)
        result = {}
        for i, name in enumerate(names):
            static_par = (
                params[-1, :, i, :].unsqueeze(0).expand(n_steps, -1, -1)
            )
            dynamic_par = params[:, :, i, :]
            mask = torch.bernoulli(pmat).detach_()
            combined = dynamic_par * (1 - mask) + static_par * mask
            result[name] = change_param_range(combined, bounds[name])
        return result

    def _descale_routing_params(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            name: change_param_range(
                params[:, i], self.routing_parameter_bounds[name]
            )
            for i, name in enumerate(self.routing_parameter_bounds.keys())
        }

    def unpack_parameters(
        self, parameters: Tuple[Union[None, torch.Tensor], torch.Tensor]
    ) -> Tuple[Dict[str, Union[None, torch.Tensor]], Dict[str, torch.Tensor], torch.Tensor]:
        """Split concatenated parameter tensors into per-model blocks."""
        dy_count_total = sum(len(self.dynamic_params.get(m, [])) for m in self.model_order)
        static_counts = [
            len(self.parameter_bounds_by_model[m])
            - len(self.dynamic_params.get(m, []))
            for m in self.model_order
        ]

        raw_phy_dy, raw_phy_static = parameters

        phy_dy_dict: Dict[str, Optional[torch.Tensor]] = {m: None for m in self.model_order}
        if raw_phy_dy is not None:
            phy_dy = raw_phy_dy.view(
                raw_phy_dy.shape[0], raw_phy_dy.shape[1], dy_count_total, self.nmul
            )
            offset = 0
            for m in self.model_order:
                dy_count = len(self.dynamic_params.get(m, []))
                if dy_count > 0:
                    phy_dy_dict[m] = phy_dy[:, :, offset : offset + dy_count, :]
                offset += dy_count

        # static + routing
        total_static = sum(static_counts)
        static_block = raw_phy_static[:, : total_static * self.nmul]
        routing_block = raw_phy_static[:, total_static * self.nmul :]

        phy_static = static_block.view(
            static_block.shape[0], total_static, self.nmul
        )

        phy_static_dict: Dict[str, torch.Tensor] = {}
        offset = 0
        for m, sc in zip(self.model_order, static_counts):
            phy_static_dict[m] = phy_static[:, offset : offset + sc, :]
            offset += sc

        return phy_dy_dict, phy_static_dict, routing_block

    # Prepare parameters per model (descale dynamic/static when available)
    def get_model_params(
        self,
        m: str,
        phy_dy_dict: Dict[str, Optional[torch.Tensor]],
        phy_static_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return a merged parameter dict (dynamic overrides static)."""
        bounds = self.parameter_bounds_by_model[m]
        dy_names = self.dynamic_params.get(m, [])
        static_names = [p for p in bounds.keys() if p not in dy_names]

        dy_params: Dict[str, torch.Tensor] = {}
        if phy_dy_dict[m] is not None and len(dy_names) > 0:
            dy_p = phy_dy_dict[m]
            if dy_p is not None:
                dy_params = self._descale_dynamic_params(dy_p, dy_names, bounds)

        static_params = self._descale_params(
            phy_static_dict[m], static_names, bounds
        )

        # merged: dynamic (if present) else static
        merged: Dict[str, torch.Tensor] = {}
        for name in bounds.keys():
            if name in dy_params:
                merged[name] = dy_params[name]
            else:
                merged[name] = static_params[name]
        return merged


    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _apply_routing(
        self, Qsim: torch.Tensor, n_steps: int, n_grid: int
    ) -> torch.Tensor:
        UH = uh_gamma(
            self.routing_param_dict["rout_a"].repeat(n_steps, 1).unsqueeze(-1),
            self.routing_param_dict["rout_b"].repeat(n_steps, 1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])

        rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
        Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
        return Qsrout

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Union[None, torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x = x_dict["x_phy"]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        phy_dy_dict, phy_static_dict, phy_route = self.unpack_parameters(parameters)
        self.routing_param_dict = self._descale_routing_params(phy_route)

        n_steps, n_grid = x.shape[:2]

        # Prepare forcing
        P = (
            x[:, :, self.variables.index("prcp")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        T = (
            x[:, :, self.variables.index("tmean")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        PET = (
            x[:, :, self.variables.index("pet")]
            .unsqueeze(-1)
            .repeat(1, 1, self.nmul)
        )

        per_model_qsim: Dict[str, torch.Tensor] = {}

        # 2. 并行执行 (Explicit CUDA Streams)
        for model_name in self.model_order:
            current_params = self.get_model_params(
                model_name, phy_dy_dict, phy_static_dict
            )
            stream = self.streams[model_name]

            with torch.cuda.stream(stream):
                # 获取该模型对应的 Kernel
                kernel = self.kernels[model_name]
                q = kernel(
                    P, T, PET, **current_params, nearzero=self.nearzero
                )[0]
                per_model_qsim[model_name] = q

        # 等待所有流完成
        torch.cuda.synchronize(device=self.device)

        # ==========================================
        # 并行计算结束，per_model_qsim 已填充完毕
        # ==========================================

        # Blend: Concatenate experts from all models and average the whole pool
        all_q = torch.cat(
            [per_model_qsim[m] for m in self.model_order], dim=-1
        )
        blend_q = all_q.mean(-1)

        Qrouted = self._apply_routing(blend_q, n_steps, n_grid)

        result: Dict[str, torch.Tensor] = {
            "streamflow": Qrouted,
            "blend_prerouting": blend_q,
        }

        # attach per-model summary results for inspection
        for name, q in per_model_qsim.items():
            result[f"{name.lower()}_prerouting"] = q.mean(-1)

        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
