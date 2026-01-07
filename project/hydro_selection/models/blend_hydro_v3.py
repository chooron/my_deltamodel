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
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma
from dmg.models.neural_networks.layers.moe import MoeLayer

from project.hydro_selection.models.layers import hydro_core


class HybridStateNorm(nn.Module):
    def __init__(self, total_dim, meteo_dim=3):
        """
        参数:
        total_dim: 输入的总维度 (气象 + 所有状态)
        meteo_dim: 气象数据的维度 (前几列不处理)，默认为 3 (P, T, PET)
        """
        super().__init__()
        self.meteo_dim = meteo_dim
        self.state_dim = total_dim - meteo_dim

        # 这里的 LayerNorm 只针对后面的状态部分
        self.state_ln = nn.LayerNorm(self.state_dim)

    def forward(self, x):
        """
        输入 x: [Batch, Time, Total_Dim] 或 [Batch, Total_Dim]
        """
        # 1. 切片分离 (Slicing)
        x_meteo = x[..., : self.meteo_dim]  # 前3列：保持原样 (已全局归一化)
        x_states = x[..., self.meteo_dim :]  # 后面的列：需要处理的物理状态

        # 2. Log1p 变换 (Log Transformation)
        # torch.relu 是为了兜底，防止物理模型算崩了出现负数导致 log 报错
        # log1p(x) = log(x + 1)，把 0~10000 压缩到 0~9 左右
        x_states_log = torch.log1p(torch.relu(x_states))

        # 3. LayerNorm 归一化 (Normalization)
        # 将压缩后的状态拉到 均值0 方差1
        x_states_norm = self.state_ln(x_states_log)

        # 4. 拼接回原形状 (Concatenation)
        out = torch.cat([x_meteo, x_states_norm], dim=-1)

        return out


class BlendHydroV3(nn.Module):
    """Blend of multiple JIT hydrological cores.

    这个v2.1版本是通过流域静态属性来预测模型的组合权重,实现每个模型的加权求和,考虑预测16个加权结果,后续再考虑4个模型集成后的加权结果

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
    MODEL_STATES_NUM = hydro_core.MODEL_STATES_NUM

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
        self.activate = F.sigmoid

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

        # 直接初始化 MoE 层
        self.use_moe = True
        self.moe_embed_dim = 128
        self.moe_smoothing_k = 7
        # 最大时间步长，需要足够大以支持各种输入长度
        self.moe_target_points = 730
        self.moe_weights = None

        # 3 (x_norm features) + 5 (padded states) = 8
        self.norm_layer = HybridStateNorm(total_dim=8, meteo_dim=3).to(
            self.device
        )
        self.moe_layer = MoeLayer(
            enc_in=8,  # 3 forcing + 5 states
            num_experts=16,  # todo Fixed number of experts
            target_points=self.moe_target_points,
            embed_dim=self.moe_embed_dim,
            smoothing_k=self.moe_smoothing_k,
            num_layers=2,
            num_heads=4,
            causal=True,
        ).to(self.device)

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
            self.n_models = len(selected)

        # New: compilation setting (jit, torch.compile, none)
        self.compile_type = config.get("compile_type", "jit")

    def _set_parameters(self) -> None:
        self.routing_param_names = list(self.routing_parameter_bounds.keys())

        # We treat the configured nmul as the total target budget
        self.nmul = max(1, self.nmul // self.n_models)

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
        self,
        params: torch.Tensor,
        names: List[str],
        bounds: Dict[str, List[float]],
    ) -> Dict[str, torch.Tensor]:
        return {
            name: change_param_range(params[:, i, :], bounds[name])
            for i, name in enumerate(names)
        }

    def _descale_dynamic_params(
        self,
        params: torch.Tensor,
        names: List[str],
        bounds: Dict[str, List[float]],
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

    def _descale_routing_params(
        self, params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            name: change_param_range(
                params[:, i], self.routing_parameter_bounds[name]
            )
            for i, name in enumerate(self.routing_parameter_bounds.keys())
        }

    def _apply_moe_weighting(
        self, Qsimmu: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """MoE 动态加权

        Parameters
        ----------
        Qsimmu : torch.Tensor
            各专家模型的流量输出，形状: (T, B, E)
            T=时间步, B=流域数, E=专家数(nmul)
        n_steps : int
            时间步数

        Returns
        -------
        torch.Tensor
            加权后的流量，形状: (T, B)
        """
        if self.use_moe and self.nmul > 1:
            # MoeLayer 期望输入形状: (Seq, Batch, NumExperts, Feature)
            # Qsimmu 形状: (T, B, E) -> 需要添加 Feature 维度
            moe_input = Qsimmu.unsqueeze(-1)  # (T, B, E, 1)
            # 获取门控权重，形状: (T, B, E, 1)
            gating_weights = self.moe_layer(moe_input)
            # 保存权重（去掉最后一维）用于分析
            self.moe_weights = gating_weights.squeeze(-1)  # (T, B, E)
            # 加权求和: (T, B, E, 1) * (T, B, E, 1) -> sum over E -> (T, B, 1) -> (T, B)
            return (gating_weights * moe_input).sum(dim=2).squeeze(-1)
        else:
            self.moe_weights = None
            return Qsimmu.mean(-1)

    def unpack_parameters(
        self, parameters: Tuple[Union[None, torch.Tensor], torch.Tensor]
    ) -> Tuple[
        Dict[str, Union[None, torch.Tensor]],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """Split concatenated parameter tensors into per-model blocks."""
        dy_count_total = sum(
            len(self.dynamic_params.get(m, [])) for m in self.model_order
        )
        static_counts = [
            len(self.parameter_bounds_by_model[m])
            - len(self.dynamic_params.get(m, []))
            for m in self.model_order
        ]

        raw_phy_dy, raw_phy_static = parameters

        phy_dy_dict: Dict[str, Optional[torch.Tensor]] = {
            m: None for m in self.model_order
        }
        if raw_phy_dy is not None:
            phy_dy = raw_phy_dy.view(
                raw_phy_dy.shape[0],
                raw_phy_dy.shape[1],
                dy_count_total,
                self.nmul,
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
        static_block = self.activate(static_block)
        routing_block = self.activate(routing_block)

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
        x_norm = x_dict["x_nn_norm"]
        x_norm = x_norm.unsqueeze(2).repeat(
            1, 1, 4, 1
        )  # TODO fixed to 4 of each model
        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        phy_dy_dict, phy_static_dict, phy_route = self.unpack_parameters(
            parameters
        )
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
        per_model_states_list = []
        # 2. 并行执行 (Explicit CUDA Streams)
        for model_name in self.model_order:
            current_params = self.get_model_params(
                model_name, phy_dy_dict, phy_static_dict
            )
            stream = self.streams[model_name]

            with torch.cuda.stream(stream):
                # 获取该模型对应的 Kernel
                kernel = self.kernels[model_name]
                outputs = kernel(
                    P, T, PET, **current_params, nearzero=self.nearzero
                )

                raw_states = outputs[1 : self.MODEL_STATES_NUM[model_name] + 1]
                # 将状态堆叠为单个张量，并补齐到 5 个通道
                if isinstance(raw_states, torch.Tensor):
                    model_states = raw_states
                else:
                    model_states = torch.stack(list(raw_states), dim=-1)
                if model_states.shape[-1] < 5:
                    pad_size = 5 - model_states.shape[-1]
                    pad_shape = (*model_states.shape[:-1], pad_size)
                    pad = torch.zeros(
                        pad_shape,
                        device=model_states.device,
                        dtype=model_states.dtype,
                    )
                    model_states = torch.cat([model_states, pad], dim=-1)

                norm_model_states = self.norm_layer(
                    torch.cat([x_norm, model_states], dim=-1)
                )

                per_model_states_list.append(norm_model_states)
                per_model_qsim[model_name] = outputs[0]

        # 等待所有流完成
        torch.cuda.synchronize(device=self.device)
        # 形状: (T, B, num_models, 8)，按模型取均值得到固定 5 维状态
        stacked_states = torch.cat(per_model_states_list, dim=2)
        model_dyn_weights = self.moe_layer(stacked_states).squeeze(
            -1
        )  # (T, B, E)
        self.moe_weights = model_dyn_weights
        # ==========================================
        # 并行计算结束，per_model_qsim 已填充完毕
        # ==========================================
        # Blend: Concatenate experts from all models and average the whole pool
        all_q = torch.cat([per_model_qsim[m] for m in self.model_order], dim=-1)
        blend_q = (model_dyn_weights * all_q).sum(dim=2).squeeze(-1)

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
