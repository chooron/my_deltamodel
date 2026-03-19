"""
Blend hydrological model (V1 - Ultra Optimized)
- 统一时间循环：所有模型在同一个for循环中执行
- 仅支持静态参数：移除所有动态参数处理逻辑
- torch.compile加速：每个step函数使用torch.compile编译
- 高效状态管理：预分配所有状态张量

核心改进：
1. 单一for循环替代多模型并行流
2. 去除动态参数相关的所有分支和计算
3. 编译step函数以提升性能
4. 保持forward接口完全兼容

Author: chooron
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from project.hydro_selection.models.layers import hydro_core


class BlendHydroV1(nn.Module):
    """优化版Blend水文模型：统一时间循环 + 纯静态参数 + torch.compile"""

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

        self.name = "BlendHydroV4"
        self.config = config or {}
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
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
        self._setup_compiled_kernels()

    def _load_config(self, config: Dict[str, Any]) -> None:
        """加载配置（去除动态参数相关）"""
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

        if "selected_models" in config:
            selected = [m.upper() for m in config["selected_models"]]
            order = [m for m in self.all_supported_models if m in selected]
            if order:
                self.model_order = order

    def _set_parameters(self) -> None:
        """设置参数名称（仅静态参数）"""
        self.routing_param_names = list(self.routing_parameter_bounds.keys())

        # 每个模型的参数名
        self.phy_param_names_by_model: Dict[str, List[str]] = {}
        total_params = 0
        for name in self.model_order:
            bounds = self.parameter_bounds_by_model[name]
            param_names = list(bounds.keys())
            self.phy_param_names_by_model[name] = param_names
            total_params += len(param_names)

        # 总参数数量：模型参数 * nmul + 路由参数
        self.learnable_param_count = total_params * self.nmul + len(
            self.routing_param_names
        )

    def _setup_compiled_kernels(self) -> None:
        """编译每个模型的step函数"""
        self.compiled_steps = {}

        # 使用torch.compile编译每个step函数
        self.compiled_steps["HBV"] = torch.compile(hydro_core.hbv_step)
        self.compiled_steps["SHM"] = torch.compile(hydro_core.shm_step)
        self.compiled_steps["EXPHYDRO"] = torch.compile(
            hydro_core.exphydro_step
        )
        self.compiled_steps["HYMOD"] = torch.compile(hydro_core.hymod_step)

    def _descale_params(
        self,
        params: torch.Tensor,
        names: List[str],
        bounds: Dict[str, List[float]],
    ) -> Dict[str, torch.Tensor]:
        """反归一化参数"""
        return {
            name: change_param_range(params[:, i, :], bounds[name])
            for i, name in enumerate(names)
        }

    def _descale_routing_params(
        self, params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """反归一化路由参数"""
        return {
            name: change_param_range(
                params[:, i], self.routing_parameter_bounds[name]
            )
            for i, name in enumerate(self.routing_param_names)
        }

    def unpack_parameters(
        self, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        解包参数（从MultiHeadNet字典输出提取）

        Args:
            parameters: 字典，包含各模型参数和GAMMA_UH路由参数
                例如: {"HBV": [B, 14*nmul], "SHM": [B, 7*nmul], 
                      "GAMMA_UH": [B, 2], ...}

        Returns:
            - phy_static_dict: 各模型的静态参数字典 (已reshape)
            - routing_block: 路由参数张量
        """
        # 提取路由参数
        routing_block = self.activate(parameters["GAMMA_UH"])
        
        # 提取并处理各模型参数
        phy_static_dict: Dict[str, torch.Tensor] = {}
        for model_name in self.model_order:
            if model_name in parameters:
                # 应用sigmoid激活
                raw_params = self.activate(parameters[model_name])
                # Reshape为 [batch, n_params, nmul]
                n_params = len(self.phy_param_names_by_model[model_name])
                phy_static_dict[model_name] = raw_params.view(
                    raw_params.shape[0], n_params, self.nmul
                )

        return phy_static_dict, routing_block

    def get_model_params(
        self, model_name: str, phy_static_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """获取单个模型的参数字典（仅静态参数）"""
        bounds = self.parameter_bounds_by_model[model_name]
        param_names = self.phy_param_names_by_model[model_name]

        return self._descale_params(
            phy_static_dict[model_name], param_names, bounds
        )

    def _apply_routing(
        self, Qsim: torch.Tensor, n_steps: int, n_grid: int
    ) -> torch.Tensor:
        """应用单位线路由"""
        UH = uh_gamma(
            self.routing_param_dict["rout_a"].repeat(n_steps, 1).unsqueeze(-1),
            self.routing_param_dict["rout_b"].repeat(n_steps, 1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])

        rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
        Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
        return Qsrout

    def _initialize_states(
        self, n_steps: int, n_grid: int
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """初始化所有模型的状态变量"""
        states = {}

        # HBV: 5个状态
        if "HBV" in self.model_order:
            states["HBV"] = {
                "S1": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S2": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S3": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S4": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S5": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
            }

        # SHM: 4个状态（无雪层存储）
        if "SHM" in self.model_order:
            states["SHM"] = {
                "sf": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "su": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "si": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "sb": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
            }

        # EXPHYDRO: 2个状态
        if "EXPHYDRO" in self.model_order:
            states["EXPHYDRO"] = {
                "soil_storage": torch.zeros(
                    n_grid, self.nmul, device=self.device
                )
                + self.nearzero,
                "snow_storage": torch.zeros(
                    n_grid, self.nmul, device=self.device
                )
                + self.nearzero,
            }

        # HYMOD: 5个状态
        if "HYMOD" in self.model_order:
            states["HYMOD"] = {
                "S1": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S2": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S3": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S4": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
                "S5": torch.zeros(n_grid, self.nmul, device=self.device)
                + self.nearzero,
            }

        return states

    def _unified_timestep_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        params_dict: Dict[str, Dict[str, torch.Tensor]],
        n_steps: int,
        n_grid: int,
    ) -> Dict[str, torch.Tensor]:
        """
        统一的时间循环：在单个循环内执行所有模型

        Args:
            P: [n_steps, n_grid, nmul]
            T: [n_steps, n_grid, nmul]
            PET: [n_steps, n_grid, nmul]
            params_dict: 各模型的参数字典
            n_steps: 时间步数
            n_grid: 网格数

        Returns:
            各模型的输出时间序列
        """
        # 初始化状态
        states = self._initialize_states(n_steps, n_grid)

        # 预分配输出张量
        outputs = {}
        for model_name in self.model_order:
            outputs[model_name] = torch.zeros(
                n_steps, n_grid, self.nmul, device=self.device
            )

        # ===== 核心：统一时间循环 =====
        for t in range(n_steps):
            P_t = P[t]
            T_t = T[t]
            PET_t = PET[t]

            # HBV模型
            if "HBV" in self.model_order:
                params = params_dict["HBV"]
                hbv_step_fn = self.compiled_steps["HBV"]

                Q, Ea, S1, S2, S3, S4, S5 = hbv_step_fn(
                    P_t,
                    T_t,
                    PET_t,
                    params["tt"],
                    params["tti"],
                    params["ttm"],
                    params["cfr"],
                    params["cfmax"],
                    params["whc"],
                    params["cflux"],
                    params["fc"],
                    params["lp"],
                    params["beta"],
                    params["k0"],
                    params["alpha"],
                    params["perc"],
                    params["k1"],
                    states["HBV"]["S1"],
                    states["HBV"]["S2"],
                    states["HBV"]["S3"],
                    states["HBV"]["S4"],
                    states["HBV"]["S5"],
                    self.nearzero,
                )

                outputs["HBV"][t] = Q
                states["HBV"].update(
                    {"S1": S1, "S2": S2, "S3": S3, "S4": S4, "S5": S5}
                )

            # SHM模型
            if "SHM" in self.model_order:
                params = params_dict["SHM"]
                shm_step_fn = self.compiled_steps["SHM"]

                Q, ret, su, sf, si, sb = shm_step_fn(
                    P_t,
                    T_t,
                    PET_t,
                    params["f_thr"],
                    params["sumax"],
                    params["beta"],
                    params["perc"],
                    params["kf"],
                    params["ki"],
                    params["kb"],
                    self.nearzero,
                    states["SHM"]["su"],
                    states["SHM"]["sf"],
                    states["SHM"]["si"],
                    states["SHM"]["sb"],
                )

                outputs["SHM"][t] = Q
                states["SHM"].update({"sf": sf, "su": su, "si": si, "sb": sb})

            # EXPHYDRO模型
            if "EXPHYDRO" in self.model_order:
                params = params_dict["EXPHYDRO"]
                exphydro_step_fn = self.compiled_steps["EXPHYDRO"]

                q, et, soil_storage, snow_storage, melt = exphydro_step_fn(
                    P_t,
                    T_t,
                    PET_t,
                    params["f"],
                    params["ddf"],
                    params["smax"],
                    params["qmax"],
                    params["mint"],
                    params["maxt"],
                    self.nearzero,
                    states["EXPHYDRO"]["soil_storage"],
                    states["EXPHYDRO"]["snow_storage"],
                )

                outputs["EXPHYDRO"][t] = q
                states["EXPHYDRO"].update(
                    {"soil_storage": soil_storage, "snow_storage": snow_storage}
                )

            # HYMOD模型
            if "HYMOD" in self.model_order:
                params = params_dict["HYMOD"]
                hymod_step_fn = self.compiled_steps["HYMOD"]

                Q, Ea, S1, S2, S3, S4, S5 = hymod_step_fn(
                    P_t,
                    T_t,
                    PET_t,
                    params["smax"],
                    params["b_exp"],
                    params["a_split"],
                    params["kf"],
                    params["ks"],
                    states["HYMOD"]["S1"],
                    states["HYMOD"]["S2"],
                    states["HYMOD"]["S3"],
                    states["HYMOD"]["S4"],
                    states["HYMOD"]["S5"],
                    self.nearzero,
                )

                outputs["HYMOD"][t] = Q
                states["HYMOD"].update(
                    {"S1": S1, "S2": S2, "S3": S3, "S4": S4, "S5": S5}
                )

        return outputs

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x_dict: 包含'x_phy'的字典 [n_steps, n_grid, n_vars]
            parameters: MultiHeadNet输出的字典
                       {"HBV": tensor, "SHM": tensor, "GAMMA_UH": tensor, ...}

        Returns:
            包含'streamflow'等输出的字典
        """
        x = x_dict["x_phy"]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        # 解包参数（从字典提取）
        phy_static_dict, phy_route = self.unpack_parameters(parameters)
        self.routing_param_dict = self._descale_routing_params(phy_route)

        n_steps, n_grid = x.shape[:2]

        # 准备驱动数据
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
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )

        # 获取各模型参数
        params_dict = {}
        for model_name in self.model_order:
            params_dict[model_name] = self.get_model_params(
                model_name, phy_static_dict
            )

        # 统一时间循环执行所有模型
        model_outputs = self._unified_timestep_loop(
            P, T, PET, params_dict, n_steps, n_grid
        )

        # 集成所有模型输出（平均）
        all_q = torch.cat([model_outputs[m] for m in self.model_order], dim=-1)
        blend_q = all_q.mean(-1)

        # 应用路由
        Qrouted = self._apply_routing(blend_q, n_steps, n_grid)

        # 构造返回字典
        result: Dict[str, torch.Tensor] = {
            "streamflow": Qrouted,
            "blend_prerouting": blend_q,
            "target": x_dict["target"]
        }

        # 添加各模型的单独输出
        for name, output in model_outputs.items():
            result[f"{name.lower()}_prerouting"] = output.mean(-1)

        # 截断warmup
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
