"""
Blend hydrological model (V2 - LSTM Gating Enhanced)

相比V1的核心改进：
1. **物理-数据混合门控机制**：
   - 新增 PhysicsGatingNetwork (LSTM)
   - 基于气象驱动、模型状态、流域属性动态计算专家权重
   - 替代原始的简单平均融合策略

2. **状态感知融合**：
   - 时间循环同时输出流量和状态序列
   - 提取各模型核心土壤湿度状态（HBV:S3, SHM:su, EXPHYDRO:soil, HYMOD:S1）
   - 状态归一化后作为门控网络输入

3. **保留V1的高效特性**：
   - 统一时间循环
   - 纯静态参数
   - torch.compile加速
   - 接口兼容

Architecture:
    Input (P,T,PET,Attr)
      ↓
    Physics Layer (4 Models in Unified Loop)
      ↓
    [Q₁, Q₂, Q₃, Q₄] + [S₁, S₂, S₃, S₄]
      ↓
    Gating Network (LSTM)
      ↓
    Weights [w₁, w₂, w₃, w₄]
      ↓
    Q_blend = Σ(wᵢ × Qᵢ)
      ↓
    Routing
      ↓
    Streamflow

Author: chooron
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from project.hydro_selection.models.layers import hydro_core


class PhysicsGatingNetwork(nn.Module):
    """
    基于物理状态和气象驱动的LSTM门控网络 (输出模型级权重)

    核心机制：
        接收 (气象 + 所有Nmul的物理状态 + 静态属性) 作为输入，
        输出 NumModels 个权重（对应4个模型），并在该维度上进行 Softmax。
        每个模型的多个参数组结果会先求平均，然后用模型权重加权。

    输入：
        - x_dynamic: [Time, Batch, Dyn_Dim]
          其中 Dyn_Dim = 3(气象) + (Nmul * NumModels)(所有专家的状态)
        - x_static:  [Time, Batch, Stat_Dim] 或 [Batch, Stat_Dim]

    输出：
        - weights: [Time, Batch, NumModels]，Softmax归一化
    """

    def __init__(
        self,
        input_dim: int,  # 总输入维度：3 + (Nmul*NumModels) + StaticAttr
        num_models: int,  # 模型数量（例如4）
        hidden_dim: int = 128,  # 建议增大隐藏层，因为输入包含了大量状态特征
        num_layers: int = 2,  # LSTM层数
        dropout: float = 0.2,  # Dropout
    ):
        super().__init__()

        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层：处理时序依赖
        # 输入维度可能很大 (例如 3 + 100*4 + 10 = 413)，LSTM需要足够宽来提取特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,  # 输入保持：[Time, Batch, Feature]
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 投影层：将 LSTM 特征映射到模型权重空间
        self.fc_out = nn.Linear(hidden_dim, num_models)

        # 可选：添加 LayerNorm 稳定大维度输入的训练
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_dynamic: torch.Tensor,  # [Time, Batch, DynamicDim] (含状态)
        x_static: torch.Tensor,  # [Time, Batch, StaticDim] (含属性)
    ) -> torch.Tensor:
        """
        计算各时间步的模型权重

        Returns:
            weights: [Time, Batch, NumModels]，满足 Sum(dim=-1) = 1
        """
        # 1. 对齐静态特征的时间维度
        if x_static.dim() == 2:
            # [Batch, Static] -> [Time, Batch, Static]
            x_static = x_static.unsqueeze(0).expand(x_dynamic.shape[0], -1, -1)

        # 2. 特征拼接
        # 此时 input 包含：气象驱动 + 所有参数组的干湿状态 + 流域属性
        x_input = torch.cat([x_dynamic, x_static], dim=-1)

        # 3. LSTM前向传播
        lstm_out, _ = self.lstm(x_input)  # [Time, Batch, Hidden]

        # 4. (可选) 层归一化，有助于收敛
        lstm_out = self.ln(lstm_out)

        # 5. 映射到 logits
        logits = self.fc_out(lstm_out)  # [Time, Batch, Total_Experts]

        # 6. 计算权重 (Softmax)
        # dim=-1 使得所有模型的权重加起来等于 1
        weights = F.softmax(logits, dim=-1)

        return weights


class BlendHydroV2(nn.Module):
    """优化版Blend水文模型：统一时间循环 + 纯静态参数 + torch.compile"""

    HBV_BOUNDS = hydro_core.HBV_PARAMS_BOUNDS
    SHM_BOUNDS = hydro_core.SHM_PARAMS_BOUNDS
    HYMOD_BOUNDS = hydro_core.HYMOD_PARAMS_BOUNDS
    EXPHYDRO_BOUNDS = hydro_core.EXPHYDRO_PARAMS_BOUNDS
    # 每个模型独立的路由参数边界
    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}
    ROUTING_PARAM_NAMES = ["rout_a", "rout_b"]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.name = "BlendHydroV2"
        self.config = config or {}
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.num_attributes = 0
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

        # 初始化LSTM门控网络
        # 输入维度：3(气象) + Nmul*NumModels(状态) + static_attr_dim(属性)
        # 输出维度：NumModels（只预测模型级权重）
        gating_input_dim = (
            3 + self.nmul * len(self.model_order) + self.num_attributes
        )
        print(
            f"Gating input dim: {gating_input_dim}, Output models: {len(self.model_order)}"
        )
        self.gating_net = PhysicsGatingNetwork(
            input_dim=gating_input_dim,
            num_models=len(self.model_order),
            hidden_dim=64,
            num_layers=2,
        ).to(self.device)

    def _load_config(self, config: Dict[str, Any]) -> None:
        """加载配置（去除动态参数相关）"""
        simple_attrs = [
            "warm_up",
            "warm_up_states",
            "variables",
            "nearzero",
            "nmul",
            "num_attributes",
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

        # 总参数数量：模型参数 * nmul + 每个模型的路由参数
        # 每个模型有2个路由参数 (rout_a, rout_b)
        total_routing_params = len(self.routing_param_names) * len(
            self.model_order
        )
        self.learnable_param_count = (
            total_params * self.nmul + total_routing_params
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
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """反归一化路由参数（每个模型独立）

        Args:
            params: [batch, num_models * 2] 所有模型的路由参数

        Returns:
            嵌套字典 {model_name: {param_name: tensor}}
        """
        routing_dict = {}
        num_routing_params = len(self.routing_param_names)

        for model_idx, model_name in enumerate(self.model_order):
            model_routing = {}
            for param_idx, param_name in enumerate(self.routing_param_names):
                # 计算在params中的索引位置
                global_idx = model_idx * num_routing_params + param_idx
                model_routing[param_name] = change_param_range(
                    params[:, global_idx],
                    self.routing_parameter_bounds[param_name],
                )
            routing_dict[model_name] = model_routing

        return routing_dict

    def unpack_parameters(
        self, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        解包参数（从MultiHeadNet字典输出提取）

        Args:
            parameters: 字典，包含各模型参数和GAMMA_UH路由参数
                例如: {"HBV": [B, 14*nmul], "SHM": [B, 7*nmul],
                      "GAMMA_UH": [B, 8], ...}  # 8 = 4个模型 × 2个路由参数

        Returns:
            - phy_static_dict: 各模型的静态参数字典 (已reshape)
            - routing_block: 所有模型的路由参数张量 [B, num_models*2]
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
        self, Qsim: torch.Tensor, model_name: str, n_steps: int, n_grid: int
    ) -> torch.Tensor:
        """应用单位线路由（使用模型特定的参数）

        Args:
            Qsim: [n_steps, n_grid] 产流量
            model_name: 模型名称，用于获取对应的路由参数
            n_steps: 时间步数
            n_grid: 网格数

        Returns:
            Qsrout: [n_steps, n_grid] 路由后的流量
        """
        # 获取该模型的路由参数
        model_routing = self.routing_param_dict[model_name]

        UH = uh_gamma(
            model_routing["rout_a"].repeat(n_steps, 1).unsqueeze(-1),
            model_routing["rout_b"].repeat(n_steps, 1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])

        rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
        Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
        return Qsrout

    def _normalize_states(
        self,
        raw_states: torch.Tensor,
        params_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        归一化各模型的核心状态到 [0, 1] 区间

        Args:
            raw_states: [Time, Grid, Nmul, NumModels] 原始状态值
            params_dict: 各模型参数字典

        Returns:
            norm_states: [Time, Grid, Nmul, NumModels] 归一化后的状态
        """
        norm_states = torch.zeros_like(raw_states)

        # HBV: S3 / fc (field capacity)
        if "HBV" in self.model_order:
            idx = self.model_order.index("HBV")
            cap = params_dict["HBV"]["fc"]  # [Grid, Nmul]
            # 广播到时间维度
            norm_states[:, :, :, idx] = raw_states[:, :, :, idx] / (
                cap.unsqueeze(0) + self.nearzero
            )

        # SHM: su / sumax
        if "SHM" in self.model_order:
            idx = self.model_order.index("SHM")
            cap = params_dict["SHM"]["sumax"]
            norm_states[:, :, :, idx] = raw_states[:, :, :, idx] / (
                cap.unsqueeze(0) + self.nearzero
            )

        # EXPHYDRO: soil_storage / smax
        if "EXPHYDRO" in self.model_order:
            idx = self.model_order.index("EXPHYDRO")
            cap = params_dict["EXPHYDRO"]["smax"]
            norm_states[:, :, :, idx] = raw_states[:, :, :, idx] / (
                cap.unsqueeze(0) + self.nearzero
            )

        # HYMOD: S1 / smax
        if "HYMOD" in self.model_order:
            idx = self.model_order.index("HYMOD")
            cap = params_dict["HYMOD"]["smax"]
            norm_states[:, :, :, idx] = raw_states[:, :, :, idx] / (
                cap.unsqueeze(0) + self.nearzero
            )

        # 裁剪到合理范围
        norm_states = torch.clamp(norm_states, 0.0, 2.0)

        return norm_states

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
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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
            outputs: 各模型的流量输出字典
            soil_states_seq: 各模型的核心状态序列 [n_steps, n_grid, nmul, num_models]
        """
        # 初始化状态
        states = self._initialize_states(n_steps, n_grid)

        # 预分配输出张量
        outputs = {}
        for model_name in self.model_order:
            outputs[model_name] = torch.zeros(
                n_steps, n_grid, self.nmul, device=self.device
            )

        # 预分配状态序列（记录各模型核心土壤湿度状态）
        soil_states_seq = torch.zeros(
            n_steps,
            n_grid,
            self.nmul,
            len(self.model_order),
            device=self.device,
        )

        # ===== 核心：统一时间循环 =====
        for t in range(n_steps):
            P_t = P[t]
            T_t = T[t]
            PET_t = PET[t]

            # HBV模型
            if "HBV" in self.model_order:
                idx = self.model_order.index("HBV")
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
                # 记录核心状态：S3 (Soil Moisture)
                soil_states_seq[t, :, :, idx] = S3

            # SHM模型
            if "SHM" in self.model_order:
                idx = self.model_order.index("SHM")
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
                # 记录核心状态：su (Soil Moisture Storage)
                soil_states_seq[t, :, :, idx] = su

            # EXPHYDRO模型
            if "EXPHYDRO" in self.model_order:
                idx = self.model_order.index("EXPHYDRO")
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
                # 记录核心状态：soil_storage
                soil_states_seq[t, :, :, idx] = soil_storage

            # HYMOD模型
            if "HYMOD" in self.model_order:
                idx = self.model_order.index("HYMOD")
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
                # 记录核心状态：S1 (Catchment Moisture Storage)
                soil_states_seq[t, :, :, idx] = S1

        return outputs, soil_states_seq

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（全局权重版：所有Nmul和Model一起归一化）

        Args:
            x_dict: 包含输入数据的字典
            parameters: MultiHeadNet输出的字典
                       {"HBV": tensor, "SHM": tensor, "GAMMA_UH": tensor, ...}
        """
        # 0. 基础数据提取
        x_phy = x_dict["x_phy"]  # [Time, Grid, Var_phy]
        x_nn_norm = x_dict["x_nn_norm"]  # [Time, Grid, Var_nn]
        c_nn_norm = x_dict["c_nn_norm"]  # [Grid, Attr]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        # 1. 解包参数（从字典提取）
        phy_static_dict, phy_route = self.unpack_parameters(parameters)
        self.routing_param_dict = self._descale_routing_params(phy_route)

        n_steps, n_grid = x_phy.shape[:2]

        # 2. 准备物理模型驱动数据 (广播到 Nmul 用于物理计算)
        P = (
            x_phy[:, :, self.variables.index("prcp")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        T = (
            x_phy[:, :, self.variables.index("tmean")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        PET = (
            x_phy[:, :, self.variables.index("pet")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )

        # 3. 获取物理参数
        params_dict = {}
        for model_name in self.model_order:
            params_dict[model_name] = self.get_model_params(
                model_name, phy_static_dict
            )

        # 4. 物理层计算
        # outputs: {Model: [Time, Grid, Nmul]}
        # raw_states: [Time, Grid, Nmul, NumModels]
        model_outputs, raw_states_seq = self._unified_timestep_loop(
            P, T, PET, params_dict, n_steps, n_grid
        )

        # ==========================================================
        # 5. 准备门控网络输入 (Batch = Grid)
        # ==========================================================

        # 5.1 处理状态数据：展平 Nmul 和 Model 维度，作为特征
        # [Time, Grid, Nmul, NumModels] -> [Time, Grid, Nmul * NumModels]
        norm_states = self._normalize_states(raw_states_seq, params_dict)
        flat_states_feat = norm_states.view(n_steps, n_grid, -1)

        # 5.2 拼接动态输入
        # x_nn_norm: [Time, Grid, 3]
        # x_dynamic: [Time, Grid, 3 + (Nmul*NumModels)]
        # 这里 LSTM 同时看到了所有 Parameter Set 的状态，可以学习它们之间的竞争关系
        x_dynamic = torch.cat([x_nn_norm, flat_states_feat], dim=-1)

        # 5.3 处理静态属性
        # [Grid, Attr] -> [Time, Grid, Attr]
        x_static_input = c_nn_norm.unsqueeze(0).expand(n_steps, -1, -1)

        # ==========================================================
        # 6. 门控层计算 & 模型权重生成
        # ==========================================================

        # 输入: [Time, Grid, Feature_Dim]
        # 输出: [Time, Grid, NumModels]  (只有4个权重)
        # Softmax 已经在 PhysicsGatingNetwork 内部对最后一维执行，保证和为1
        weights = self.gating_net(x_dynamic, x_static_input)

        # ==========================================================
        # 7. 对每个模型分别应用路由（使用各自的路由参数）
        # ==========================================================

        # 堆叠物理模型输出 (路由前的产流)
        # stack: [Time, Grid, Nmul, NumModels]
        all_q_stack = torch.stack(
            [model_outputs[m] for m in self.model_order], dim=-1
        )

        # 第一阶段：对每个模型的参数组求平均
        # mean over Nmul: [Time, Grid, Nmul, NumModels] -> [Time, Grid, NumModels]
        model_avg_q = all_q_stack.mean(dim=2)

        # 对每个模型分别应用路由
        routed_outputs = []  # 存储各模型路由后的输出
        for idx, model_name in enumerate(self.model_order):
            # 提取该模型的产流 [Time, Grid]
            q_prerouting = model_avg_q[:, :, idx]
            # 使用该模型的路由参数进行路由
            q_routed = self._apply_routing(
                q_prerouting, model_name, n_steps, n_grid
            )
            routed_outputs.append(q_routed)

        # 堆叠所有模型路由后的输出 [Time, Grid, NumModels]
        all_routed_stack = torch.stack(routed_outputs, dim=-1).squeeze(dim=2)

        # ==========================================================
        # 8. 使用门控网络权重对路由后的输出进行加权融合
        # ==========================================================

        # weights: [Time, Grid, NumModels]
        # all_routed_stack: [Time, Grid, NumModels]
        # 结果: [Time, Grid]
        Qrouted = (weights * all_routed_stack).sum(dim=-1)

        # 9. 构造返回字典
        # weights: [Time, Grid, NumModels] (模型级权重，无需reshape)
        result: Dict[str, torch.Tensor] = {
            "streamflow": Qrouted,  # 加权融合后的最终输出
            # "weights": weights,  # [Time, Grid, NumModels] 门控网络生成的权重
            "model_avg_outputs": model_avg_q,  # 各模型的平均产流（路由前）
        }

        for i, m in enumerate(self.model_order):
            result[f"{m}_weights"] = weights[:, :, i]

        # 添加各模型的单独输出（路由前的产流和路由后的流量）
        for idx, name in enumerate(self.model_order):
            # 路由前的产流 (nmul平均值)
            result[f"{name.lower()}_prerouting"] = model_outputs[name].mean(-1)
            # 路由后的流量
            result[f"{name.lower()}_streamflow"] = all_routed_stack[:, :, idx]

        # 10. 截断warmup
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
