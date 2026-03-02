from typing import Any, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma
from project.flex_mopex.models import mopex_core


class DiffMopexV1(nn.Module):
    """MOPEX 水文模型：动态参数版本 (ddf, alpha, Sb2, Se 动态预测)"""

    MOPEX_PARAMS_BOUNDS = {
        "Sb1": [0.01, 50.0],
        "tw": [0.01, 5.0],
        "tu": [1.0, 2000.0],
        "Se": [1.0, 1000.0],
        "tc": [0.1, 30.0],
        "ddf": [0.0, 20.0],
        "tcrit": [-3.0, 3.0],
        "Sb2": [1.0, 1500.0],
        "alpha": [0.0, 1.0],
        "is_time": [0.0, 365.0],
        "tmin": [-10.0, 5.0],
        "tmax": [5.0, 30.0],
    }

    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.name = "FlexMopexV4"
        self.config = config or {}
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.activate = F.sigmoid

        # MOPEX 模型参数
        self.mopex_param_names = list(self.MOPEX_PARAMS_BOUNDS.keys())
        self.routing_param_names = list(self.ROUTING_BOUNDS.keys())

        # 动态参数：由网络预测
        self.dynamic_param_names = ["ddf", "alpha", "Sb2", "Se"]

        # 静态参数：保持固定（从静态属性预测，但不随时间变化）
        self.static_param_names = [p for p in self.mopex_param_names if p not in self.dynamic_param_names]

        # 总参数数量：静态参数 * nmul + 路由参数
        # 动态参数由 MultiHeadNetParam 预测，不计入此处
        self.learnable_param_count = (
            len(self.static_param_names) * self.nmul
            + len(self.routing_param_names)
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if config is not None:
            self._load_config(config)

        self._setup_compiled_kernels()

    def _load_config(self, config: Dict[str, Any]) -> None:
        """加载配置"""
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

    def _setup_compiled_kernels(self) -> None:
        """使用 torch.compile 编译 mopex_step_static 函数"""
        self.mopex_step_compiled = torch.compile(mopex_core.mopex_step_static)

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
            name: change_param_range(params[:, i], self.ROUTING_BOUNDS[name])
            for i, name in enumerate(self.routing_param_names)
        }

    def unpack_parameters(
        self, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解包参数（从MultiHeadNetParam字典输出提取）

        Args:
            parameters: 字典，包含静态参数、动态参数和路由参数
                例如: {"static_params": [B, 8*nmul],
                      "dynamic_params": [n_steps, B, 4*nmul],
                      "gamma_uh": [B, 2]}

        Returns:
            - static_params: 静态 MOPEX 参数 [B, 8, nmul]
            - dynamic_params: 动态 MOPEX 参数 [n_steps, B, 4, nmul]
            - routing_params: 路由参数 [B, 2]
        """
        # 提取并激活静态参数
        raw_static = self.activate(parameters["static_params"])
        static_params = raw_static.view(
            raw_static.shape[0], len(self.static_param_names), self.nmul
        )

        # 提取并激活动态参数
        raw_dynamic = self.activate(parameters["dynamic_params"])  # [n_steps, B, 4*nmul]
        n_steps, batch_size, _ = raw_dynamic.shape
        dynamic_params = raw_dynamic.view(
            n_steps, batch_size, len(self.dynamic_param_names), self.nmul
        )

        # 提取并激活路由参数
        routing_params = self.activate(parameters["gamma_uh"])

        return static_params, dynamic_params, routing_params

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
    ) -> Dict[str, torch.Tensor]:
        """初始化 MOPEX 模型的状态变量"""
        return {
            "S1": torch.zeros(n_grid, self.nmul, device=self.device)
            + self.nearzero,
            "S2": torch.zeros(n_grid, self.nmul, device=self.device)
            + self.nearzero,
            "Sc1": torch.zeros(n_grid, self.nmul, device=self.device)
            + self.nearzero,
            "Sc2": torch.zeros(n_grid, self.nmul, device=self.device)
            + self.nearzero,
            "Sn": torch.zeros(n_grid, self.nmul, device=self.device)
            + self.nearzero,
        }

    def _mopex_timestep_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        doy: torch.Tensor,
        static_params: Dict[str, torch.Tensor],
        dynamic_params: Dict[str, torch.Tensor],
        n_steps: int,
        n_grid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MOPEX 模型的时间循环（动态参数版本）

        Args:
            P: [n_steps, n_grid, nmul]
            T: [n_steps, n_grid, nmul]
            PET: [n_steps, n_grid, nmul]
            doy: [n_steps, n_grid, nmul]
            static_params: 静态参数字典，每个参数形状为 [n_grid, nmul]
            dynamic_params: 动态参数字典，每个参数形状为 [n_steps, n_grid, nmul]
            n_steps: 时间步数
            n_grid: 网格数

        Returns:
            Q: 径流 [n_steps, n_grid, nmul]
            ET: 蒸散发 [n_steps, n_grid, nmul]
        """
        # 初始化状态
        states = self._initialize_states(n_steps, n_grid)

        # 预分配输出张量
        Q_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)
        ET_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)

        # 时间循环
        for t in range(n_steps):
            # 合并静态参数和当前时间步的动态参数
            current_params = {}

            # 添加静态参数
            for name in self.static_param_names:
                current_params[name] = static_params[name]

            # 添加动态参数（当前时间步）
            for name in self.dynamic_param_names:
                current_params[name] = dynamic_params[name][t]  # [n_grid, nmul]

            # 调用 mopex_step_static
            Q, ET, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new = (
                self.mopex_step_compiled(
                    P=P[t],
                    T=T[t],
                    PET=PET[t],
                    doy=doy[t],
                    Sb1=current_params["Sb1"],
                    tw=current_params["tw"],
                    tu=current_params["tu"],
                    Se=current_params["Se"],
                    tc=current_params["tc"],
                    ddf=current_params["ddf"],
                    tcrit=current_params["tcrit"],
                    Sb2=current_params["Sb2"],
                    alpha=current_params["alpha"],
                    is_time=current_params["is_time"],
                    tmin=current_params["tmin"],
                    tmax=current_params["tmax"],
                    S1=states["S1"],
                    S2=states["S2"],
                    Sc1=states["Sc1"],
                    Sc2=states["Sc2"],
                    Sn=states["Sn"],
                    nearzero=self.nearzero,
                )
            )

            # 保存输出
            Q_out[t] = Q
            ET_out[t] = ET

            # 更新状态
            states["S1"] = S1_new
            states["S2"] = S2_new
            states["Sc1"] = Sc1_new
            states["Sc2"] = Sc2_new
            states["Sn"] = Sn_new

        return Q_out, ET_out

    def forward(
        self, x: torch.Tensor, x_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 [n_steps, n_grid, n_features]
            x_dict: 包含额外信息的字典
                - "target": 目标值
                - "doy": 日序
                - "parameters": 参数字典（从MultiHeadNetParam输出）

        Returns:
            result: 包含预测结果的字典
        """
        n_steps, n_grid, _ = x.shape

        # 解包参数
        static_params, dynamic_params, routing_params = self.unpack_parameters(
            x_dict["parameters"]
        )

        # 反归一化静态参数
        self.static_param_dict = self._descale_params(
            static_params, self.static_param_names, self.MOPEX_PARAMS_BOUNDS
        )

        # 反归一化动态参数
        # dynamic_params: [n_steps, B, 4, nmul]
        dynamic_params_descaled = {}
        for i, name in enumerate(self.dynamic_param_names):
            # 提取第i个参数: [n_steps, B, nmul]
            param_values = dynamic_params[:, :, i, :]
            # 转置到 [n_steps, B, nmul] -> [B, n_steps, nmul] -> 反归一化 -> [n_steps, B, nmul]
            param_values_transposed = param_values.permute(1, 0, 2)  # [B, n_steps, nmul]
            param_descaled = change_param_range(
                param_values_transposed, self.MOPEX_PARAMS_BOUNDS[name]
            )
            dynamic_params_descaled[name] = param_descaled.permute(1, 0, 2)  # [n_steps, B, nmul]

        # 反归一化路由参数
        self.routing_param_dict = self._descale_routing_params(routing_params)

        # 准备输入
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
        doy = x_dict["doy"].repeat(1, 1, self.nmul)

        # MOPEX 时间循环
        Q_mopex, ET_mopex = self._mopex_timestep_loop(
            P, T, PET, doy, self.static_param_dict, dynamic_params_descaled, n_steps, n_grid
        )

        # 平均 nmul 维度
        Q_mean = Q_mopex.mean(-1)
        ET_mean = ET_mopex.mean(-1)

        # 应用路由
        Qrouted = self._apply_routing(Q_mean, n_steps, n_grid)

        # 构造返回字典
        result: Dict[str, torch.Tensor] = {
            "streamflow": Qrouted,
            "target": x_dict["target"],
        }

        # 保存动态参数用于分析
        for param_name in self.dynamic_param_names:
            # dynamic_params_descaled[param_name] 形状: [n_steps, n_grid, nmul]
            # 转换为 [n_steps, nmul, n_grid] 以匹配输出格式
            result[param_name] = dynamic_params_descaled[param_name].permute(0, 2, 1)

        # 截断warmup
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
