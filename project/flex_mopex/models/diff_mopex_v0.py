from typing import Any, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma
from project.flex_mopex.models import mopex_core


class DiffMopexV0(nn.Module):
    """MOPEX 水文模型：全静态版本（无神经网络权重预测）

    版本零：完全静态的MOPEX模型
    - 直接调用 mopex_step_static，所有权重都是1（全开）
    - 所有12个MOPEX参数都是静态的
    - 兼容 MultiHeadNetStatic 的输出格式
    """

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

        self.name = "DiffMopexV0"
        self.config = config or {}
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.activate = F.sigmoid

        # MOPEX 模型参数（全部为静态参数）
        self.mopex_param_names = list(self.MOPEX_PARAMS_BOUNDS.keys())
        self.routing_param_names = list(self.ROUTING_BOUNDS.keys())
        self.learnable_param_count = (
            len(self.mopex_param_names) * self.nmul
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
        """使用 torch.compile 编译关键函数以提升性能"""
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
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        解包参数（从MultiHeadNetStatic字典输出提取）

        Args:
            parameters: 字典，包含静态参数和路由参数
                例如: {
                    "static_params": [B, 12*nmul],  # 全部12个参数
                    "gamma_uh": [B, 2]
                }

        Returns:
            - static_params_dict: 静态MOPEX参数字典 {param_name: [B, nmul]}
            - routing_params: 路由参数 [B, 2]
        """
        # 提取并激活静态参数 (12个)
        raw_static = self.activate(parameters["static_params"])
        static_params_reshaped = raw_static.view(
            raw_static.shape[0], len(self.mopex_param_names), self.nmul
        )  # [B, 12, nmul]

        # 构建静态参数字典
        static_params_dict = {}
        for i, name in enumerate(self.mopex_param_names):
            static_params_dict[name] = static_params_reshaped[:, i, :]  # [B, nmul]

        # 提取并激活路由参数
        routing_params = self.activate(parameters["gamma_uh"])

        return static_params_dict, routing_params

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
            "S1": torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            "S2": torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            "Sc1": torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            "Sc2": torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
            "Sn": torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero,
        }

    def _mopex_timestep_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        doy: torch.Tensor,
        mopex_params_static: Dict[str, torch.Tensor],
        n_steps: int,
        n_grid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MOPEX 模型的时间循环（全静态版本，无权重预测）
        直接调用 mopex_step_static，所有权重都是1（全开）
        """
        # 初始化状态
        states = self._initialize_states(n_steps, n_grid)

        # 预分配输出张量
        Q_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)
        ET_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)

        # 预提取静态参数
        Sb1 = mopex_params_static["Sb1"]
        tw = mopex_params_static["tw"]
        tu = mopex_params_static["tu"]
        Se = mopex_params_static["Se"]
        tc = mopex_params_static["tc"]
        ddf = mopex_params_static["ddf"]
        tcrit = mopex_params_static["tcrit"]
        Sb2 = mopex_params_static["Sb2"]
        alpha = mopex_params_static["alpha"]
        is_time = mopex_params_static["is_time"]
        tmin = mopex_params_static["tmin"]
        tmax = mopex_params_static["tmax"]

        # 预提取状态变量引用
        S1, S2, Sc1, Sc2, Sn = states["S1"], states["S2"], states["Sc1"], states["Sc2"], states["Sn"]

        # 时间循环
        for t in range(n_steps):
            P_t = P[t]
            T_t = T[t]
            PET_t = PET[t]
            doy_t = doy[t]

            # 调用编译后的 mopex_step_static（所有权重都是1，全开）
            Q, ET, S1, S2, Sc1, Sc2, Sn = self.mopex_step_compiled(
                P_t, T_t, PET_t, doy_t,
                Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax,
                S1, S2, Sc1, Sc2, Sn,
                self.nearzero,
            )

            Q_out[t] = Q
            ET_out[t] = ET

        return Q_out, ET_out

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x_dict: 包含以下键的字典
                - 'x_phy': [n_steps, n_grid, n_vars] 动态气象数据
                - 'doy': [n_steps, n_grid] 日序
                - 'target': [n_steps, n_grid, 1] 目标值
            parameters: MultiHeadNetStatic输出的字典
                       {"static_params": [B, 12*nmul], "gamma_uh": [B, 2]}

        Returns:
            result: 包含 'streamflow' 和 'target' 的字典
        """
        # 解包参数
        mopex_params_static_raw, routing_params = self.unpack_parameters(parameters)

        # 反归一化MOPEX参数
        mopex_params_static = self._descale_params(
            torch.stack([mopex_params_static_raw[name] for name in self.mopex_param_names], dim=1),
            self.mopex_param_names,
            self.MOPEX_PARAMS_BOUNDS,
        )

        # 反归一化路由参数
        self.routing_param_dict = self._descale_routing_params(routing_params)

        # 提取输入
        x = x_dict["x_phy"]
        n_steps, n_grid, _ = x.shape

        # 提取气象强迫
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

        # MOPEX 时间循环（全静态版本）
        Q_mopex, ET_mopex = self._mopex_timestep_loop(
            P, T, PET, doy, mopex_params_static, n_steps, n_grid
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

        # 截断warmup
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result

