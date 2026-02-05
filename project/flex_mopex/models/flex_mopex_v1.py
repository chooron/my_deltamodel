from typing import Any, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma
from project.flex_mopex.models import mopex_core


class FlexMopexV1(nn.Module):
    """MOPEX 水文模型：结构权重 + 物理参数 + torch.compile"""

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

        self.name = "FlexMopexV1"
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
        self.weight_names = ["w_phen", "w_int", "w_snow", "w_sub"]

        # 总参数数量：模型参数 * nmul + 路由参数 + 权重参数 * 2(On/Off)
        # 每个权重需要2个logits（Off和On状态），用于独立的二分类决策
        # 注意：权重不再乘以nmul，直接预测单个权重值
        self.learnable_param_count = (
            len(self.mopex_param_names) * self.nmul
            + len(self.routing_param_names)
            + len(self.weight_names) * 2  # 注意：不再乘以nmul
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
        """使用 torch.compile 编译 mopex_step 函数"""
        self.mopex_step_compiled = torch.compile(mopex_core.mopex_step)

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

    def _descale_weights(
        self, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        反归一化结构权重
        注意：weights已经通过gumbel_softmax激活，值域已经在[0,1]之间
        这里直接映射到目标范围（实际上WEIGHT_BOUNDS都是[0,1]，所以不需要变换）

        weights shape: [B, 4] - 每个过程一个权重值
        """
        result = {}
        for i, name in enumerate(self.weight_names):
            result[name] = weights[:, i]  # [B]
        return result

    def unpack_parameters(
        self, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解包参数（从MultiHeadNet字典输出提取）

        Args:
            parameters: 字典，包含 MOPEX 参数、权重和路由参数
                例如: {"params": [B, 12*nmul], "weights": [B, 8],
                      "gamma_uh": [B, 2]}
                注意：weights 维度是 8，因为每个过程有2个logits (Off/On)，不考虑nmul

        Returns:
            - mopex_params: MOPEX 模型参数 [B, 12, nmul]
            - weights: 结构权重 [B, 4] (每个过程独立经过gumbel_softmax)
            - routing_params: 路由参数 [B, 2]
        """
        # 提取并激活 MOPEX 参数（使用sigmoid）
        raw_mopex = self.activate(parameters["params"])
        mopex_params = raw_mopex.view(
            raw_mopex.shape[0], len(self.mopex_param_names), self.nmul
        )

        # 权重处理 - 不考虑nmul维度
        raw_weights = parameters["weights"]  # [B, 8]
        weights_reshaped = raw_weights.view(
            raw_weights.shape[0], len(self.weight_names), 2
        )  # [B, 4, 2]
        weights_clipped = torch.clamp(weights_reshaped, min=-10.0, max=10.0)
        if self.training:
            weights_probs = F.gumbel_softmax(
                weights_clipped, tau=1.0, hard=False, dim=-1
            )  # [B, 4, 2]
        else:
            weights_probs = F.softmax(weights_clipped, dim=-1)
        weights_on = weights_probs[..., 1]  # [B, 4] - 取On状态的概率

        # 提取并激活路由参数
        routing_params = self.activate(parameters["gamma_uh"])

        return mopex_params, weights_on, routing_params

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
        mopex_params: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        n_steps: int,
        n_grid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MOPEX 模型的时间循环

        Args:
            P: [n_steps, n_grid, nmul]
            T: [n_steps, n_grid, nmul]
            PET: [n_steps, n_grid, nmul]
            doy: [n_steps, n_grid, nmul]
            mopex_params: MOPEX 参数字典
            weights: 结构权重字典 - 每个权重形状为 [n_grid]，需要扩展到 [n_grid, nmul]
            n_steps: 时间步数
            n_grid: 网格数

        Returns:
            Q: 径流 [n_steps, n_grid, nmul]
            ET: 蒸散发 [n_steps, n_grid, nmul]
        """
        # 初始化状态
        states = self._initialize_states(n_steps, n_grid)

        # 扩展 weights 到 [n_grid, nmul] 维度
        weights_expanded = {}
        for name in self.weight_names:
            # weights[name] 形状: [n_grid]
            # 扩展到 [n_grid, nmul]
            weights_expanded[name] = weights[name].unsqueeze(-1).repeat(1, self.nmul)

        # 预分配输出张量
        Q_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)
        ET_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)

        # 时间循环
        for t in range(n_steps):
            P_t = P[t]
            T_t = T[t]
            PET_t = PET[t]
            doy_t = doy[t]

            # 调用编译后的 mopex_step
            Q, ET, S1, S2, Sc1, Sc2, Sn = self.mopex_step_compiled(
                P_t,
                T_t,
                PET_t,
                doy_t,
                # 结构权重 - 使用扩展后的权重
                weights_expanded["w_phen"],
                weights_expanded["w_int"],
                weights_expanded["w_snow"],
                weights_expanded["w_sub"],
                # 模型参数
                mopex_params["Sb1"],
                mopex_params["tw"],
                mopex_params["tu"],
                mopex_params["Se"],
                mopex_params["tc"],
                mopex_params["ddf"],
                mopex_params["tcrit"],
                mopex_params["Sb2"],
                mopex_params["alpha"],
                mopex_params["is_time"],
                mopex_params["tmin"],
                mopex_params["tmax"],
                # 状态变量
                states["S1"],
                states["S2"],
                states["Sc1"],
                states["Sc2"],
                states["Sn"],
                self.nearzero,
            )

            Q_out[t] = Q
            ET_out[t] = ET

            # 更新状态
            states["S1"] = S1
            states["S2"] = S2
            states["Sc1"] = Sc1
            states["Sc2"] = Sc2
            states["Sn"] = Sn

        return Q_out, ET_out

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
                       {"MOPEX": tensor, "WEIGHTS": tensor, "GAMMA_UH": tensor}

        Returns:
            包含'streamflow'等输出的字典
        """
        x = x_dict["x_phy"]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        # 解包参数
        mopex_params_raw, weights_raw, routing_raw = self.unpack_parameters(
            parameters
        )

        # 反归一化参数
        mopex_params = self._descale_params(
            mopex_params_raw, self.mopex_param_names, self.MOPEX_PARAMS_BOUNDS
        )
        weights = self._descale_weights(weights_raw)
        self.routing_param_dict = self._descale_routing_params(routing_raw)

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
        doy = x_dict["doy"].repeat(1, 1, self.nmul)

        # MOPEX 时间循环
        Q_mopex, ET_mopex = self._mopex_timestep_loop(
            P, T, PET, doy, mopex_params, weights, n_steps, n_grid
        )

        # 平均 nmul 维度
        Q_mean = Q_mopex.mean(-1)
        ET_mean = ET_mopex.mean(-1)

        # 应用路由
        Qrouted = self._apply_routing(Q_mean, n_steps, n_grid)

        # 构造返回字典
        result: Dict[str, torch.Tensor] = {
            "streamflow": Qrouted,
            "mopex_prerouting": Q_mean,
            "et": ET_mean,
            "target": x_dict["target"],
        }

        # 截断warmup
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        # save weights - 扩展到 [n_steps, n_grid, nmul] 以便保存
        for weight_name in self.weight_names:
            # weights[weight_name] 形状: [n_grid]
            # 扩展到 [n_steps, n_grid, nmul]
            result[weight_name] = (
                weights[weight_name]
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(n_steps, 1, self.nmul)
            )

        return result
