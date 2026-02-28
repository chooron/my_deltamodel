from typing import Any, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma
from project.flex_mopex.models import mopex_core


class WeightMLPGate(nn.Module):
    """MLP门控网络用于基于内部状态、气象输入和静态属性预测动态权重"""

    def __init__(
        self,
        state_size: int = 5,  # S1, S2, Sc1, Sc2, Sn
        forcing_size: int = 3,  # P, T, PET
        static_size: int = 0,  # 静态属性维度
        hidden_size: int = 32,
        num_weights: int = 4,  # 4个过程权重
    ):
        super().__init__()

        # 总输入维度 = 状态 + 气象强迫 + 静态属性
        input_size = state_size + forcing_size + static_size

        # MLP层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 输出层：预测每个权重的2个logits (Off/On)
        self.fc_out = nn.Linear(hidden_size, num_weights * 2)

    def forward(
        self,
        states: torch.Tensor,
        forcings: torch.Tensor,
        static_attrs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            states: [n_grid, 5] 内部状态变量 (S1, S2, Sc1, Sc2, Sn)
            forcings: [n_grid, 3] 气象强迫 (P, T, PET)
            static_attrs: [n_grid, static_size] 静态属性 (可选)

        Returns:
            weights_logits: [n_grid, 4, 2] 权重logits
        """
        # 拼接输入
        inputs = [states, forcings]
        if static_attrs is not None:
            inputs.append(static_attrs)
        x = torch.cat(inputs, dim=-1)  # [n_grid, state_size + forcing_size + static_size]

        # MLP前向传播
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        weights_logits = self.fc_out(h)  # [n_grid, 8]

        # 重塑为 [n_grid, 4, 2]
        weights_logits = weights_logits.view(-1, 4, 2)

        return weights_logits


class FlexMopexV3(nn.Module):
    """MOPEX 水文模型：MLP Gate动态权重 + 物理参数 + torch.compile

    版本二：MLP Gate 基于内部状态预测动态权重
    - 权重不再是静态的，而是由MLP在每个时间步根据前一时间步的内部状态动态生成
    - MLP读取上一步的模型内部物理状态（土壤含水量、积雪储量等），输出四个过程的激活概率
    - 使用前一时间步的状态变量而非当前时间步，以避免同步循环依赖
    - 训练时保持现有预热机制：730天输入，365天预热，仅后365天计算损失
    - 预热期内MLP Gate同样参与前向传播使内部状态充分演化，但不参与梯度更新
    - AIC稀疏正则化仍作用于每个时间步的weights_on
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

        self.name = "FlexMopexV1MLPGate"
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

        # MLP Gate配置
        self.mlp_hidden_size = config.get("mlp_hidden_size", 32) if config else 32
        self.static_size = config.get("static_size", 0) if config else 0  # 静态属性维度

        # 总参数数量：模型参数 * nmul + 路由参数
        # 注意：权重现在由MLP Gate预测，不再从MultiHeadNet输出
        self.learnable_param_count = (
            len(self.mopex_param_names) * self.nmul
            + len(self.routing_param_names)
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 初始化MLP Gate权重预测器
        self.weight_mlp_gate = WeightMLPGate(
            state_size=5,  # S1, S2, Sc1, Sc2, Sn
            forcing_size=3,  # P, T, PET
            static_size=self.static_size,  # 静态属性
            hidden_size=self.mlp_hidden_size,
            num_weights=len(self.weight_names),
        ).to(self.device)

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

    def unpack_parameters(
        self, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解包参数（从MultiHeadNet字典输出提取）

        Args:
            parameters: 字典，包含 MOPEX 参数和路由参数
                例如: {"params": [B, 12*nmul], "gamma_uh": [B, 2]}
                注意：不再包含weights，因为权重由MLP Gate动态生成

        Returns:
            - mopex_params: MOPEX 模型参数 [B, 12, nmul]
            - routing_params: 路由参数 [B, 2]
        """
        # 提取并激活 MOPEX 参数（使用sigmoid）
        raw_mopex = self.activate(parameters["params"])
        mopex_params = raw_mopex.view(
            raw_mopex.shape[0], len(self.mopex_param_names), self.nmul
        )

        # 提取并激活路由参数
        routing_params = self.activate(parameters["gamma_uh"])

        return mopex_params, routing_params

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

    def _predict_weights_from_states(
        self,
        states: Dict[str, torch.Tensor],
        forcings: torch.Tensor,
        static_attrs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        使用MLP Gate基于内部状态、气象强迫和静态属性预测权重

        Args:
            states: 字典，包含内部状态变量
                   每个状态形状: [n_grid, nmul]
            forcings: [n_grid, 3] 气象强迫 (P, T, PET)
            static_attrs: [n_grid, static_size] 静态属性 (可选)

        Returns:
            weights_on: [n_grid, 4] 权重激活概率
        """
        # 准备MLP输入：[n_grid, 5]
        # 取nmul维度的第一个值（如果nmul=1，则直接squeeze）
        state_vector = torch.stack(
            [
                states["S1"][:, 0],
                states["S2"][:, 0],
                states["Sc1"][:, 0],
                states["Sc2"][:, 0],
                states["Sn"][:, 0],
            ],
            dim=-1,
        )  # [n_grid, 5]

        # MLP前向传播
        weights_logits = self.weight_mlp_gate(
            state_vector, forcings, static_attrs
        )  # [n_grid, 4, 2]

        # 应用gumbel_softmax或softmax
        weights_logits_clipped = torch.clamp(weights_logits, min=-10.0, max=10.0)
        if self.training:
            weights_probs = F.gumbel_softmax(
                weights_logits_clipped, tau=1.0, hard=False, dim=-1
            )  # [n_grid, 4, 2]
        else:
            weights_probs = F.softmax(weights_logits_clipped, dim=-1)

        weights_on = weights_probs[..., 1]  # [n_grid, 4] - 取On状态的概率

        return weights_on

    def _mopex_timestep_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        doy: torch.Tensor,
        mopex_params: Dict[str, torch.Tensor],
        static_attrs: torch.Tensor,
        n_steps: int,
        n_grid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MOPEX 模型的时间循环（使用MLP Gate动态权重）

        Args:
            P: [n_steps, n_grid, nmul]
            T: [n_steps, n_grid, nmul]
            PET: [n_steps, n_grid, nmul]
            doy: [n_steps, n_grid, nmul]
            mopex_params: MOPEX 参数字典
            static_attrs: [n_grid, static_size] 静态属性
            n_steps: 时间步数
            n_grid: 网格数

        Returns:
            Q: 径流 [n_steps, n_grid, nmul]
            ET: 蒸散发 [n_steps, n_grid, nmul]
            weights_dynamic: [n_steps, n_grid, 4] 动态权重
        """
        # 初始化状态
        states = self._initialize_states(n_steps, n_grid)

        # 预分配输出张量
        Q_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)
        ET_out = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)
        weights_out = torch.zeros(n_steps, n_grid, 4, device=self.device)

        # 时间循环
        for t in range(n_steps):
            P_t = P[t]
            T_t = T[t]
            PET_t = PET[t]
            doy_t = doy[t]

            # 准备当前时间步的气象强迫 [n_grid, 3]
            forcings_t = torch.stack(
                [P_t[:, 0], T_t[:, 0], PET_t[:, 0]], dim=-1
            )  # [n_grid, 3]

            # 使用前一时间步的状态、当前时间步的气象强迫和静态属性预测当前时间步的权重
            weights_t = self._predict_weights_from_states(
                states, forcings_t, static_attrs
            )  # [n_grid, 4]

            # 扩展权重到 [n_grid, nmul]
            w_phen_t = weights_t[:, 0].unsqueeze(-1).repeat(1, self.nmul)
            w_int_t = weights_t[:, 1].unsqueeze(-1).repeat(1, self.nmul)
            w_snow_t = weights_t[:, 2].unsqueeze(-1).repeat(1, self.nmul)
            w_sub_t = weights_t[:, 3].unsqueeze(-1).repeat(1, self.nmul)

            # 调用编译后的 mopex_step
            Q, ET, S1, S2, Sc1, Sc2, Sn = self.mopex_step_compiled(
                P_t,
                T_t,
                PET_t,
                doy_t,
                # 结构权重 - 使用基于前一时间步状态预测的权重
                w_phen_t,
                w_int_t,
                w_snow_t,
                w_sub_t,
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
            weights_out[t] = weights_t

            # 更新状态（用于下一时间步的权重预测）
            states["S1"] = S1
            states["S2"] = S2
            states["Sc1"] = Sc1
            states["Sc2"] = Sc2
            states["Sn"] = Sn

        return Q_out, ET_out, weights_out

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
                - 'c_nn_norm': [n_grid, static_size] 静态属性
                - 'doy': [n_steps, n_grid] 日序
                - 'target': [n_steps, n_grid, 1] 目标值
            parameters: MultiHeadNet输出的字典
                       {"params": tensor, "gamma_uh": tensor}
                       注意：不再包含weights

        Returns:
            包含'streamflow'等输出的字典
        """
        x = x_dict["x_phy"]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        # 解包参数（不包含weights）
        mopex_params_raw, routing_raw = self.unpack_parameters(parameters)

        # 反归一化参数
        mopex_params = self._descale_params(
            mopex_params_raw, self.mopex_param_names, self.MOPEX_PARAMS_BOUNDS
        )
        self.routing_param_dict = self._descale_routing_params(routing_raw)

        n_steps, n_grid = x.shape[:2]

        # 提取静态属性（如果存在）
        static_attrs = None
        if "c_nn_norm" in x_dict and self.static_size > 0:
            static_attrs = x_dict["c_nn_norm"]  # [n_grid, static_size]

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

        # MOPEX 时间循环（使用MLP Gate动态权重）
        Q_mopex, ET_mopex, weights_dynamic = self._mopex_timestep_loop(
            P, T, PET, doy, mopex_params, static_attrs, n_steps, n_grid
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

        # save weights - 权重现在是时变的 [n_steps, n_grid, 4]
        # 转换为 [n_steps, n_grid, 1] 格式以便保存
        for i, weight_name in enumerate(self.weight_names):
            result[weight_name] = weights_dynamic[:, :, i].unsqueeze(-1)

        # 截断warmup
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
