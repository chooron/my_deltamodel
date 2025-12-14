"""
CFE 水文模型 (DPL 版本)

基于 CFE (Conceptual Functional Equivalent) 概念式水文模型的实现，
支持深度学习参数化。使用 Classic 模式（非 ODE 模式）。

CFE 参考文献:
    Ogden, F. L., et al.: A Conceptual Functional Equivalent (CFE) model 
    for use in the Next Generation Water Resources Modeling Framework.

Author: chooron
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from .jit_core import cfe_timestep_loop


class Cfe(torch.nn.Module):
    """
    CFE 水文模型 (DPL 版本)

    Conceptual Functional Equivalent 模型实现，支持多模型集成和深度学习参数化。
    使用 Classic 模式进行土壤水分计算。

    模型结构:
    - ET 模块: 先从降雨中蒸发，再从土壤中蒸发（Budyko 类型）
    - Schaake 分配: 入渗/径流分配
    - 土壤水库: 双出口非线性水库（渗透 + 侧向流）
    - 地下水库: 指数型出流
    - Nash 级联: 侧向流汇流演算

    Parameters
    ----------
    config : dict, optional
        模型配置字典
    device : torch.device, optional
        运行设备
    """

    # 物理参数边界
    PARAM_BOUNDS = {
        # Schaake partitioning
        "schaake_const": [0.0, 0.1],      # Schaake 常数 [1/day]
        # Soil parameters
        "smcmax": [0.3, 0.6],             # 最大土壤含水量（孔隙度）[-]
        "soil_depth": [0.5, 3.0],         # 土壤深度 [m]
        "wltsmc": [0.05, 0.2],            # 凋萎点含水量 [-]
        "satpsi": [0.01, 0.5],            # 饱和水势 [m]
        "bb": [2.0, 12.0],                # 土壤水力特性参数 b [-]
        "mult": [100.0, 2000.0],          # 渗透乘数 [-]
        "satdk": [1e-7, 1e-4],            # 饱和导水率 [m/s]
        "slop": [0.0, 1.0],               # 坡度 [-]
        # Groundwater parameters
        "max_gw_storage": [0.01, 0.5],    # 最大地下水蓄水量 [m]
        "Cgw": [1e-6, 1e-3],              # 地下水出流系数 [m/s]
        "expon": [1.0, 8.0],              # 地下水出流指数 [-]
        # Nash cascade parameters
        "K_nash": [0.01, 0.5],            # Nash 退水系数 [-]
    }
    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    # 初始状态
    INITIAL_STATES = {
        "soil_storage": 0.05,   # 土壤蓄水 [m]
        "gw_storage": 0.01,     # 地下水蓄水 [m]
    }

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # 默认配置
        self.name = "CFE"
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.variables = ["prcp", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.num_nash_reservoirs = 3  # Nash 级联水库数量
        self.timestep_d = 1.0  # 时间步长（天）

        # 参数边界
        self.parameter_bounds = self.PARAM_BOUNDS
        self.routing_parameter_bounds = self.ROUTING_BOUNDS
        self.initial_states = self.INITIAL_STATES

        # 设备配置
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 从配置更新参数
        if config is not None:
            self._load_config(config)

        self._set_parameters()

    def _load_config(self, config: dict) -> None:
        """从配置字典加载参数"""
        simple_attrs = [
            "warm_up",
            "warm_up_states",
            "variables",
            "routing",
            "nearzero",
            "nmul",
            "num_nash_reservoirs",
            "timestep_d",
        ]
        for attr in simple_attrs:
            if attr in config:
                setattr(self, attr, config[attr])

        if "dynamic_params" in config:
            self.dynamic_params = config["dynamic_params"].get(
                self.__class__.__name__, []
            )

    def _set_parameters(self) -> None:
        """设置参数名称和可学习参数数量"""
        self.phy_param_names = list(self.parameter_bounds.keys())
        self.routing_param_names = list(self.routing_parameter_bounds.keys())

        static_count = len(self.phy_param_names) - len(self.dynamic_params)
        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul
        self.learnable_param_count2 = static_count * self.nmul + len(
            self.routing_param_names
        )
        self.learnable_param_count = (
            self.learnable_param_count1 + self.learnable_param_count2
        )

    def _init_state(self, n_grid: int, state_name: str) -> torch.Tensor:
        """初始化状态张量"""
        init_value = self.initial_states.get(state_name, self.nearzero)
        return (
            torch.zeros(
                [n_grid, self.nmul], dtype=torch.float32, device=self.device
            )
            + init_value
        )

    def _init_nash_storage(self, n_grid: int) -> torch.Tensor:
        """初始化 Nash 级联蓄水"""
        return torch.zeros(
            [n_grid, self.nmul, self.num_nash_reservoirs],
            dtype=torch.float32, device=self.device
        ) + self.nearzero

    def _init_output(self, shape: tuple) -> torch.Tensor:
        """初始化输出张量"""
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _descale_params(
        self, params: torch.Tensor, names: list, bounds: dict
    ) -> dict:
        """通用参数反缩放"""
        return {
            name: change_param_range(params[:, i, :], bounds[name])
            for i, name in enumerate(names)
        }

    def _descale_dynamic_params(
        self, params: torch.Tensor, names: list
    ) -> dict:
        """动态参数反缩放（带 dropout）"""
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
            result[name] = change_param_range(
                combined, self.parameter_bounds[name]
            )
        return result

    def _descale_routing_params(self, params: torch.Tensor) -> dict:
        """汇流参数反缩放"""
        return {
            name: change_param_range(
                params[:, i], self.routing_parameter_bounds[name]
            )
            for i, name in enumerate(self.routing_parameter_bounds.keys())
        }

    def unpack_parameters(
        self, parameters: tuple[Union[None, torch.Tensor], torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor, torch.Tensor]:
        """解包神经网络输出的参数"""
        dy_count = len(self.dynamic_params)
        static_count = len(self.parameter_bounds) - dy_count
        raw_phy_dy, raw_phy_static = parameters

        # 动态参数: (T, B, dy_count, nmul)
        if raw_phy_dy is not None:
            phy_dy = raw_phy_dy.view(
                raw_phy_dy.shape[0], raw_phy_dy.shape[1], dy_count, self.nmul
            )
        else:
            phy_dy = None

        # 静态参数: (B, static_count, nmul)
        phy_static = raw_phy_static[:, : static_count * self.nmul].view(
            raw_phy_static.shape[0], static_count, self.nmul
        )

        # 汇流参数
        phy_route = raw_phy_static[:, static_count * self.nmul :]

        return phy_dy, phy_static, phy_route

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[Union[None, torch.Tensor], torch.Tensor],
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """前向传播"""
        x = x_dict["x_phy"]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        # 解包参数
        phy_dy, phy_static, phy_route = self.unpack_parameters(parameters)
        self.routing_param_dict = self._descale_routing_params(phy_route)

        n_grid = x.size(1)

        # 初始化状态
        soil_storage = self._init_state(n_grid, "soil_storage")
        gw_storage = self._init_state(n_grid, "gw_storage")
        nash_storage = self._init_nash_storage(n_grid)

        # 反缩放参数
        static_names = [
            p for p in self.phy_param_names if p not in self.dynamic_params
        ]
        if phy_dy is not None:
            phy_dy_dict = self._descale_dynamic_params(phy_dy, self.dynamic_params)
        else:
            phy_dy_dict = {}
        phy_static_dict = self._descale_params(
            phy_static, static_names, self.parameter_bounds
        )

        return self._run_model(
            x, soil_storage, gw_storage, nash_storage,
            phy_dy_dict, phy_static_dict
        )

    def _run_model(
        self,
        forcing: torch.Tensor,
        soil_storage: torch.Tensor,
        gw_storage: torch.Tensor,
        nash_storage: torch.Tensor,
        dy_params: dict,
        static_params: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """CFE 模型核心计算（使用 JIT 优化的时间步循环）"""
        n_steps, n_grid = forcing.shape[:2]

        # 提取并扩展驱动数据
        P = (
            forcing[:, :, self.variables.index("prcp")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        PET = (
            forcing[:, :, self.variables.index("pet")]
            .unsqueeze(-1)
            .repeat(1, 1, self.nmul)
        )

        # 准备参数：对于动态参数保持 (T, B, E)，静态参数保持 (B, E)
        def get_param(name: str) -> torch.Tensor:
            if name in dy_params:
                return dy_params[name]  # (T, B, E)
            else:
                return static_params[name]  # (B, E)
        
        # 调用 JIT 优化的时间步循环
        (
            Qsim_out, Qsurf_out, Qlat_out, Qgw_out, AET_out,
            soil_storage_out, gw_storage_out, infiltration_out, 
            percolation_out, runoff_out,
            soil_storage_final, gw_storage_final
        ) = cfe_timestep_loop(
            P, PET,
            soil_storage, gw_storage,
            get_param("schaake_const"),
            get_param("smcmax"),
            get_param("soil_depth"),
            get_param("wltsmc"),
            get_param("satpsi"),
            get_param("bb"),
            get_param("mult"),
            get_param("satdk"),
            get_param("slop"),
            get_param("max_gw_storage"),
            get_param("Cgw"),
            get_param("expon"),
            get_param("K_nash"),
            nash_storage,
            self.num_nash_reservoirs,
            self.nearzero,
            self.timestep_d,
        )

        # 处理初始化模式
        if self.initialize:
            return soil_storage_final, gw_storage_final

        output = (
            Qsim_out,
            Qsurf_out,
            Qlat_out,
            Qgw_out,
            AET_out,
            soil_storage_out,
            gw_storage_out,
            infiltration_out,
            percolation_out,
            runoff_out,
            PET,
        )

        return self._finalize_output(
            output,
            n_steps,
            n_grid,
        )

    def _apply_averaging(
        self, Qsimmu: torch.Tensor
    ) -> torch.Tensor:
        """多模型平均
        
        Parameters
        ----------
        Qsimmu : torch.Tensor
            各模型的流量输出，形状: (T, B, E) 
            T=时间步, B=流域数, E=模型数(nmul)
            
        Returns
        -------
        torch.Tensor
            平均后的流量，形状: (T, B)
        """
        return Qsimmu.mean(-1)

    def _apply_routing(
        self, Qsim: torch.Tensor, Q_components: list, n_steps: int, n_grid: int
    ) -> tuple:
        """应用汇流演算"""
        UH = uh_gamma(
            self.routing_param_dict["rout_a"].repeat(n_steps, 1).unsqueeze(-1),
            self.routing_param_dict["rout_b"].repeat(n_steps, 1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])

        def route(q):
            rf = q.mean(-1, keepdim=True).permute([1, 2, 0])
            return uh_conv(rf, UH).permute([2, 0, 1])

        rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
        Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
        Qsurf_rout, Qlat_rout, Qgw_rout = [route(q) for q in Q_components]
        return Qsrout, Qsurf_rout, Qlat_rout, Qgw_rout

    def _finalize_output(
        self,
        output,
        n_steps,
        n_grid,
    ) -> dict:
        """整理最终输出"""
        (
            Qsim_out,
            Qsurf_out,
            Qlat_out,
            Qgw_out,
            AET_out,
            soil_storage_out,
            gw_storage_out,
            infiltration_out,
            percolation_out,
            runoff_out,
            PET,
        ) = output
        
        Qsimavg = self._apply_averaging(Qsim_out)
        Qsim = Qsimavg
        Qs, Qsurf_rout, Qlat_rout, Qgw_rout = self._apply_routing(
            Qsim, [Qsurf_out, Qlat_out, Qgw_out], n_steps, n_grid
        )

        # 构建输出字典
        result = {
            "streamflow": Qs,
            "surface_runoff": Qsurf_rout,
            "lateral_flow": Qlat_rout,
            "groundwater_flow": Qgw_rout,
            "AET_hydro": AET_out.mean(-1, keepdim=True),
            "PET_hydro": PET.mean(-1, keepdim=True),
            "surface_runoff_no_rout": Qsurf_out.mean(-1, keepdim=True),
            "lateral_flow_no_rout": Qlat_out.mean(-1, keepdim=True),
            "groundwater_flow_no_rout": Qgw_out.mean(-1, keepdim=True),
            "soil_storage": soil_storage_out.mean(-1, keepdim=True),
            "gw_storage": gw_storage_out.mean(-1, keepdim=True),
            "infiltration": infiltration_out.mean(-1, keepdim=True),
            "percolation": percolation_out.mean(-1, keepdim=True),
            "Qsimmu": Qsim_out,
        }

        # 裁剪预热期
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模型配置
    config = {
        "warm_up": 365,
        "warm_up_states": False,
        "nmul": 16,
        "nearzero": 1e-5,
        "num_nash_reservoirs": 3,
        "timestep_d": 1.0,
    }
    
    # 初始化模型
    model = Cfe(config=config, device=device)
    model.to(device)
    
    print(f"Model name: {model.name}")
    print(f"nmul: {model.nmul}")
    print(f"learnable_param_count: {model.learnable_param_count}")
    print(f"  - learnable_param_count1 (dynamic): {model.learnable_param_count1}")
    print(f"  - learnable_param_count2 (static + routing): {model.learnable_param_count2}")
    print(f"Parameters: {model.phy_param_names}")
    
    # 模拟输入数据
    batch_size = 10
    n_steps = 730
    n_features = 2  # prcp, pet
    
    # 物理输入
    x_phy = torch.rand(n_steps, batch_size, n_features, device=device)
    x_dict = {"x_phy": x_phy}
    
    # 神经网络输出的参数
    # dynamic params: None (不使用动态参数)
    # static params: (B, learnable_param_count2)
    raw_phy_dy = None
    raw_phy_static = torch.rand(batch_size, model.learnable_param_count2, device=device)
    parameters = (raw_phy_dy, raw_phy_static)
    
    print(f"\nInput shapes:")
    print(f"  x_phy: {x_phy.shape}")
    print(f"  raw_phy_static: {raw_phy_static.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x_dict, parameters)
    
    print(f"\nOutput shapes:")
    for key, value in output.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
    
    print("\n✓ Test passed!")
