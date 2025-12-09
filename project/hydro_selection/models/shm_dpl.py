"""
SHM 水文模型 (DPL 版本)

基于 SHM 概念式水文模型的实现，支持深度学习参数化。

SHM (Simple Hydrological Model) 参考文献:
    Ehret, U., et al.: Adaptive clustering: reducing the computational costs of 
    distributed (hydrological) modelling by exploiting time-variable similarity 
    among model elements, HESS, 24, 4389–4411, 2020.

Author: chooron
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from .core import shm_timestep_loop


class Shm(torch.nn.Module):
    """
    SHM 水文模型 (DPL 版本)

    简单水文模型实现，支持多模型集成和深度学习参数化。

    模型结构:
    - 积雪模块: 度日法融雪
    - 快速流模块: 超渗产流
    - 非饱和带模块: 蓄满产流
    - 壤中流模块: 线性水库
    - 基流模块: 线性水库

    Parameters
    ----------
    config : dict, optional
        模型配置字典
    device : torch.device, optional
        运行设备
    """

    # 物理参数边界
    PARAM_BOUNDS = {
        "dd": [0.0, 10.0],       # 度日因子 [mm/°C/day]
        "f_thr": [10.0, 60.0],   # 快速流阈值 [mm]
        "sumax": [20.0, 700.0],  # 非饱和带最大蓄水量 [mm]
        "beta": [1.0, 6.0],      # 非饱和带形状参数 [-]
        "perc": [0.0, 1.0],      # 下渗比例 [-]
        "kf": [1.0, 20.0],       # 快速流退水系数 [day]
        "ki": [1.0, 100.0],      # 壤中流退水系数 [day]
        "kb": [10.0, 1000.0],    # 基流退水系数 [day]
    }
    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    # 初始状态
    INITIAL_STATES = {
        "ss": 0.0,   # 积雪 [mm]
        "sf": 1.0,   # 快速流蓄水 [mm]
        "su": 5.0,   # 非饱和带蓄水 [mm]
        "si": 10.0,  # 壤中流蓄水 [mm]
        "sb": 15.0,  # 基流蓄水 [mm]
    }

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # 默认配置
        self.name = "SHM"
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.variables = ["prcp", "tmin", "tmax", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1

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
        ss = self._init_state(n_grid, "ss")
        sf = self._init_state(n_grid, "sf")
        su = self._init_state(n_grid, "su")
        si = self._init_state(n_grid, "si")
        sb = self._init_state(n_grid, "sb")
        states = [ss, sf, su, si, sb]

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

        return self._run_model(x, states, phy_dy_dict, phy_static_dict)

    def _run_model(
        self,
        forcing: torch.Tensor,
        states: list,
        dy_params: dict,
        static_params: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """SHM 模型核心计算（使用 JIT 优化的时间步循环）"""
        ss, sf, su, si, sb = states
        n_steps, n_grid = forcing.shape[:2]

        # 提取并扩展驱动数据
        P = (
            forcing[:, :, self.variables.index("prcp")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        T_min = (
            forcing[:, :, self.variables.index("tmin")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        T_max = (
            forcing[:, :, self.variables.index("tmax")]
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
            Qsim_out, Qf_out, Qi_out, Qb_out, AET_out,
            ss_out, sf_out, su_out, si_out, sb_out,
            ss_final, sf_final, su_final, si_final, sb_final
        ) = shm_timestep_loop(
            P, T_min, T_max, PET,
            ss, sf, su, si, sb,
            get_param("dd"),
            get_param("f_thr"),
            get_param("sumax"),
            get_param("beta"),
            get_param("perc"),
            get_param("kf"),
            get_param("ki"),
            get_param("kb"),
            self.nearzero,
        )

        # 处理初始化模式
        if self.initialize:
            return ss_final, sf_final, su_final, si_final, sb_final

        output = (
            Qsim_out,
            Qf_out,
            Qi_out,
            Qb_out,
            AET_out,
            ss_out,
            sf_out,
            su_out,
            si_out,
            sb_out,
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
        Qf_rout, Qi_rout, Qb_rout = [route(q) for q in Q_components]
        return Qsrout, Qf_rout, Qi_rout, Qb_rout

    def _finalize_output(
        self,
        output,
        n_steps,
        n_grid,
    ) -> dict:
        """整理最终输出"""
        (
            Qsim_out,
            Qf_out,
            Qi_out,
            Qb_out,
            AET_out,
            ss_out,
            sf_out,
            su_out,
            si_out,
            sb_out,
            PET,
        ) = output
        
        Qsimavg = self._apply_averaging(Qsim_out)
        Qsim = Qsimavg
        Qs, Qf_rout, Qi_rout, Qb_rout = self._apply_routing(
            Qsim, [Qf_out, Qi_out, Qb_out], n_steps, n_grid
        )

        # 构建输出字典
        result = {
            "streamflow": Qs,
            "fastflow": Qf_rout,
            "interflow": Qi_rout,
            "baseflow": Qb_rout,
            "AET_hydro": AET_out.mean(-1, keepdim=True),
            "PET_hydro": PET.mean(-1, keepdim=True),
            "SWE": ss_out.mean(-1, keepdim=True),
            "fastflow_no_rout": Qf_out.mean(-1, keepdim=True),
            "interflow_no_rout": Qi_out.mean(-1, keepdim=True),
            "baseflow_no_rout": Qb_out.mean(-1, keepdim=True),
            "snow_storage": ss_out.mean(-1, keepdim=True),
            "fastflow_storage": sf_out.mean(-1, keepdim=True),
            "unsaturated_storage": su_out.mean(-1, keepdim=True),
            "interflow_storage": si_out.mean(-1, keepdim=True),
            "baseflow_storage": sb_out.mean(-1, keepdim=True),
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
    }
    
    # 初始化模型
    model = Shm(config=config, device=device)
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
    n_features = 4  # prcp, tmin, tmax, pet
    
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
