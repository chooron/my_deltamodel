"""
HBV 水文模型

基于 HBV 概念式水文模型的实现。

Author: chooron
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from project.hydro_selection.models.layers.jit_core import hbv_timestep_loop


class Hbv(torch.nn.Module):
    """
    HBV 水文模型

    经典 HBV 物理模型实现，支持多模型集成。

    Parameters
    ----------
    config : dict, optional
        模型配置字典
    device : torch.device, optional
        运行设备
    """

    # 物理参数边界
    PARAM_BOUNDS = {
        "parBETA": [1.0, 6.0],
        "parFC": [50, 1000],
        "parK0": [0.05, 0.9],
        "parK1": [0.01, 0.5],
        "parK2": [0.001, 0.2],
        "parLP": [0.2, 1],
        "parPERC": [0, 10],
        "parUZL": [0, 100],
        "parTT": [-2.5, 2.5],
        "parCFMAX": [0.5, 10],
        "parCFR": [0, 0.1],
        "parCWH": [0, 0.2],
        "parBETAET": [0.3, 5],
        "parC": [0, 1],
        "parRT": [0, 20],
        "parAC": [0, 2500],
    }
    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # 默认配置
        self.name = "HBV"
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1

        # 参数边界
        self.parameter_bounds = self.PARAM_BOUNDS
        self.routing_parameter_bounds = self.ROUTING_BOUNDS

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

    def _init_state(self, n_grid: int) -> torch.Tensor:
        """初始化状态张量"""
        return (
            torch.zeros(
                [n_grid, self.nmul], dtype=torch.float32, device=self.device
            )
            + self.nearzero
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
        states = [self._init_state(n_grid) for _ in range(5)]

        # 反缩放参数
        static_names = [
            p for p in self.phy_param_names if p not in self.dynamic_params
        ]
        if phy_dy is not None:
            phy_dy_dict = self._descale_dynamic_params(
                phy_dy, self.dynamic_params
            )
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
        """HBV 模型核心计算（使用 JIT 优化的时间步循环）"""
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states
        n_steps, n_grid = forcing.shape[:2]

        # 提取并扩展驱动数据
        P = (
            forcing[:, :, self.variables.index("prcp")]
            .unsqueeze(2)
            .repeat(1, 1, self.nmul)
        )
        T = (
            forcing[:, :, self.variables.index("tmean")]
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
        hbv_timestep_loop_fast = torch.compile(hbv_timestep_loop)
        # 调用 JIT 优化的时间步循环
        (
            Qsim_out,
            Q0_out,
            Q1_out,
            Q2_out,
            AET_out,
            recharge_out,
            excs_out,
            evapfactor_out,
            tosoil_out,
            PERC_out,
            SWE_out,
            SM_out,
            capillary_out,
            soil_wetness_out,
            SNOWPACK_final,
            MELTWATER_final,
            SM_final,
            SUZ_final,
            SLZ_final,
        ) = hbv_timestep_loop_fast(
            P,
            T,
            PET,
            SNOWPACK=SNOWPACK,
            MELTWATER=MELTWATER,
            SM=SM,
            SUZ=SUZ,
            SLZ=SLZ,
            parTT=get_param("parTT"),
            parCFMAX=get_param("parCFMAX"),
            parCFR=get_param("parCFR"),
            parCWH=get_param("parCWH"),
            parFC=get_param("parFC"),
            parBETA=get_param("parBETA"),
            parLP=get_param("parLP"),
            parBETAET=get_param("parBETAET"),
            parC=get_param("parC"),
            parPERC=get_param("parPERC"),
            parK0=get_param("parK0"),
            parK1=get_param("parK1"),
            parK2=get_param("parK2"),
            parUZL=get_param("parUZL"),
            nearzero=self.nearzero,
        )

        # 处理初始化模式
        if self.initialize:
            return (
                SNOWPACK_final,
                MELTWATER_final,
                SM_final,
                SUZ_final,
                SLZ_final,
            )

        output = (
            Qsim_out,
            Q0_out,
            Q1_out,
            Q2_out,
            AET_out,
            recharge_out,
            excs_out,
            evapfactor_out,
            tosoil_out,
            PERC_out,
            SWE_out,
            SM_out,
            capillary_out,
            soil_wetness_out,
            PET,
        )

        return self._finalize_output(
            output,
            n_steps,
            n_grid,
        )

    def _apply_averaging(self, Qsimmu: torch.Tensor) -> torch.Tensor:
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
        Q0_rout, Q1_rout, Q2_rout = [route(q) for q in Q_components]
        return Qsrout, Q0_rout, Q1_rout, Q2_rout

    def _finalize_output(
        self,
        output,
        n_steps,
        n_grid,
    ) -> dict:
        """整理最终输出"""
        (
            Qsim_out,
            Q0_out,
            Q1_out,
            Q2_out,
            AET_out,
            recharge_out,
            excs_out,
            evapfactor_out,
            tosoil_out,
            PERC_out,
            SWE_out,
            SM_out,
            capillary_out,
            soil_wetness_out,
            PET,
        ) = output
        Qsimavg = self._apply_averaging(Qsim_out)
        Qsim = Qsimavg
        Qs, Q0_rout, Q1_rout, Q2_rout = self._apply_routing(
            Qsim, [Q0_out, Q1_out, Q2_out], n_steps, n_grid
        )

        # 构建输出字典
        result = {
            "streamflow": Qs,
            "srflow": Q0_rout,
            "ssflow": Q1_rout,
            "gwflow": Q2_rout,
            "AET_hydro": AET_out.mean(-1, keepdim=True),
            "AET_full": AET_out,
            "SM_full": SM_out,
            "soilwetness_full": soil_wetness_out,
            "tosoil_full": tosoil_out,
            "recharge_full": recharge_out,
            "PET_hydro": PET.mean(-1, keepdim=True),
            "SWE": SWE_out.mean(-1, keepdim=True),
            "srflow_no_rout": Q0_out.mean(-1, keepdim=True),
            "ssflow_no_rout": Q1_out.mean(-1, keepdim=True),
            "gwflow_no_rout": Q2_out.mean(-1, keepdim=True),
            "recharge": recharge_out.mean(-1, keepdim=True),
            "excs": excs_out.mean(-1, keepdim=True),
            "evapfactor": evapfactor_out.mean(-1, keepdim=True),
            "tosoil": tosoil_out.mean(-1, keepdim=True),
            "percolation": PERC_out.mean(-1, keepdim=True),
            "soilwater": SM_out.mean(-1, keepdim=True),
            "capillary": capillary_out.mean(-1, keepdim=True),
            "Qsimmu": Qsim_out,
        }

        # 裁剪预热期
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
