"""
HBV 水文模型

基于 HBV 概念式水文模型的实现。

支持两种后端:
- triton: 使用 Triton 加速的前向和手写 backward (可能有梯度问题)
- autograd: 使用纯 PyTorch + torch.compile，自动微分保证梯度正确

Author: chooron
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma

from project.triton_accelerate.models.hbv_triton_split import hbv_step_split
from project.triton_accelerate.models.hbv_triton_fused import hbv_step_fused
from project.triton_accelerate.models.hbv_triton_autograd import (
    hbv_step_pytorch,
    hbv_step_compiled,
    hbv_step_jit,
)


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
    # 修复: 调整边界以避免数值不稳定（除以零、log(0)等问题）
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
    }
    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "triton",  # "triton" 或 "autograd"
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
        
        # 后端选择 - 优先从配置中读取
        # 支持: "triton", "fused", "autograd"
        if config is not None and "backend" in config:
            self.backend = config["backend"]
        else:
            self.backend = backend
        
        # 参数边界
        self.parameter_bounds = self.PARAM_BOUNDS
        self.routing_parameter_bounds = self.ROUTING_BOUNDS

        # 设备配置
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 选择计算后端
        if self.backend == "triton":
            self.step_fn = self._step_triton
        elif self.backend == "fused":
            self.step_fn = self._step_fused
        elif self.backend == "compile":
            self.step_fn = self._step_compile
        elif self.backend == "jit":
            self.step_fn = self._step_autograd
        else:
            # autograd 后端使用 torch.compile 加速
            self.step_fn = self._step_compile

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
        """HBV 模型核心计算（支持 Triton 或 autograd 后端）"""
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states
        n_steps, n_grid = forcing.shape[:2]

        # 提取并扩展驱动数据 (T, B, E)
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

        def get_param(name: str) -> torch.Tensor:
            p = dy_params[name] if name in dy_params else static_params[name]
            return p

        # 获取参数
        parTT = get_param("parTT")
        parCFMAX = get_param("parCFMAX")
        parCFR = get_param("parCFR")
        parCWH = get_param("parCWH")
        parFC = get_param("parFC")
        parBETA = get_param("parBETA")
        parLP = get_param("parLP")
        parBETAET = get_param("parBETAET")
        parC = get_param("parC")
        parPERC = get_param("parPERC")
        parK0 = get_param("parK0")
        parK1 = get_param("parK1")
        parK2 = get_param("parK2")
        parUZL = get_param("parUZL")

        # 存储输出
        Qsim_list = []
        


        # 时间步循环
        for t in range(n_steps):
            # 获取当前时刻的参数（支持动态参数）
            tt_t = parTT[t] if parTT.dim() == 3 else parTT
            cfmax_t = parCFMAX[t] if parCFMAX.dim() == 3 else parCFMAX
            cfr_t = parCFR[t] if parCFR.dim() == 3 else parCFR
            cwh_t = parCWH[t] if parCWH.dim() == 3 else parCWH
            fc_t = parFC[t] if parFC.dim() == 3 else parFC
            beta_t = parBETA[t] if parBETA.dim() == 3 else parBETA
            lp_t = parLP[t] if parLP.dim() == 3 else parLP
            betaet_t = parBETAET[t] if parBETAET.dim() == 3 else parBETAET
            c_t = parC[t] if parC.dim() == 3 else parC
            perc_t = parPERC[t] if parPERC.dim() == 3 else parPERC
            k0_t = parK0[t] if parK0.dim() == 3 else parK0
            k1_t = parK1[t] if parK1.dim() == 3 else parK1
            k2_t = parK2[t] if parK2.dim() == 3 else parK2
            uzl_t = parUZL[t] if parUZL.dim() == 3 else parUZL

            # 调用单步计算
            SNOWPACK, MELTWATER, SM, SUZ, SLZ, Q = self.step_fn(
                P[t], T[t], PET[t],
                SNOWPACK, MELTWATER, SM, SUZ, SLZ,
                tt_t, cfmax_t, cfr_t, cwh_t,
                fc_t, beta_t, lp_t, betaet_t, c_t,
                perc_t, k0_t, k1_t, k2_t, uzl_t,
            )
            Qsim_list.append(Q)

        # 堆叠输出 (T, B, E)
        Qsim_out = torch.stack(Qsim_list, dim=0)

        # 处理初始化模式
        if self.initialize:
            return (SNOWPACK, MELTWATER, SM, SUZ, SLZ)

        return self._finalize_output(Qsim_out, n_steps, n_grid)

    def _step_triton(self, p, t_val, pet, snow, melt, sm, suz, slz,
                     tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                     perc, k0, k1, k2, uzl):
        """Triton 后端单步计算（分离的 Snow/Soil/Routing kernels）"""
        return hbv_step_split(
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
        )

    def _step_fused(self, p, t_val, pet, snow, melt, sm, suz, slz,
                    tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                    perc, k0, k1, k2, uzl):
        """融合 Triton 后端单步计算（整体 forward/backward kernel）"""
        return hbv_step_fused(
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
        )

    def _step_compile(self, p, t_val, pet, snow, melt, sm, suz, slz,
                       tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                       perc, k0, k1, k2, uzl):
        """Autograd 后端单步计算 (使用 torch.compile 加速)"""
        return hbv_step_compiled(
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
            nearzero=self.nearzero,
        )
        
    def _step_jit(self, p, t_val, pet, snow, melt, sm, suz, slz,
                       tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                       perc, k0, k1, k2, uzl):
        """Autograd 后端单步计算 (使用 torch.compile 加速)"""
        return hbv_step_jit(
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
            nearzero=self.nearzero,
        )        
    def _step_autograd(self, p, t_val, pet, snow, melt, sm, suz, slz,
                       tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                       perc, k0, k1, k2, uzl):
        """Autograd 后端单步计算 (使用 torch.compile 加速)"""
        return hbv_step_pytorch(
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
            nearzero=self.nearzero,
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
        self, Qsim: torch.Tensor, n_steps: int, n_grid: int
    ) -> torch.Tensor:
        """简化的汇流演算（只处理总径流）"""
        UH = uh_gamma(
            self.routing_param_dict["rout_a"].repeat(n_steps, 1).unsqueeze(-1),
            self.routing_param_dict["rout_b"].repeat(n_steps, 1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])

        rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
        Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
        return Qsrout

    def _finalize_output(
        self,
        Qsim_out: torch.Tensor,
        n_steps: int,
        n_grid: int,
    ) -> dict:
        """整理最终输出（简化版本，只返回径流）"""
        # 多模型平均
        Qsimavg = self._apply_averaging(Qsim_out)

        # 汇流演算
        Qs = self._apply_routing(Qsimavg, n_steps, n_grid)

        # 构建输出字典（简化版本）
        result = {
            "streamflow": Qs,
            "Qsimmu": Qsim_out,
        }

        # 裁剪预热期
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
