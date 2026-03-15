"""
DiffBlendV2 - 可微分公式混合水文模型 (两阶段训练版)

相比 V1 的主要变化:
  - 权重 logits 从参数网络输出中分离，改为模型内部的独立 nn.Parameter
  - 新增 self.stage 控制训练阶段:
    stage=1: 预训练阶段，权重固定为均匀分布，仅训练物理参数
    stage=2: 权重训练阶段，通过 Gumbel-Softmax 学习过程权重
  - learnable_param_count 不再包含权重 logits（由参数网络输出物理+路由参数）

过程及选项数 (与 V1 相同):
  雨雪分割(3) · 雪平衡(3) · 入渗(3) · 蒸发(3) · 快速流(3) · 基流(2)
  渗漏(固定) · Gamma UH 路由(固定)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.hydrodl2 import uh_conv, uh_gamma

# 复用 V1 的公式函数和工具
from project.blend_formula.models.diff_blend_v1 import (
    activate_weights,
    safe_pow,
    rainsnow_hbv, rainsnow_dingman, rainsnow_threshold,
    snobal_simple, snobal_hbv, snobal_hmets,
    inf_hmets, inf_vic_arno, inf_hbv,
    soilevap_all, soilevap_linear, soilevap_vic,
    quick_linear, quick_vic, quick_topmodel,
    base_linear, base_power,
    perc_linear,
    PROCESS_OPTIONS, TOTAL_WEIGHT_LOGITS,
    diff_blend_step,
)

class DiffBlendV2(nn.Module):
    """可微分公式混合水文模型 V2 (两阶段训练版)

    与 V1 物理等价，核心区别：
    - 权重 logits 作为模型内部独立参数 (nn.Parameter)
    - stage=1 时权重固定为均匀分布，stage=2 时通过 Gumbel-Softmax 学习
    - learnable_param_count 仅包含物理参数 + 路由参数（不含权重 logits）
    """

    PARAM_BOUNDS = {
        "x1":  [0.0, 1.0],    # hmets_runoff_coeff
        "x2":  [0.3, 3.0],    # b_exp (VIC_ARNO)
        "x3":  [0.5, 3.0],    # hbv_beta
        "x4":  [-5.0, -1.0],  # log_k_quick
        "x5":  [0.001, 0.3],  # (unused, reserved)
        "x6":  [0.3, 1.5],    # n_quick
        "x7":  [0.5, 3.0],    # topmodel_lambda
        "x8":  [0.0, 3.0],    # pet_correction
        "x9":  [0.0, 0.05],   # sat_wilt
        "x10": [0.0, 0.45],   # delta_fc
        "x11": [-5.0, -2.0],  # log_k_base
        "x12": [0.5, 2.0],    # n_base
        "x13": [0.0, 0.1],    # swi_min
        "x14": [0.01, 0.3],   # delta_swi_max
        "x15": [0.005, 0.1],  # swi_reduct
        "x16": [-5.0, 2.0],   # refreeze_temp
        "x17": [0.3, 1.0],    # refreeze_exp
        "x18": [0.0, 5.0],    # refreeze_factor
        "x19": [0.0, 0.4],    # snow_swi_hbv
        "x20": [0.3, 20.0],   # gamma_shape_surf
        "x21": [0.01, 5.0],   # gamma_scale_surf
        "x22": [0.5, 13.0],   # gamma_shape_delay
        "x23": [0.15, 1.5],   # gamma_scale_delay
        "x24": [1.5, 3.0],    # min_melt_factor
        "x25": [0.0, 5.0],    # delta_melt_factor
        "x26": [-1.0, 1.0],   # dd_melt_temp
        "x27": [0.01, 0.2],   # dd_aggradation
        "x28": [0.00001, 0.02],# perc_coeff_top
        "x29": [0.0, 0.5],    # thickness_top
        "x30": [0.0, 2.0],    # thickness_phreatic
        "x31": [-3.0, 3.0],   # rainsnow_temp
        "x32": [0.5, 4.0],    # rainsnow_delta
        "x33": [0.8, 1.2],    # rain_correction
        "x34": [0.8, 1.2],    # snow_correction
        "x35": [0.0, 0.02],   # perc_coeff_phreatic
        "x36": [0.3, 3.0],    # soilevap_vic_gamma
    }
    ROUTING_BOUNDS = {"rout_a": [0.0, 2.9], "rout_b": [0.0, 6.5]}

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.name = "DiffBlendV2"
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        self.weight_method = "gumbel_softmax"
        self.tau = 1.0
        self.balance_window = 30  # 滑动窗口天数

        # 两阶段训练控制: 1=预训练(均匀权重), 2=权重精调(Gumbel-Softmax)
        self.stage = 1

        self.param_names = list(self.PARAM_BOUNDS.keys())
        self.n_phy = len(self.param_names)

        # learnable_param_count 包含物理参数 + 路由参数 + 公式权重 logits
        self.learnable_param_count = (
            self.n_phy * self.nmul + len(self.ROUTING_BOUNDS) + TOTAL_WEIGHT_LOGITS
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config:
            self._load_config(config)

        self._step_fn = torch.compile(diff_blend_step)

    def _load_config(self, config: Dict[str, Any]) -> None:
        for k in ["warm_up", "warm_up_states", "variables", "nearzero", "nmul",
                  "weight_method", "tau", "balance_window"]:
            if k in config:
                setattr(self, k, config[k])
        # learnable_param_count 包含物理参数 + 路由参数 + 公式权重 logits
        self.learnable_param_count = (
            self.n_phy * self.nmul + len(self.ROUTING_BOUNDS) + TOTAL_WEIGHT_LOGITS
        )

    # ---------------------------------------------------------------
    # 参数解包 (unpack_parameters → _get_weights)
    # ---------------------------------------------------------------

    def unpack_parameters(self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]):
        """解包参数网络输出为物理参数、路由参数和公式权重。"""
        _, raw = parameters
        B = raw.shape[0]
        n_rout = len(self.ROUTING_BOUNDS)

        raw_phy  = raw[:, :self.n_phy * self.nmul]
        raw_rout = raw[:, self.n_phy * self.nmul: self.n_phy * self.nmul + n_rout]
        raw_w    = raw[:, self.n_phy * self.nmul + n_rout:]  # [B, TOTAL_WEIGHT_LOGITS]

        # 物理参数: sigmoid 激活 + 线性缩放到边界
        act = torch.sigmoid(raw_phy).view(B, self.n_phy, self.nmul)
        p = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.PARAM_BOUNDS[name]
            p[name] = act[:, i, :] * (hi - lo) + lo

        # 路由参数
        act_r = torch.sigmoid(raw_rout)
        rout = {}
        for i, (name, (lo, hi)) in enumerate(self.ROUTING_BOUNDS.items()):
            rout[name] = act_r[:, i] * (hi - lo) + lo

        # 过程权重: 根据 stage 决定来源
        w = self._get_weights(raw_w, B)

        return p, w, rout

    def _get_weights(self, raw_w: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """根据训练阶段获取过程权重。

        stage=1: 均匀分布权重，raw_w 不参与梯度计算（weight head 已被冻结）
        stage=2: 使用网络预测的流域特异性 raw_w，通过 Gumbel-Softmax 转换
        """
        w = {}
        idx = 0
        for proc, n_opt in PROCESS_OPTIONS.items():
            if self.stage == 1:
                uniform = torch.ones(batch_size, n_opt, device=self.device) / n_opt
                w[proc] = uniform
            else:
                logits = raw_w[:, idx: idx + n_opt]  # [B, n_opt] 每流域不同
                w[proc] = activate_weights(
                    logits, self.weight_method, self.tau, self.training
                )
            idx += n_opt
        return w

    def _build_step_params(self, p: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """参数转换，生成单步所需的 dict（短键名）。"""
        t = {k: v for k, v in p.items()}
        t["k_quick"]  = 10.0 ** t["x4"]
        t["k_base"]   = 10.0 ** t["x11"]
        t["fc"]       = t["x9"] + t["x10"]
        t["swi_max"]  = t["x13"] + t["x14"]
        t["max_melt"] = t["x24"] + t["x25"]
        t["Smax_top"] = t["x29"] * 1000.0 + 1.0
        t["Smax_ph"]  = t["x30"] * 1000.0 + 1.0
        return t

    @staticmethod
    def _rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
        """对时间维做因果滑动累积和。"""
        T, B, n_opt = x.shape
        x_perm = x.permute(1, 2, 0).reshape(B * n_opt, 1, T)
        kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
        x_padded = F.pad(x_perm, (window - 1, 0))
        out = F.conv1d(x_padded, kernel)
        return out.reshape(B, n_opt, T).permute(2, 0, 1)

    def _apply_routing(self, Q: torch.Tensor, rout_a: torch.Tensor,
                       rout_b: torch.Tensor) -> torch.Tensor:
        n_steps, n_grid = Q.shape
        UH = uh_gamma(
            rout_a.unsqueeze(0).expand(n_steps, -1).unsqueeze(-1),
            rout_b.unsqueeze(0).expand(n_steps, -1).unsqueeze(-1),
            lenF=15,
        ).permute([1, 2, 0])
        rf = Q.unsqueeze(-1).permute([1, 2, 0])
        return uh_conv(rf, UH).permute([2, 0, 1]).squeeze(-1)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                parameters: Tuple[Optional[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播。

        V2 变化: 权重来自模型内部参数，stage 控制权重行为。
        """
        x = x_dict["x_phy"]
        n_steps, n_grid, _ = x.shape

        p_raw, weights, rout = self.unpack_parameters(parameters)
        p = self._build_step_params(p_raw)

        vi = {v: i for i, v in enumerate(self.variables)}
        P   = x[:, :, vi["prcp"]].unsqueeze(2).expand(-1, -1, self.nmul)
        T   = x[:, :, vi["tmean"]].unsqueeze(2).expand(-1, -1, self.nmul)
        PET = x[:, :, vi["pet"]].unsqueeze(2).expand(-1, -1, self.nmul)

        # 初始化状态
        z = lambda: torch.full((n_grid, self.nmul), self.nearzero, device=self.device)
        S_snow, S_liq, S_top, S_phreatic = z(), z(), z(), z()
        S_cum = torch.zeros(n_grid, self.nmul, device=self.device)

        # 权重 broadcast: [B, n_opt] -> [B, 1, n_opt]
        w_rs  = weights["rainsnow"].unsqueeze(1)
        w_sn  = weights["snowbal"].unsqueeze(1)
        w_inf = weights["infiltration"].unsqueeze(1)
        w_ev  = weights["evaporation"].unsqueeze(1)
        w_qf  = weights["quickflow"].unsqueeze(1)
        w_bf  = weights["baseflow"].unsqueeze(1)

        Q_sim = torch.zeros(n_steps, n_grid, self.nmul, device=self.device)

        # 预分配 balance buffer
        balance_buf_lists = {
            "snow_outflow": [], "infiltration": [],
            "evaporation": [], "quickflow": [], "baseflow": [],
        }

        for t in range(n_steps):
            Q_t, S_snow, S_liq, S_cum, S_top, S_phreatic, balance_opts = self._step_fn(
                P[t], T[t], PET[t],
                S_snow, S_liq, S_cum, S_top, S_phreatic,
                p, w_rs, w_sn, w_inf, w_ev, w_qf, w_bf,
                self.nearzero,
            )
            Q_sim[t] = Q_t
            for proc_name, val in balance_opts.items():
                balance_buf_lists[proc_name].append(val)

        # stack balance buffers -> [T, B, n_options]
        balance_stacked = {
            proc_name: torch.stack(lst, dim=0)
            for proc_name, lst in balance_buf_lists.items()
        }

        Q_mean = Q_sim.mean(-1)
        Qrouted = self._apply_routing(Q_mean, rout["rout_a"], rout["rout_b"])

        result: Dict[str, torch.Tensor] = {"streamflow": Qrouted}

        # 过程权重 (扩展到时间维，供分析和监控)
        for proc, n_opt in PROCESS_OPTIONS.items():
            for i in range(n_opt):
                result[f"w_{proc}_{i}"] = (
                    weights[proc][:, i].unsqueeze(0).expand(n_steps, -1)
                )

        # 公式累积平衡约束序列
        for proc_name, buf in balance_stacked.items():
            rolled = self._rolling_sum(buf, self.balance_window)
            mean_val = rolled.mean(dim=-1, keepdim=True)
            std_scale = (rolled.var(dim=-1, keepdim=True) + 1e-3).sqrt()
            abs_scale = rolled.abs().mean(dim=-1, keepdim=True) + 1e-2
            scale = torch.max(std_scale, abs_scale).detach()
            scale = torch.clamp(scale, min=0.5)

            n_opt = buf.shape[-1]
            for j in range(n_opt):
                normalized_diff = (rolled[:, :, j] - mean_val.squeeze(-1)) / scale.squeeze(-1)
                normalized_diff = torch.clamp(normalized_diff, min=-5.0, max=5.0)
                result[f"balance_{proc_name}_{j}"] = normalized_diff

        if not self.warm_up_states:
            cutoff = self.warm_up if self.pred_cutoff == 0 else self.pred_cutoff
            result = {k: v[cutoff:] for k, v in result.items()}

        return result
