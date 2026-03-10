"""
DiffBlendV2 - 可微分公式混合水文模型 (精简高性能版)

相比 V1 的主要变化:
  - 删除所有 flux 记录 / balance buffer 基础设施，大幅减少内存分配
  - 删除 NamedTuple 包装层，直接传递 tensor，减少 Python 对象开销
  - 删除 _rolling_sum / balance loss 辅助逻辑（由调用方按需实现）
  - 删除 STEP_FLUX_KEYS / FLUX_OPTION_DIMS 等冗余常量
  - 权重 unsqueeze 移至循环外，避免重复操作
  - 保留全部物理公式和参数体系，与 V1 完全兼容

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


# ===================================================================
# 权重激活
# ===================================================================

def activate_weights(logits: torch.Tensor, method: str = "softmax",
                     tau: float = 1.0, training: bool = True) -> torch.Tensor:
    logits = torch.clamp(logits, -10.0, 10.0)
    if method == "gumbel_softmax":
        return F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1) if training \
               else F.softmax(logits / tau, dim=-1)
    if method == "sparsemax":
        return _sparsemax(logits)
    if method == "entmax15":
        return _entmax15(logits)
    return F.softmax(logits / tau, dim=-1)


def _sparsemax(logits: torch.Tensor) -> torch.Tensor:
    sorted_z, _ = torch.sort(logits, descending=True, dim=-1)
    n = logits.shape[-1]
    k = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
    cumsum = sorted_z.cumsum(-1)
    support = (1 + k * sorted_z) > cumsum
    k_z = support.sum(-1, keepdim=True).float()
    tau = (cumsum.gather(-1, (k_z - 1).long().clamp(min=0)) - 1) / k_z
    return F.relu(logits - tau)


def _entmax15(logits: torch.Tensor, n_iter: int = 25) -> torch.Tensor:
    lo = logits.min(-1, keepdim=True).values - 1
    hi = logits.max(-1, keepdim=True).values
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        s = F.relu(logits - mid).pow(2.0).sum(-1, keepdim=True)
        lo = torch.where(s > 1, mid, lo)
        hi = torch.where(s > 1, hi, mid)
    return F.relu(logits - (lo + hi) / 2).pow(2.0)


# ===================================================================
# 水文过程公式
# ===================================================================

# --- 雨雪分割 ---

def rainsnow_hbv(P, T, tt, tti, eps=1e-6):
    sf = torch.clamp((tt + tti / 2 - T) / (tti + eps), 0.0, 1.0)
    return P * sf, P * (1.0 - sf)

def rainsnow_dingman(P, T, ts):
    dc = F.relu(ts - T)
    dw = F.relu(T - ts)
    sf = torch.clamp(0.5 * (1.0
        + torch.exp(-2.2 * (dw + 1e-6).pow(1.3))
        - torch.exp(-2.2 * (dc + 1e-6).pow(1.3))), 0.0, 1.0)
    return P * sf, P * (1.0 - sf)

def rainsnow_threshold(P, T, tt):
    sf = torch.sigmoid(5.0 * (tt - T))
    return P * sf, P * (1.0 - sf)


# --- 雪平衡 ---

def snobal_simple(snowfall, rainfall, T, S_snow, S_cum, ddf, tt_melt, eps=1e-6):
    melt = torch.minimum(S_snow, ddf * F.relu(T - tt_melt))
    S_cum_new = torch.where(S_snow > eps, S_cum + melt, torch.zeros_like(S_cum))
    return melt + rainfall, S_snow + snowfall - melt, torch.zeros_like(S_snow), S_cum_new

def snobal_hbv(snowfall, rainfall, T, S_snow, S_liq, S_cum,
               ddf, tt_melt, cfr, tt_refreeze, swi, eps=1e-6):
    melt = torch.minimum(S_snow, ddf * F.relu(T - tt_melt))
    refreeze = torch.minimum(S_liq, cfr * F.relu(tt_refreeze - T))
    outflow = F.relu(S_liq + rainfall + melt - swi * S_snow)
    S_liq_new = F.relu(S_liq + melt + rainfall - refreeze - outflow)
    S_cum_new = torch.where(S_snow > eps, S_cum + melt, torch.zeros_like(S_cum))
    return outflow, S_snow + snowfall - melt + refreeze, S_liq_new, S_cum_new

def snobal_hmets(snowfall, rainfall, T, S_snow, S_liq, S_cum,
                 ddf_min, ddf_max, tt_melt, kf, tt_refreeze, refreeze_exp,
                 swi_min, swi_max, alpha_swi, dd_agg, eps=1e-6):
    ddf = torch.minimum(ddf_max, ddf_min * (1.0 + dd_agg * S_cum))
    melt = torch.minimum(S_snow, ddf * F.relu(T - tt_melt))
    refreeze = torch.minimum(S_liq, kf * (F.relu(tt_refreeze - T) + eps).pow(refreeze_exp))
    swi = torch.maximum(swi_min, swi_max * (1.0 - alpha_swi * S_cum))
    outflow = F.relu(S_liq + rainfall + melt - swi * S_snow)
    S_liq_new = F.relu(S_liq + melt + rainfall - refreeze - outflow)
    S_cum_new = torch.where(S_snow > eps, S_cum + melt, torch.zeros_like(S_cum))
    return outflow, S_snow + snowfall - melt + refreeze, S_liq_new, S_cum_new


# --- 入渗 ---

def inf_hmets(P_eff, S, Smax, c, eps=1e-6):
    return P_eff * F.relu(1.0 - c * S / (Smax + eps))

def inf_vic_arno(P_eff, S, Smax, b, eps=1e-6):
    ratio = torch.clamp(1.0 - S / (Smax + eps), 0.0, 1.0)
    return P_eff * (1.0 - ratio.clamp(min=0.01).pow(b))

def inf_hbv(P_eff, S, Smax, beta, eps=1e-6):
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    return P_eff * (1.0 - ratio.clamp(min=0.01).pow(beta))


# --- 蒸发 ---

def soilevap_all(PET, c, S, **_):
    return torch.minimum(PET * c, S)

def soilevap_linear(PET, c, S, S_tension, eps=1e-6):
    return torch.minimum(PET * c * torch.clamp(S / (S_tension + eps), 0.0, 1.0), S)

def soilevap_vic(PET, c, S, Smax, gamma, eps=1e-6):
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    return torch.minimum(PET * c * (1.0 - (1.0 - ratio).clamp(min=0.01).pow(gamma)), S)


# --- 快速流 ---

def quick_linear(S, k, **_):
    return S * (1.0 - torch.exp(-k))

def quick_vic(S, Smax, q_max, n, eps=1e-6):
    return torch.minimum(q_max * (S / (Smax + eps) + eps).clamp(0.0, 1.0).pow(n), S)

def quick_topmodel(S, Smax, q_max, n, lam, eps=1e-6):
    ratio = torch.clamp(S / (Smax + eps), 0.0, 1.0)
    return torch.minimum(q_max * torch.exp(-lam * (1.0 - ratio)), S)


# --- 基流 ---

def base_linear(S, k, **_):
    return S * (1.0 - torch.exp(-k))

def base_power(S, k, n, eps=1e-6):
    return torch.minimum(k * (S + eps).pow(n), S)


# --- 渗漏 (固定) ---

def perc_linear(S, k):
    return torch.minimum(k * S, S)


# ===================================================================
# 过程配置
# ===================================================================

PROCESS_OPTIONS = {
    "rainsnow":     3,
    "snowbal":      3,
    "infiltration": 3,
    "evaporation":  3,
    "quickflow":    3,
    "baseflow":     2,
}
TOTAL_WEIGHT_LOGITS = sum(PROCESS_OPTIONS.values())  # 17


# ===================================================================
# 单步计算 (纯函数，无 NamedTuple 包装)
# ===================================================================

def diff_blend_step(
    P_t, T_t, PET_t,
    S_snow, S_liq, S_cum, S_top, S_phreatic,
    p: Dict[str, torch.Tensor],
    w_rs, w_sn, w_inf, w_ev, w_qf, w_bf,
    eps: float = 1e-6,
):
    """单时间步，返回 (Q_total, S_snow, S_liq, S_cum, S_top, S_phreatic)。"""

    # 1. 雨雪分割
    sf1, rf1 = rainsnow_hbv(P_t, T_t, p["x31"], p["x32"])
    sf2, rf2 = rainsnow_dingman(P_t, T_t, p["x31"])
    sf3, rf3 = rainsnow_threshold(P_t, T_t, p["x31"])
    sc, rc = p["x34"], p["x33"]
    snowfall = (torch.stack([sf1, sf2, sf3], -1) * w_rs).sum(-1) * sc
    rainfall = (torch.stack([rf1, rf2, rf3], -1) * w_rs).sum(-1) * rc

    # 2. 雪平衡
    o1, Ss1, Sl1, Sc1 = snobal_simple(snowfall, rainfall, T_t, S_snow, S_cum,
                                       p["x24"], p["x26"], eps)
    o2, Ss2, Sl2, Sc2 = snobal_hbv(snowfall, rainfall, T_t, S_snow, S_liq, S_cum,
                                     p["max_melt"], p["x26"], p["x18"], p["x16"],
                                     p["x19"], eps)
    o3, Ss3, Sl3, Sc3 = snobal_hmets(snowfall, rainfall, T_t, S_snow, S_liq, S_cum,
                                      p["x24"], p["max_melt"], p["x26"], p["x18"],
                                      p["x16"], p["x17"], p["x13"], p["swi_max"],
                                      p["x15"], p["x27"], eps)
    snow_out = (torch.stack([o1, o2, o3], -1) * w_sn).sum(-1)
    S_snow = (torch.stack([Ss1, Ss2, Ss3], -1) * w_sn).sum(-1)
    S_liq  = (torch.stack([Sl1, Sl2, Sl3], -1) * w_sn).sum(-1)
    S_cum  = (torch.stack([Sc1, Sc2, Sc3], -1) * w_sn).sum(-1)

    # 3. 入渗
    i1 = inf_hmets(snow_out, S_top, p["Smax_top"], p["x1"], eps)
    i2 = inf_vic_arno(snow_out, S_top, p["Smax_top"], p["x2"], eps)
    i3 = inf_hbv(snow_out, S_top, p["Smax_top"], p["x3"], eps)
    infiltration = (torch.stack([i1, i2, i3], -1) * w_inf).sum(-1)
    surface_runoff = snow_out - infiltration

    # 4. 蒸发
    e1 = soilevap_all(PET_t, p["x8"], S_top)
    e2 = soilevap_linear(PET_t, p["x8"], S_top, p["fc"] * p["Smax_top"], eps)
    e3 = soilevap_vic(PET_t, p["x8"], S_top, p["Smax_top"], p["x36"], eps)
    evaporation = (torch.stack([e1, e2, e3], -1) * w_ev).sum(-1)

    # 5. 渗漏 + 更新表层
    perc_top = perc_linear(S_top, p["x28"])
    S_top = F.relu(S_top + infiltration - evaporation - perc_top)
    overflow = F.relu(S_top - p["Smax_top"])
    S_top = S_top - overflow
    surface_runoff = surface_runoff + overflow

    # 6. 快速流
    q1 = quick_linear(S_top, p["k_quick"])
    q2 = quick_vic(S_top, p["Smax_top"], p["x5"], p["x6"], eps)
    q3 = quick_topmodel(S_top, p["Smax_top"], p["x5"], p["x6"], p["x7"], eps)
    quickflow = (torch.stack([q1, q2, q3], -1) * w_qf).sum(-1)
    S_top = F.relu(S_top - quickflow)

    # 7. 潜水层 + 基流
    perc_ph = perc_linear(S_phreatic, p["x35"])
    b1 = base_linear(S_phreatic, p["k_base"])
    b2 = base_power(S_phreatic, p["k_base"], p["x12"], eps)
    baseflow = (torch.stack([b1, b2], -1) * w_bf).sum(-1)
    S_phreatic = F.relu(S_phreatic + perc_top - perc_ph - baseflow)
    overflow_ph = F.relu(S_phreatic - p["Smax_ph"])
    S_phreatic = S_phreatic - overflow_ph
    baseflow = baseflow + overflow_ph

    Q_total = surface_runoff + quickflow + baseflow
    return Q_total, S_snow, S_liq, S_cum, S_top, S_phreatic


# ===================================================================
# 主模型
# ===================================================================

class DiffBlendV2(nn.Module):
    """可微分公式混合水文模型 V2 (精简版)

    与 V1 物理等价，去除 flux 记录和 balance loss 基础设施，
    专注于训练/推理性能。
    """

    PARAM_BOUNDS = {
        "x1":  [0.0, 1.0],    # hmets_runoff_coeff
        "x2":  [0.3, 3.0],    # b_exp (VIC_ARNO)
        "x3":  [0.5, 3.0],    # hbv_beta
        "x4":  [-5.0, -1.0],  # log_k_quick
        "x5":  [0.0, 100.0],  # q_max
        "x6":  [0.5, 2.0],    # n_quick
        "x7":  [5.0, 10.0],   # topmodel_lambda
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

        self.param_names = list(self.PARAM_BOUNDS.keys())
        self.n_phy = len(self.param_names)
        self.learnable_param_count = (
            self.n_phy * self.nmul
            + len(self.ROUTING_BOUNDS)
            + TOTAL_WEIGHT_LOGITS
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config:
            self._load_config(config)
        self._step_fn = torch.compile(diff_blend_step)

    def _load_config(self, config: Dict[str, Any]) -> None:
        for k in ["warm_up", "warm_up_states", "variables", "nearzero", "nmul",
                  "weight_method", "tau"]:
            if k in config:
                setattr(self, k, config[k])
        self.learnable_param_count = (
            self.n_phy * self.nmul + len(self.ROUTING_BOUNDS) + TOTAL_WEIGHT_LOGITS
        )

    # ---------------------------------------------------------------
    # 参数解包
    # ---------------------------------------------------------------

    def unpack_parameters(self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]):
        _, raw = parameters
        B = raw.shape[0]
        n_rout = len(self.ROUTING_BOUNDS)

        raw_phy  = raw[:, :self.n_phy * self.nmul]
        raw_rout = raw[:, self.n_phy * self.nmul: self.n_phy * self.nmul + n_rout]
        raw_w    = raw[:, self.n_phy * self.nmul + n_rout:]

        # 物理参数
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

        # 过程权重
        w, idx = {}, 0
        for proc, n_opt in PROCESS_OPTIONS.items():
            logits = raw_w[:, idx: idx + n_opt]
            w[proc] = activate_weights(logits, self.weight_method, self.tau, self.training)
            idx += n_opt

        return p, w, rout

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

    # ---------------------------------------------------------------
    # 路由
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def forward(self, x_dict: Dict[str, torch.Tensor],
                parameters: Tuple[Optional[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
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

        for t in range(n_steps):
            Q_t, S_snow, S_liq, S_cum, S_top, S_phreatic = self._step_fn(
                P[t], T[t], PET[t],
                S_snow, S_liq, S_cum, S_top, S_phreatic,
                p, w_rs, w_sn, w_inf, w_ev, w_qf, w_bf,
                self.nearzero,
            )
            Q_sim[t] = Q_t

        Q_mean = Q_sim.mean(-1)
        Qrouted = self._apply_routing(Q_mean, rout["rout_a"], rout["rout_b"])

        result: Dict[str, torch.Tensor] = {"streamflow": Qrouted}

        # 过程权重 (扩展到时间维，供分析用)
        for proc, w in weights.items():
            for i in range(w.shape[-1]):
                result[f"w_{proc}_{i}"] = w[:, i].unsqueeze(0).expand(n_steps, -1)

        if not self.warm_up_states:
            cutoff = self.warm_up if self.pred_cutoff == 0 else self.pred_cutoff
            result = {k: v[cutoff:] for k, v in result.items()}

        return result
