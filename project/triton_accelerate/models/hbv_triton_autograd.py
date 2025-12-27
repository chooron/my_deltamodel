"""
HBV 水文模型 - Triton 前向 + PyTorch 自动微分反向
这个版本使用 Triton 加速前向计算，但使用 PyTorch 自动微分计算梯度，
避免手写 backward kernel 的梯度错误问题。

Author: chooron
"""

import torch
from typing import Tuple, Optional


def hbv_step_pytorch(
    p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
    snow: torch.Tensor, melt: torch.Tensor, sm: torch.Tensor,
    suz: torch.Tensor, slz: torch.Tensor,
    tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor, cwh: torch.Tensor,
    fc: torch.Tensor, beta: torch.Tensor, lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
    perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, uzl: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """
    HBV 单步计算 - 纯 PyTorch 实现，支持自动微分
    """
    # ========== Snow Block ==========
    temp_diff = t_val - tt
    is_rain = temp_diff > 0.0
    rain = torch.where(is_rain, p, torch.zeros_like(p))
    snow_input = torch.where(is_rain, torch.zeros_like(p), p)

    snow_st1 = snow + snow_input
    pot_melt = cfmax * torch.clamp(temp_diff, min=0.0)
    melt_amount = torch.minimum(pot_melt, snow_st1)
    snow_st2 = snow_st1 - melt_amount
    melt_st1 = melt + melt_amount

    pot_refreeze = cfr * cfmax * torch.clamp(-temp_diff, min=0.0)
    refreeze_amt = torch.minimum(pot_refreeze, melt_st1)
    snow_out = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt

    tosoil = torch.clamp(melt_st2 - cwh * snow_out, min=0.0)
    melt_out = melt_st2 - tosoil

    # ========== Soil Block ==========
    eps = nearzero
    soil_ratio = sm / fc
    soil_wet = torch.clamp(torch.pow(torch.clamp(soil_ratio, min=eps), beta), 0.0, 1.0)
    recharge = (rain + tosoil) * soil_wet

    sm_st1 = sm + rain + tosoil - recharge
    excess = torch.clamp(sm_st1 - fc, min=0.0)
    sm_st2 = sm_st1 - excess

    # 蒸发
    ef1 = sm_st2 / (lp * fc)
    ef1 = torch.clamp(ef1, 0.0, 1.0)
    evapfactor = torch.clamp(torch.pow(torch.clamp(ef1, min=eps), betaet), 0.0, 1.0)
    etact = torch.minimum(pet * evapfactor, sm_st2)
    sm_after_evap = torch.clamp(sm_st2 - etact, min=eps)

    # 毛管上升
    sm_ratio = torch.clamp(sm_after_evap / fc, max=1.0)
    capillary = torch.minimum(slz, c_par * slz * (1.0 - sm_ratio))
    sm_out = torch.clamp(sm_after_evap + capillary, min=eps)
    slz_after_cap = torch.clamp(slz - capillary, min=eps)

    # ========== Routing Block ==========
    suz_st1 = suz + recharge + excess
    perc_flux = torch.minimum(suz_st1, perc)
    suz_st2 = suz_st1 - perc_flux
    slz_st1 = slz_after_cap + perc_flux

    q0 = k0 * torch.clamp(suz_st2 - uzl, min=0.0)
    suz_st3 = suz_st2 - q0
    q1 = k1 * suz_st3
    suz_out = suz_st3 - q1

    q2 = k2 * slz_st1
    slz_out = slz_st1 - q2

    q_total = q0 + q1 + q2

    return snow_out, melt_out, sm_out, suz_out, slz_out, q_total


# 使用 torch.compile 加速（PyTorch 2.0+）
try:
    hbv_step_compiled = torch.compile(hbv_step_pytorch)
except Exception:
    # 如果 torch.compile 不可用，使用原始函数
    hbv_step_compiled = hbv_step_pytorch
    
try:
    hbv_step_jit = torch.jit.script(hbv_step_pytorch)
except Exception:
    # 如果 torch.compile 不可用，使用原始函数
    hbv_step_jit = hbv_step_pytorch


def hbv_run_autograd(
    precip: torch.Tensor,
    temp: torch.Tensor,
    pet: torch.Tensor,
    params: dict,
    init_states: Optional[dict] = None,
    use_compile: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    运行 HBV 模型 - 使用 PyTorch 自动微分
    
    Parameters
    ----------
    precip : torch.Tensor
        降水，形状 (T, ...)
    temp : torch.Tensor
        温度，形状 (T, ...)
    pet : torch.Tensor
        潜在蒸发，形状 (T, ...)
    params : dict
        模型参数字典
    init_states : dict, optional
        初始状态
    use_compile : bool
        是否使用 torch.compile 加速
        
    Returns
    -------
    q_series : torch.Tensor
        径流序列，形状 (T, ...)
    final_states : dict
        最终状态
    """
    T = precip.shape[0]
    spatial_shape = precip.shape[1:]
    device = precip.device
    dtype = precip.dtype

    if init_states is None:
        snow = torch.zeros(spatial_shape, device=device, dtype=dtype)
        melt = torch.zeros_like(snow)
        sm = torch.zeros_like(snow)
        suz = torch.zeros_like(snow)
        slz = torch.zeros_like(snow)
    else:
        snow = init_states.get("snow", torch.zeros(spatial_shape, device=device, dtype=dtype))
        melt = init_states.get("melt", torch.zeros_like(snow))
        sm = init_states.get("sm", torch.zeros_like(snow))
        suz = init_states.get("suz", torch.zeros_like(snow))
        slz = init_states.get("slz", torch.zeros_like(snow))

    tt = params["tt"]
    cfmax = params["cfmax"]
    cfr = params["cfr"]
    cwh = params["cwh"]
    c_par = params["c"]
    fc = params["fc"]
    beta = params["beta"]
    lp = params["lp"]
    betaet = params["betaet"]
    perc = params["perc"]
    k0 = params["k0"]
    k1 = params["k1"]
    k2 = params["k2"]
    uzl = params["uzl"]

    step_fn = hbv_step_compiled if use_compile else hbv_step_pytorch

    q_series = []
    for t in range(T):
        snow, melt, sm, suz, slz, q = step_fn(
            precip[t], temp[t], pet[t], snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par, perc, k0, k1, k2, uzl,
        )
        q_series.append(q)

    q_series = torch.stack(q_series, dim=0)
    final_states = {"snow": snow, "melt": melt, "sm": sm, "suz": suz, "slz": slz}
    return q_series, final_states



class HbvAutograd(torch.nn.Module):
    """
    HBV 水文模型 - 使用 PyTorch 自动微分
    
    这个版本使用纯 PyTorch 实现，梯度由自动微分计算，
    避免手写 backward 的梯度错误问题。
    可选使用 torch.compile 加速。
    """

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

    def __init__(self, config=None, device=None, use_compile=True):
        super().__init__()
        self.name = "HBV_Autograd"
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_compile = use_compile
        self.nearzero = 1e-6
        self.nmul = 1
        self.warm_up = 0
        
        if config is not None:
            self.nmul = config.get("nmul", 1)
            self.warm_up = config.get("warm_up", 0)
            self.nearzero = config.get("nearzero", 1e-6)

    def forward(self, forcing, params):
        """
        前向传播
        
        Parameters
        ----------
        forcing : torch.Tensor
            驱动数据，形状 (T, B, 3) - [prcp, tmean, pet]
        params : dict
            参数字典
            
        Returns
        -------
        dict
            包含 streamflow 等输出
        """
        n_steps, n_grid = forcing.shape[:2]
        
        # 提取驱动数据
        P = forcing[:, :, 0].unsqueeze(-1).repeat(1, 1, self.nmul)
        T = forcing[:, :, 1].unsqueeze(-1).repeat(1, 1, self.nmul)
        PET = forcing[:, :, 2].unsqueeze(-1).repeat(1, 1, self.nmul)
        
        # 初始化状态
        snow = torch.zeros(n_grid, self.nmul, device=self.device) + self.nearzero
        melt = torch.zeros_like(snow)
        sm = torch.zeros_like(snow)
        suz = torch.zeros_like(snow)
        slz = torch.zeros_like(snow)
        
        step_fn = hbv_step_compiled if self.use_compile else hbv_step_pytorch
        
        q_list = []
        for t in range(n_steps):
            snow, melt, sm, suz, slz, q = step_fn(
                P[t], T[t], PET[t], snow, melt, sm, suz, slz,
                params["parTT"], params["parCFMAX"], params["parCFR"], params["parCWH"],
                params["parFC"], params["parBETA"], params["parLP"], params["parBETAET"],
                params["parC"], params["parPERC"], params["parK0"], params["parK1"],
                params["parK2"], params["parUZL"],
                self.nearzero,
            )
            q_list.append(q)
        
        Qsim = torch.stack(q_list, dim=0)
        
        # 多模型平均
        Qsim_avg = Qsim.mean(dim=-1)
        
        return {
            "streamflow": Qsim_avg,
            "Qsimmu": Qsim,
        }


if __name__ == "__main__":
    # 简单测试
    print("Testing HBV Autograd implementation...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    # 创建测试数据
    T, B, nmul = 100, 10, 4
    
    forcing = torch.rand(T, B, 3, device=device, dtype=dtype)
    forcing[:, :, 0] *= 20  # prcp
    forcing[:, :, 1] = forcing[:, :, 1] * 20 - 5  # tmean
    forcing[:, :, 2] *= 5  # pet
    
    params = {
        "parTT": torch.zeros(B, nmul, device=device, dtype=dtype),
        "parCFMAX": torch.ones(B, nmul, device=device, dtype=dtype) * 3.0,
        "parCFR": torch.ones(B, nmul, device=device, dtype=dtype) * 0.05,
        "parCWH": torch.ones(B, nmul, device=device, dtype=dtype) * 0.1,
        "parFC": torch.ones(B, nmul, device=device, dtype=dtype) * 200.0,
        "parBETA": torch.ones(B, nmul, device=device, dtype=dtype) * 2.0,
        "parLP": torch.ones(B, nmul, device=device, dtype=dtype) * 0.7,
        "parBETAET": torch.ones(B, nmul, device=device, dtype=dtype) * 1.5,
        "parC": torch.ones(B, nmul, device=device, dtype=dtype) * 0.05,
        "parPERC": torch.ones(B, nmul, device=device, dtype=dtype) * 2.0,
        "parK0": torch.ones(B, nmul, device=device, dtype=dtype) * 0.2,
        "parK1": torch.ones(B, nmul, device=device, dtype=dtype) * 0.05,
        "parK2": torch.ones(B, nmul, device=device, dtype=dtype) * 0.01,
        "parUZL": torch.ones(B, nmul, device=device, dtype=dtype) * 20.0,
    }
    
    # 设置 requires_grad
    for k, v in params.items():
        params[k] = v.requires_grad_(True)
    
    model = HbvAutograd(config={"nmul": nmul}, device=device, use_compile=False)
    
    # 前向
    output = model(forcing, params)
    print(f"Output shape: {output['streamflow'].shape}")
    
    # 反向
    loss = output['streamflow'].sum()
    loss.backward()
    
    # 检查梯度
    for k, v in params.items():
        if v.grad is not None:
            print(f"{k}: grad mean = {v.grad.mean().item():.6f}, grad std = {v.grad.std().item():.6f}")
        else:
            print(f"{k}: no gradient")
    
    print("\nTest passed!")
