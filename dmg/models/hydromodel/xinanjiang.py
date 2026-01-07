import torch
from typing import Optional, Tuple

# 注意：这里假设你已经创建了对应的核心组件
from .flux.saturation import saturation_14, saturation_2
from .flux.interflow import interflow_5
from .flux.baseflow import baseflow_1
from .flux.split import split_1, split_2
from .flux.evap import evap_21

# 参数取值范围字典 (基于 flux m_28_xinanjiang_12p_4s)
XINANJIANG_PARAMS_BOUNDS = {
    "aim": [0.0, 1.0],
    "par_a": [-0.49, 0.49],
    "par_b": [0.0, 10.0],
    "stot": [1.0, 2000.0],
    "fwm": [0.01, 0.99],
    "flm": [0.01, 0.99],
    "par_c": [0.01, 0.99],
    "ex": [0.0, 10.0],
    "ki": [0.0, 1.0],
    "kg": [0.0, 1.0],
    "ci": [0.0, 1.0],
    "cg": [0.0, 1.0],
}

# 参数描述字典
XINANJIANG_PARAMS_DESC = {
    "aim": "Fraction impervious area [-]",
    "par_a": "Tension water distribution inflection parameter [-]",
    "par_b": "Tension water distribution shape parameter [-]",
    "stot": "Total soil moisture storage (W+S) [mm]",
    "fwm": "Fraction of Stot that is Wmax [-]",
    "flm": "Fraction of wmax that is LM [-]",
    "par_c": "Fraction of LM for second evaporation change [-]",
    "ex": "Free water distribution shape parameter [-]",
    "ki": "Free water interflow parameter [d-1]",
    "kg": "Free water groundwater parameter [d-1]",
    "ci": "Interflow time coefficient [d-1]",
    "cg": "Baseflow time coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for Xinanjiang model.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def xinanjiang_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 XINANJIANG_PARAMS_BOUNDS 的键顺序完全一致
    aim: torch.Tensor,
    par_a: torch.Tensor,
    par_b: torch.Tensor,
    stot: torch.Tensor,
    fwm: torch.Tensor,
    flm: torch.Tensor,
    par_c: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    新安江模型单步计算函数.
    """
    # 辅助参数 (MATLAB 中 init 函数的部分内容)
    wmax = fwm * stot
    smax = (1.0 - fwm) * stot
    lm = flm * wmax

    # 1. 降水分离 (不透水面/入渗)
    flux_rb = split_1(aim, P, nearzero)
    flux_pi = split_2(aim, P, nearzero)

    # 2. 张性水储蓄 (Tension Water S1)
    # 产流计算 (Jayawardena 修正版本)
    flux_r = saturation_14(par_a, par_b, S1, wmax, flux_pi, nearzero)

    # 蒸发计算 (三段式)
    flux_e = evap_21(lm, par_c, S1, PET, nearzero)
    flux_e = torch.minimum(flux_e, S1)

    # 更新 S1
    S1 = S1 + flux_pi - flux_e - flux_r
    S1 = torch.clamp(S1, min=nearzero)

    # 3. 自由水储蓄 (Free Water S2)
    S2 = S2 + flux_r

    # 自由水面产流与补给
    flux_rs = saturation_2(S2, smax, ex, flux_r, nearzero)
    flux_ri_pot = saturation_2(S2, smax, ex, S2 * ki, nearzero)
    flux_rg_pot = saturation_2(S2, smax, ex, S2 * kg, nearzero)

    # 限制流出以保护质量守恒
    total_slow = flux_ri_pot + flux_rg_pot
    scale = torch.where(
        total_slow > S2, S2 / (total_slow + nearzero), torch.ones_like(S2)
    )
    flux_ri = flux_ri_pot * scale
    flux_rg = flux_rg_pot * scale

    S2 = S2 - flux_rs - flux_ri - flux_rg
    S2 = torch.clamp(S2, min=nearzero)

    # 4. 汇流路径储蓄 (S3: 壤中流, S4: 地下水)
    S3 = S3 + flux_ri
    S4 = S4 + flux_rg

    flux_qi = interflow_5(ci, S3, nearzero)
    flux_qg = baseflow_1(cg, S4, nearzero)

    S3 = S3 - flux_qi
    S4 = S4 - flux_qg

    # 5. 总产流聚合
    Qsim = flux_rb + flux_rs + flux_qi + flux_qg
    Ea = flux_e

    return Qsim, Ea, S1, S2, S3, S4


def xinanjiang_step_all(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # 参数顺序与 XINANJIANG_PARAMS_BOUNDS 的键顺序完全一致
    aim: torch.Tensor,
    par_a: torch.Tensor,
    par_b: torch.Tensor,
    stot: torch.Tensor,
    fwm: torch.Tensor,
    flm: torch.Tensor,
    par_c: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """
    新安江模型单步计算函数.
    """
    # 辅助参数 (MATLAB 中 init 函数的部分内容)
    wmax = fwm * stot
    smax = (1.0 - fwm) * stot
    lm = flm * wmax

    # 1. 降水分离 (不透水面/入渗)
    flux_rb = split_1(aim, P, nearzero)
    flux_pi = split_2(aim, P, nearzero)

    # 2. 张性水储蓄 (Tension Water S1)
    # 产流计算 (Jayawardena 修正版本)
    flux_r = saturation_14(par_a, par_b, S1, wmax, flux_pi, nearzero)

    # 蒸发计算 (三段式)
    flux_e = evap_21(lm, par_c, S1, PET, nearzero)
    flux_e = torch.minimum(flux_e, S1)

    # 更新 S1
    S1 = S1 + flux_pi - flux_e - flux_r
    S1 = torch.clamp(S1, min=nearzero)

    # 3. 自由水储蓄 (Free Water S2)
    S2 = S2 + flux_r

    # 自由水面产流与补给
    flux_rs = saturation_2(S2, smax, ex, flux_r, nearzero)
    flux_ri_pot = saturation_2(S2, smax, ex, S2 * ki, nearzero)
    flux_rg_pot = saturation_2(S2, smax, ex, S2 * kg, nearzero)

    # 限制流出以保护质量守恒
    total_slow = flux_ri_pot + flux_rg_pot
    scale = torch.where(
        total_slow > S2, S2 / (total_slow + nearzero), torch.ones_like(S2)
    )
    flux_ri = flux_ri_pot * scale
    flux_rg = flux_rg_pot * scale

    S2 = S2 - flux_rs - flux_ri - flux_rg
    S2 = torch.clamp(S2, min=nearzero)

    # 4. 汇流路径储蓄 (S3: 壤中流, S4: 地下水)
    S3 = S3 + flux_ri
    S4 = S4 + flux_rg

    flux_qi = interflow_5(ci, S3, nearzero)
    flux_qg = baseflow_1(cg, S4, nearzero)

    S3 = S3 - flux_qi
    S4 = S4 - flux_qg

    # 5. 总产流聚合
    Qsim = flux_rb + flux_rs + flux_qi + flux_qg
    Ea = flux_e

    return (
        Qsim,
        Ea,
        flux_r,
        flux_e,
        flux_rs,
        flux_ri,
        flux_rg,
        flux_qi,
        flux_qg,
        S1,
        S2,
        S3,
        S4,
    )
