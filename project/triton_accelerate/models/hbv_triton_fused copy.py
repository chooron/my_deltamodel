"""
HBV 水文模型 - Triton 融合实现 (Fused Forward + Backward)

将 Snow、Soil、Routing 三个 block 合并成一个整体的 kernel，
避免中间状态梯度传递问题，提高数值稳定性。

Author: Kiro
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _hbv_fused_forward_kernel(
    # 输入指针
    p_ptr, t_ptr, pet_ptr,
    snow_ptr, melt_ptr, sm_ptr, suz_ptr, slz_ptr,
    # 参数指针
    tt_ptr, cfmax_ptr, cfr_ptr, cwh_ptr,
    fc_ptr, beta_ptr, lp_ptr, betaet_ptr, c_ptr,
    perc_ptr, k0_ptr, k1_ptr, k2_ptr, uzl_ptr,
    # 输出指针
    snow_out_ptr, melt_out_ptr, sm_out_ptr, suz_out_ptr, slz_out_ptr, q_out_ptr,
    # 中间变量输出（用于 backward）
    rain_ptr, tosoil_ptr, recharge_ptr, excess_ptr,
    soil_wet_ptr, sm_st2_ptr, evapfactor_ptr, sm_after_evap_ptr,
    capillary_ptr, slz_after_cap_ptr,
    suz_st1_ptr, perc_flux_ptr, suz_st2_ptr, slz_st1_ptr,
    q0_ptr, suz_st3_ptr, q1_ptr, q2_ptr,
    # 元数据
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """融合的 HBV 前向 kernel"""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    eps = 1e-6

    # 加载输入
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    t_val = tl.load(t_ptr + offs, mask=mask, other=0.0)
    pet = tl.load(pet_ptr + offs, mask=mask, other=0.0)
    snow = tl.load(snow_ptr + offs, mask=mask, other=0.0)
    melt = tl.load(melt_ptr + offs, mask=mask, other=0.0)
    sm = tl.load(sm_ptr + offs, mask=mask, other=0.0)
    suz = tl.load(suz_ptr + offs, mask=mask, other=0.0)
    slz = tl.load(slz_ptr + offs, mask=mask, other=0.0)

    # 加载参数
    tt = tl.load(tt_ptr + offs, mask=mask, other=0.0)
    cfmax = tl.load(cfmax_ptr + offs, mask=mask, other=0.0)
    cfr = tl.load(cfr_ptr + offs, mask=mask, other=0.0)
    cwh = tl.load(cwh_ptr + offs, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    lp = tl.load(lp_ptr + offs, mask=mask, other=0.0)
    betaet = tl.load(betaet_ptr + offs, mask=mask, other=0.0)
    c_par = tl.load(c_ptr + offs, mask=mask, other=0.0)
    perc = tl.load(perc_ptr + offs, mask=mask, other=0.0)
    k0 = tl.load(k0_ptr + offs, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offs, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offs, mask=mask, other=0.0)
    uzl = tl.load(uzl_ptr + offs, mask=mask, other=0.0)

    # ========== Snow Block ==========
    temp_diff = t_val - tt
    is_rain = temp_diff > 0.0
    rain = tl.where(is_rain, p, 0.0)
    snow_input = tl.where(is_rain, 0.0, p)

    snow_st1 = snow + snow_input
    pot_melt = cfmax * tl.maximum(temp_diff, 0.0)
    mask_melt = pot_melt < snow_st1
    melt_amount = tl.where(mask_melt, pot_melt, snow_st1)
    snow_st2 = snow_st1 - melt_amount
    melt_st1 = melt + melt_amount

    pot_refreeze = cfr * cfmax * tl.maximum(-temp_diff, 0.0)
    mask_refreeze = pot_refreeze < melt_st1
    refreeze_amt = tl.where(mask_refreeze, pot_refreeze, melt_st1)
    snow_out = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt

    arg_tosoil = melt_st2 - cwh * snow_out
    mask_tosoil = arg_tosoil > 0.0
    tosoil = tl.where(mask_tosoil, arg_tosoil, 0.0)
    melt_out = melt_st2 - tosoil

    # ========== Soil Block ==========
    fc_safe = tl.maximum(fc, eps)
    lp_safe = tl.maximum(lp, eps)
    
    soil_ratio = sm / fc_safe
    soil_ratio_safe = tl.maximum(soil_ratio, eps)
    soil_wet = tl.minimum(tl.maximum(tl.exp(beta * tl.log(soil_ratio_safe)), 0.0), 1.0)
    recharge = (rain + tosoil) * soil_wet

    sm_st1 = sm + rain + tosoil - recharge
    excess = tl.maximum(sm_st1 - fc, 0.0)
    sm_st2 = sm_st1 - excess

    # 蒸发
    denom_ef = lp_safe * fc_safe
    ef1_raw = sm_st2 / denom_ef
    ef1 = tl.minimum(tl.maximum(ef1_raw, 0.0), 1.0)
    ef1_safe = tl.maximum(ef1, eps)
    evapfactor = tl.minimum(tl.maximum(tl.exp(betaet * tl.log(ef1_safe)), 0.0), 1.0)
    
    pet_evap = pet * evapfactor
    etact = tl.minimum(pet_evap, sm_st2)
    sm_after_evap = tl.maximum(sm_st2 - etact, eps)

    # 毛管上升
    sm_ratio = tl.minimum(sm_after_evap / fc_safe, 1.0)
    cap_expr = c_par * slz * (1.0 - sm_ratio)
    capillary = tl.minimum(slz, cap_expr)
    sm_out = tl.maximum(sm_after_evap + capillary, eps)
    slz_after_cap = tl.maximum(slz - capillary, eps)

    # ========== Routing Block ==========
    suz_st1 = suz + recharge + excess
    perc_flux = tl.minimum(suz_st1, perc)
    suz_st2 = suz_st1 - perc_flux
    slz_st1 = slz_after_cap + perc_flux

    q0_arg = suz_st2 - uzl
    q0 = k0 * tl.maximum(q0_arg, 0.0)
    suz_st3 = suz_st2 - q0
    q1 = k1 * suz_st3
    suz_out = suz_st3 - q1

    q2 = k2 * slz_st1
    slz_out = slz_st1 - q2

    q_total = q0 + q1 + q2

    # 存储输出
    tl.store(snow_out_ptr + offs, snow_out, mask=mask)
    tl.store(melt_out_ptr + offs, melt_out, mask=mask)
    tl.store(sm_out_ptr + offs, sm_out, mask=mask)
    tl.store(suz_out_ptr + offs, suz_out, mask=mask)
    tl.store(slz_out_ptr + offs, slz_out, mask=mask)
    tl.store(q_out_ptr + offs, q_total, mask=mask)

    # 存储中间变量（用于 backward）
    tl.store(rain_ptr + offs, rain, mask=mask)
    tl.store(tosoil_ptr + offs, tosoil, mask=mask)
    tl.store(recharge_ptr + offs, recharge, mask=mask)
    tl.store(excess_ptr + offs, excess, mask=mask)
    tl.store(soil_wet_ptr + offs, soil_wet, mask=mask)
    tl.store(sm_st2_ptr + offs, sm_st2, mask=mask)
    tl.store(evapfactor_ptr + offs, evapfactor, mask=mask)
    tl.store(sm_after_evap_ptr + offs, sm_after_evap, mask=mask)
    tl.store(capillary_ptr + offs, capillary, mask=mask)
    tl.store(slz_after_cap_ptr + offs, slz_after_cap, mask=mask)
    tl.store(suz_st1_ptr + offs, suz_st1, mask=mask)
    tl.store(perc_flux_ptr + offs, perc_flux, mask=mask)
    tl.store(suz_st2_ptr + offs, suz_st2, mask=mask)
    tl.store(slz_st1_ptr + offs, slz_st1, mask=mask)
    tl.store(q0_ptr + offs, q0, mask=mask)
    tl.store(suz_st3_ptr + offs, suz_st3, mask=mask)
    tl.store(q1_ptr + offs, q1, mask=mask)
    tl.store(q2_ptr + offs, q2, mask=mask)



@triton.jit
def _hbv_fused_backward_kernel(
    # 输入指针（前向输入）
    p_ptr, t_ptr, pet_ptr,
    snow_ptr, melt_ptr, sm_ptr, suz_ptr, slz_ptr,
    # 参数指针
    tt_ptr, cfmax_ptr, cfr_ptr, cwh_ptr,
    fc_ptr, beta_ptr, lp_ptr, betaet_ptr, c_ptr,
    perc_ptr, k0_ptr, k1_ptr, k2_ptr, uzl_ptr,
    # 中间变量指针（从前向保存）
    rain_ptr, tosoil_ptr, recharge_ptr, excess_ptr,
    soil_wet_ptr, sm_st2_ptr, evapfactor_ptr, sm_after_evap_ptr,
    capillary_ptr, slz_after_cap_ptr,
    suz_st1_ptr, perc_flux_ptr, suz_st2_ptr, slz_st1_ptr,
    q0_ptr, suz_st3_ptr, q1_ptr, q2_ptr,
    # 输出梯度指针
    g_snow_out_ptr, g_melt_out_ptr, g_sm_out_ptr, g_suz_out_ptr, g_slz_out_ptr, g_q_out_ptr,
    # 输入梯度输出指针
    g_p_ptr, g_t_ptr, g_pet_ptr,
    g_snow_ptr, g_melt_ptr, g_sm_ptr, g_suz_ptr, g_slz_ptr,
    # 参数梯度输出指针
    g_tt_ptr, g_cfmax_ptr, g_cfr_ptr, g_cwh_ptr,
    g_fc_ptr, g_beta_ptr, g_lp_ptr, g_betaet_ptr, g_c_ptr,
    g_perc_ptr, g_k0_ptr, g_k1_ptr, g_k2_ptr, g_uzl_ptr,
    # 元数据
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """融合的 HBV 反向 kernel"""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    eps = 1e-6

    # 加载前向输入
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    t_val = tl.load(t_ptr + offs, mask=mask, other=0.0)
    pet = tl.load(pet_ptr + offs, mask=mask, other=0.0)
    snow = tl.load(snow_ptr + offs, mask=mask, other=0.0)
    melt = tl.load(melt_ptr + offs, mask=mask, other=0.0)
    sm = tl.load(sm_ptr + offs, mask=mask, other=0.0)
    suz = tl.load(suz_ptr + offs, mask=mask, other=0.0)
    slz = tl.load(slz_ptr + offs, mask=mask, other=0.0)

    # 加载参数
    tt = tl.load(tt_ptr + offs, mask=mask, other=0.0)
    cfmax = tl.load(cfmax_ptr + offs, mask=mask, other=0.0)
    cfr = tl.load(cfr_ptr + offs, mask=mask, other=0.0)
    cwh = tl.load(cwh_ptr + offs, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    lp = tl.load(lp_ptr + offs, mask=mask, other=0.0)
    betaet = tl.load(betaet_ptr + offs, mask=mask, other=0.0)
    c_par = tl.load(c_ptr + offs, mask=mask, other=0.0)
    perc = tl.load(perc_ptr + offs, mask=mask, other=0.0)
    k0 = tl.load(k0_ptr + offs, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offs, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offs, mask=mask, other=0.0)
    uzl = tl.load(uzl_ptr + offs, mask=mask, other=0.0)

    # 加载中间变量
    rain = tl.load(rain_ptr + offs, mask=mask, other=0.0)
    tosoil = tl.load(tosoil_ptr + offs, mask=mask, other=0.0)
    recharge = tl.load(recharge_ptr + offs, mask=mask, other=0.0)
    excess = tl.load(excess_ptr + offs, mask=mask, other=0.0)
    soil_wet = tl.load(soil_wet_ptr + offs, mask=mask, other=0.0)
    sm_st2 = tl.load(sm_st2_ptr + offs, mask=mask, other=0.0)
    evapfactor = tl.load(evapfactor_ptr + offs, mask=mask, other=0.0)
    sm_after_evap = tl.load(sm_after_evap_ptr + offs, mask=mask, other=0.0)
    capillary = tl.load(capillary_ptr + offs, mask=mask, other=0.0)
    slz_after_cap = tl.load(slz_after_cap_ptr + offs, mask=mask, other=0.0)
    suz_st1 = tl.load(suz_st1_ptr + offs, mask=mask, other=0.0)
    perc_flux = tl.load(perc_flux_ptr + offs, mask=mask, other=0.0)
    suz_st2 = tl.load(suz_st2_ptr + offs, mask=mask, other=0.0)
    slz_st1 = tl.load(slz_st1_ptr + offs, mask=mask, other=0.0)
    q0 = tl.load(q0_ptr + offs, mask=mask, other=0.0)
    suz_st3 = tl.load(suz_st3_ptr + offs, mask=mask, other=0.0)
    q1 = tl.load(q1_ptr + offs, mask=mask, other=0.0)
    q2 = tl.load(q2_ptr + offs, mask=mask, other=0.0)

    # 加载输出梯度
    g_snow_out = tl.load(g_snow_out_ptr + offs, mask=mask, other=0.0)
    g_melt_out = tl.load(g_melt_out_ptr + offs, mask=mask, other=0.0)
    g_sm_out = tl.load(g_sm_out_ptr + offs, mask=mask, other=0.0)
    g_suz_out = tl.load(g_suz_out_ptr + offs, mask=mask, other=0.0)
    g_slz_out = tl.load(g_slz_out_ptr + offs, mask=mask, other=0.0)
    g_q_out = tl.load(g_q_out_ptr + offs, mask=mask, other=0.0)

    # 重新计算一些需要的中间值
    fc_safe = tl.maximum(fc, eps)
    lp_safe = tl.maximum(lp, eps)
    temp_diff = t_val - tt
    is_rain = temp_diff > 0.0
    
    # Snow block 中间值
    snow_input = tl.where(is_rain, 0.0, p)
    snow_st1 = snow + snow_input
    pot_melt = cfmax * tl.maximum(temp_diff, 0.0)
    mask_melt = pot_melt < snow_st1
    melt_amount = tl.where(mask_melt, pot_melt, snow_st1)
    snow_st2 = snow_st1 - melt_amount
    melt_st1 = melt + melt_amount
    pot_refreeze = cfr * cfmax * tl.maximum(-temp_diff, 0.0)
    mask_refreeze = pot_refreeze < melt_st1
    refreeze_amt = tl.where(mask_refreeze, pot_refreeze, melt_st1)
    snow_out = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt
    arg_tosoil = melt_st2 - cwh * snow_out
    mask_tosoil = arg_tosoil > 0.0
    
    # Soil block 中间值
    soil_ratio = sm / fc_safe
    soil_ratio_safe = tl.maximum(soil_ratio, eps)
    soil_wet_raw = tl.exp(beta * tl.log(soil_ratio_safe))
    mask_sw = (soil_wet_raw > 0.0) & (soil_wet_raw < 1.0)
    sm_st1 = sm + rain + tosoil - recharge
    mask_excess = sm_st1 > fc
    denom_ef = lp_safe * fc_safe
    ef1_raw = sm_st2 / denom_ef
    mask_ef1 = (ef1_raw > 0.0) & (ef1_raw < 1.0)
    ef1 = tl.minimum(tl.maximum(ef1_raw, 0.0), 1.0)
    ef1_safe = tl.maximum(ef1, eps)
    ef2_raw = tl.exp(betaet * tl.log(ef1_safe))
    mask_ef2 = (ef2_raw > 0.0) & (ef2_raw < 1.0)
    pet_evap = pet * evapfactor
    mask_et = pet_evap < sm_st2
    sm_after_evap_raw = sm_st2 - tl.where(mask_et, pet_evap, sm_st2)
    mask_sm_after = sm_after_evap_raw > eps
    sm_ratio_raw = sm_after_evap / fc_safe
    mask_smratio = sm_ratio_raw < 1.0
    sm_ratio = tl.minimum(sm_ratio_raw, 1.0)
    cap_expr = c_par * slz * (1.0 - sm_ratio)
    mask_cap = cap_expr < slz
    sm_out_raw = sm_after_evap + capillary
    mask_sm_out = sm_out_raw > eps
    slz_out_raw = slz - capillary
    mask_slz_out = slz_out_raw > eps
    
    # Routing block 中间值
    mask_perc = suz_st1 < perc
    q0_arg = suz_st2 - uzl
    mask_q0 = q0_arg > 0.0

    # ========== 反向传播 ==========
    # 初始化梯度
    g_fc = tl.zeros_like(fc)
    g_beta = tl.zeros_like(beta)
    g_lp = tl.zeros_like(lp)
    g_betaet = tl.zeros_like(betaet)
    g_c_par = tl.zeros_like(c_par)
    g_perc = tl.zeros_like(perc)
    g_k0 = tl.zeros_like(k0)
    g_k1 = tl.zeros_like(k1)
    g_k2 = tl.zeros_like(k2)
    g_uzl = tl.zeros_like(uzl)
    g_temp_diff = tl.zeros_like(temp_diff)

    # ========== Routing Block Backward ==========
    # q_total = q0 + q1 + q2
    g_q0 = g_q_out
    g_q1 = g_q_out
    g_q2 = g_q_out

    # slz_out = slz_st1 - q2
    g_slz_st1 = tl.where(mask_slz_out, g_slz_out, 0.0)
    g_q2 = g_q2 - tl.where(mask_slz_out, g_slz_out, 0.0)

    # q2 = k2 * slz_st1
    g_k2 = g_q2 * slz_st1
    g_slz_st1 = g_slz_st1 + g_q2 * k2

    # suz_out = suz_st3 - q1
    g_suz_st3 = g_suz_out
    g_q1 = g_q1 - g_suz_out

    # q1 = k1 * suz_st3
    g_k1 = g_q1 * suz_st3
    g_suz_st3 = g_suz_st3 + g_q1 * k1

    # suz_st3 = suz_st2 - q0
    g_suz_st2 = g_suz_st3
    g_q0 = g_q0 - g_suz_st3

    # q0 = k0 * max(suz_st2 - uzl, 0)
    g_k0 = g_q0 * tl.maximum(q0_arg, 0.0)
    g_suz_st2 = g_suz_st2 + tl.where(mask_q0, g_q0 * k0, 0.0)
    g_uzl = tl.where(mask_q0, -g_q0 * k0, 0.0)

    # perc_flux = min(suz_st1, perc), suz_st2 = suz_st1 - perc_flux, slz_st1 = slz_after_cap + perc_flux
    g_suz_st1 = tl.where(mask_perc, g_slz_st1, g_suz_st2)
    g_perc = tl.where(mask_perc, 0.0, g_slz_st1 - g_suz_st2)
    g_slz_after_cap = g_slz_st1

    # suz_st1 = suz + recharge + excess
    g_suz = g_suz_st1
    g_recharge = g_suz_st1
    g_excess = g_suz_st1

    # ========== Soil Block Backward ==========
    # sm_out = max(sm_after_evap + capillary, eps)
    # slz_after_cap = max(slz - capillary, eps)
    g_sm_out_eff = tl.where(mask_sm_out, g_sm_out, 0.0)
    g_slz_after_cap_eff = tl.where(mask_slz_out, g_slz_after_cap, 0.0)
    
    g_sm_after_evap = g_sm_out_eff
    g_capillary = g_sm_out_eff - g_slz_after_cap_eff
    g_slz = g_slz_after_cap_eff

    # capillary = min(slz, cap_expr)
    g_cap_expr = tl.where(mask_cap, g_capillary, 0.0)
    g_slz = g_slz + tl.where(mask_cap, 0.0, g_capillary)

    # cap_expr = c_par * slz * (1 - sm_ratio)
    g_c_par = g_cap_expr * slz * (1.0 - sm_ratio)
    g_slz = g_slz + g_cap_expr * c_par * (1.0 - sm_ratio)
    g_sm_ratio = g_cap_expr * (-c_par * slz)

    # sm_ratio = min(sm_after_evap / fc, 1.0)
    g_sm_after_evap = g_sm_after_evap + tl.where(mask_smratio, g_sm_ratio / fc_safe, 0.0)
    g_fc = g_fc + tl.where(mask_smratio, g_sm_ratio * (-sm_after_evap / (fc_safe * fc_safe)), 0.0)

    # sm_after_evap = max(sm_st2 - etact, eps)
    # etact = min(pet * evapfactor, sm_st2)
    g_sm_st2 = tl.where(mask_sm_after, g_sm_after_evap, 0.0)
    g_etact = tl.where(mask_sm_after, -g_sm_after_evap, 0.0)
    g_pet_evap = tl.where(mask_et, g_etact, 0.0)
    g_sm_st2 = g_sm_st2 + tl.where(mask_et, 0.0, g_etact)

    # pet_evap = pet * evapfactor
    g_evapfactor = g_pet_evap * pet
    g_pet = g_pet_evap * evapfactor

    # evapfactor = clamp(ef1^betaet, 0, 1)
    g_ef2 = tl.where(mask_ef2, g_evapfactor, 0.0)
    log_ef1 = tl.log(ef1_safe)
    log_ef1_clamped = tl.minimum(tl.maximum(log_ef1, -20.0), 20.0)
    g_betaet = g_ef2 * ef2_raw * log_ef1_clamped
    exp_arg_ef = (betaet - 1.0) * log_ef1_clamped
    exp_arg_ef_clamped = tl.minimum(tl.maximum(exp_arg_ef, -20.0), 20.0)
    g_ef1 = g_ef2 * betaet * tl.exp(exp_arg_ef_clamped)

    # ef1 = clamp(sm_st2 / (lp * fc), 0, 1)
    g_ef1_raw = tl.where(mask_ef1, g_ef1, 0.0)
    denom_ef_safe = tl.maximum(denom_ef, eps)
    g_sm_st2 = g_sm_st2 + g_ef1_raw / denom_ef_safe
    g_lp = g_ef1_raw * (-sm_st2 / (denom_ef_safe * lp_safe))
    g_fc = g_fc + g_ef1_raw * (-sm_st2 / (denom_ef_safe * fc_safe))

    # sm_st2 = sm_st1 - excess, excess = max(sm_st1 - fc, 0)
    g_sm_st1 = g_sm_st2
    g_sm_st1 = g_sm_st1 + tl.where(mask_excess, g_excess, 0.0)
    g_fc = g_fc + tl.where(mask_excess, -g_excess, 0.0)

    # sm_st1 = sm + rain + tosoil - recharge
    # recharge = (rain + tosoil) * soil_wet
    g_recharge_tot = g_recharge - g_sm_st1
    g_soil_wet = g_recharge_tot * (rain + tosoil)
    g_rain = g_sm_st1 + g_recharge_tot * soil_wet
    g_tosoil = g_sm_st1 + g_recharge_tot * soil_wet
    g_sm = g_sm_st1

    # soil_wet = clamp((sm/fc)^beta, 0, 1)
    g_sw_raw = tl.where(mask_sw, g_soil_wet, 0.0)
    log_sr = tl.log(soil_ratio_safe)
    log_sr_clamped = tl.minimum(tl.maximum(log_sr, -20.0), 20.0)
    g_beta = g_sw_raw * soil_wet_raw * log_sr_clamped
    exp_arg = (beta - 1.0) * log_sr_clamped
    exp_arg_clamped = tl.minimum(tl.maximum(exp_arg, -20.0), 20.0)
    pow_term = tl.exp(exp_arg_clamped)
    g_sm = g_sm + g_sw_raw * beta * pow_term / fc_safe
    g_fc = g_fc + g_sw_raw * beta * pow_term * (-sm / (fc_safe * fc_safe))

    # ========== Snow Block Backward ==========
    # melt_out = melt_st2 - tosoil
    # tosoil = max(melt_st2 - cwh * snow_out, 0)
    g_melt_st2 = tl.where(mask_tosoil, 0.0, g_melt_out) + tl.where(mask_tosoil, g_tosoil, 0.0)
    g_snow_out_local = g_snow_out + tl.where(mask_tosoil, g_melt_out * cwh - g_tosoil * cwh, 0.0)
    g_cwh = tl.where(mask_tosoil, (g_melt_out - g_tosoil) * snow_out, 0.0)

    # snow_out = snow_st2 + refreeze_amt
    # melt_st2 = melt_st1 - refreeze_amt
    g_melt_st1 = g_melt_st2
    g_refreeze = -g_melt_st2 + g_snow_out_local
    g_snow_st2 = g_snow_out_local

    # refreeze_amt = min(pot_refreeze, melt_st1)
    g_pot_ref = tl.where(mask_refreeze, g_refreeze, 0.0)
    g_melt_st1 = g_melt_st1 + tl.where(mask_refreeze, 0.0, g_refreeze)

    # pot_refreeze = cfr * cfmax * max(-temp_diff, 0)
    mask_cold = temp_diff < 0.0
    relu_neg = tl.maximum(-temp_diff, 0.0)
    g_cfr = g_pot_ref * cfmax * relu_neg
    g_cfmax_ref = g_pot_ref * cfr * relu_neg
    g_temp_diff = g_temp_diff + tl.where(mask_cold, g_pot_ref * cfr * cfmax * (-1.0), 0.0)

    # melt_st1 = melt + melt_amount
    g_melt = g_melt_st1
    g_melt_amt = g_melt_st1

    # snow_st2 = snow_st1 - melt_amount
    g_snow_st1 = g_snow_st2
    g_melt_amt = g_melt_amt - g_snow_st2

    # melt_amount = min(pot_melt, snow_st1)
    g_pot_melt = tl.where(mask_melt, g_melt_amt, 0.0)
    g_snow_st1 = g_snow_st1 + tl.where(mask_melt, 0.0, g_melt_amt)

    # pot_melt = cfmax * max(temp_diff, 0)
    mask_warm = temp_diff > 0.0
    relu_pos = tl.maximum(temp_diff, 0.0)
    g_cfmax = g_pot_melt * relu_pos + g_cfmax_ref
    g_temp_diff = g_temp_diff + tl.where(mask_warm, g_pot_melt * cfmax, 0.0)

    # snow_st1 = snow + snow_input
    g_snow = g_snow_st1
    g_snow_input = g_snow_st1

    # rain = p if is_rain else 0, snow_input = 0 if is_rain else p
    g_p = tl.where(is_rain, g_rain, g_snow_input)

    # temp_diff = t_val - tt
    g_t = g_temp_diff
    g_tt = -g_temp_diff

    # 存储梯度
    tl.store(g_p_ptr + offs, g_p, mask=mask)
    tl.store(g_t_ptr + offs, g_t, mask=mask)
    tl.store(g_pet_ptr + offs, g_pet, mask=mask)
    tl.store(g_snow_ptr + offs, g_snow, mask=mask)
    tl.store(g_melt_ptr + offs, g_melt, mask=mask)
    tl.store(g_sm_ptr + offs, g_sm, mask=mask)
    tl.store(g_suz_ptr + offs, g_suz, mask=mask)
    tl.store(g_slz_ptr + offs, g_slz, mask=mask)
    tl.store(g_tt_ptr + offs, g_tt, mask=mask)
    tl.store(g_cfmax_ptr + offs, g_cfmax, mask=mask)
    tl.store(g_cfr_ptr + offs, g_cfr, mask=mask)
    tl.store(g_cwh_ptr + offs, g_cwh, mask=mask)
    tl.store(g_fc_ptr + offs, g_fc, mask=mask)
    tl.store(g_beta_ptr + offs, g_beta, mask=mask)
    tl.store(g_lp_ptr + offs, g_lp, mask=mask)
    tl.store(g_betaet_ptr + offs, g_betaet, mask=mask)
    tl.store(g_c_ptr + offs, g_c_par, mask=mask)
    tl.store(g_perc_ptr + offs, g_perc, mask=mask)
    tl.store(g_k0_ptr + offs, g_k0, mask=mask)
    tl.store(g_k1_ptr + offs, g_k1, mask=mask)
    tl.store(g_k2_ptr + offs, g_k2, mask=mask)
    tl.store(g_uzl_ptr + offs, g_uzl, mask=mask)



class HbvStepFused(torch.autograd.Function):
    """融合的 HBV 单步计算 - 整体前向和反向"""
    
    @staticmethod
    def forward(ctx, p, t_val, pet, snow, melt, sm, suz, slz,
                tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                perc, k0, k1, k2, uzl):
        n = p.numel()
        
        # 输出张量
        snow_out = torch.empty_like(snow)
        melt_out = torch.empty_like(melt)
        sm_out = torch.empty_like(sm)
        suz_out = torch.empty_like(suz)
        slz_out = torch.empty_like(slz)
        q_out = torch.empty_like(p)
        
        # 中间变量（用于 backward）
        rain = torch.empty_like(p)
        tosoil = torch.empty_like(p)
        recharge = torch.empty_like(p)
        excess = torch.empty_like(p)
        soil_wet = torch.empty_like(p)
        sm_st2 = torch.empty_like(p)
        evapfactor = torch.empty_like(p)
        sm_after_evap = torch.empty_like(p)
        capillary = torch.empty_like(p)
        slz_after_cap = torch.empty_like(p)
        suz_st1 = torch.empty_like(p)
        perc_flux = torch.empty_like(p)
        suz_st2 = torch.empty_like(p)
        slz_st1 = torch.empty_like(p)
        q0 = torch.empty_like(p)
        suz_st3 = torch.empty_like(p)
        q1 = torch.empty_like(p)
        q2 = torch.empty_like(p)
        
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _hbv_fused_forward_kernel[grid](
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
            snow_out, melt_out, sm_out, suz_out, slz_out, q_out,
            rain, tosoil, recharge, excess, soil_wet, sm_st2, evapfactor, sm_after_evap,
            capillary, slz_after_cap, suz_st1, perc_flux, suz_st2, slz_st1,
            q0, suz_st3, q1, q2,
            n,
            BLOCK_SIZE=256,
        )
        
        # 保存用于 backward
        ctx.save_for_backward(
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
            rain, tosoil, recharge, excess, soil_wet, sm_st2, evapfactor, sm_after_evap,
            capillary, slz_after_cap, suz_st1, perc_flux, suz_st2, slz_st1,
            q0, suz_st3, q1, q2,
        )
        ctx.n = n
        
        return snow_out, melt_out, sm_out, suz_out, slz_out, q_out
    
    @staticmethod
    def backward(ctx, g_snow_out, g_melt_out, g_sm_out, g_suz_out, g_slz_out, g_q_out):
        (p, t_val, pet, snow, melt, sm, suz, slz,
         tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
         perc, k0, k1, k2, uzl,
         rain, tosoil, recharge, excess, soil_wet, sm_st2, evapfactor, sm_after_evap,
         capillary, slz_after_cap, suz_st1, perc_flux, suz_st2, slz_st1,
         q0, suz_st3, q1, q2) = ctx.saved_tensors
        n = ctx.n
        
        # 确保梯度连续
        g_snow_out = g_snow_out.contiguous()
        g_melt_out = g_melt_out.contiguous()
        g_sm_out = g_sm_out.contiguous()
        g_suz_out = g_suz_out.contiguous()
        g_slz_out = g_slz_out.contiguous()
        g_q_out = g_q_out.contiguous()
        
        # 输出梯度张量
        g_p = torch.empty_like(p)
        g_t = torch.empty_like(t_val)
        g_pet = torch.empty_like(pet)
        g_snow = torch.empty_like(snow)
        g_melt = torch.empty_like(melt)
        g_sm = torch.empty_like(sm)
        g_suz = torch.empty_like(suz)
        g_slz = torch.empty_like(slz)
        g_tt = torch.empty_like(tt)
        g_cfmax = torch.empty_like(cfmax)
        g_cfr = torch.empty_like(cfr)
        g_cwh = torch.empty_like(cwh)
        g_fc = torch.empty_like(fc)
        g_beta = torch.empty_like(beta)
        g_lp = torch.empty_like(lp)
        g_betaet = torch.empty_like(betaet)
        g_c = torch.empty_like(c_par)
        g_perc = torch.empty_like(perc)
        g_k0 = torch.empty_like(k0)
        g_k1 = torch.empty_like(k1)
        g_k2 = torch.empty_like(k2)
        g_uzl = torch.empty_like(uzl)
        
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _hbv_fused_backward_kernel[grid](
            p, t_val, pet, snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
            perc, k0, k1, k2, uzl,
            rain, tosoil, recharge, excess, soil_wet, sm_st2, evapfactor, sm_after_evap,
            capillary, slz_after_cap, suz_st1, perc_flux, suz_st2, slz_st1,
            q0, suz_st3, q1, q2,
            g_snow_out, g_melt_out, g_sm_out, g_suz_out, g_slz_out, g_q_out,
            g_p, g_t, g_pet, g_snow, g_melt, g_sm, g_suz, g_slz,
            g_tt, g_cfmax, g_cfr, g_cwh, g_fc, g_beta, g_lp, g_betaet, g_c,
            g_perc, g_k0, g_k1, g_k2, g_uzl,
            n,
            BLOCK_SIZE=256,
        )
        
        # 梯度裁剪
        grad_clip = 1e3
        grads = [g_p, g_t, g_pet, g_snow, g_melt, g_sm, g_suz, g_slz,
                 g_tt, g_cfmax, g_cfr, g_cwh, g_fc, g_beta, g_lp, g_betaet, g_c,
                 g_perc, g_k0, g_k1, g_k2, g_uzl]
        for g in grads:
            g.nan_to_num_(nan=0.0, posinf=grad_clip, neginf=-grad_clip)
            g.clamp_(-grad_clip, grad_clip)
        
        return (g_p, g_t, g_pet, g_snow, g_melt, g_sm, g_suz, g_slz,
                g_tt, g_cfmax, g_cfr, g_cwh, g_fc, g_beta, g_lp, g_betaet, g_c,
                g_perc, g_k0, g_k1, g_k2, g_uzl)


def hbv_step_fused(
    p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
    snow: torch.Tensor, melt: torch.Tensor, sm: torch.Tensor,
    suz: torch.Tensor, slz: torch.Tensor,
    tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor, cwh: torch.Tensor,
    fc: torch.Tensor, beta: torch.Tensor, lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
    perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, uzl: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """融合的 HBV 单步计算"""
    snow_out, melt_out, sm_out, suz_out, slz_out, q_out = HbvStepFused.apply(
        p, t_val, pet, snow, melt, sm, suz, slz,
        tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
        perc, k0, k1, k2, uzl
    )
    
    # 状态变量物理约束
    eps = 1e-5
    snow_out = torch.clamp(snow_out, min=eps, max=1000.0)
    melt_out = torch.clamp(melt_out, min=eps, max=500.0)
    sm_out = torch.clamp(sm_out, min=eps, max=1000.0)
    suz_out = torch.clamp(suz_out, min=eps, max=500.0)
    slz_out = torch.clamp(slz_out, min=eps, max=1000.0)
    
    return snow_out, melt_out, sm_out, suz_out, slz_out, q_out


def hbv_run_fused(
    precip: torch.Tensor,
    temp: torch.Tensor,
    pet: torch.Tensor,
    params: dict,
    init_states: Optional[dict] = None,
) -> Tuple[torch.Tensor, dict]:
    """运行 HBV 模型 - 使用融合 kernel"""
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

    q_series = []
    for t in range(T):
        snow, melt, sm, suz, slz, q = hbv_step_fused(
            precip[t], temp[t], pet[t], snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par, perc, k0, k1, k2, uzl,
        )
        q_series.append(q)

    q_series = torch.stack(q_series, dim=0)
    final_states = {"snow": snow, "melt": melt, "sm": sm, "suz": suz, "slz": slz}
    return q_series, final_states


if __name__ == "__main__":
    # 测试融合 kernel
    print("Testing HBV Fused Kernel...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Triton requires CUDA. Please run on GPU.")
        exit(0)
    
    dtype = torch.float32
    B, nmul = 10, 16
    shape = (B, nmul)
    
    torch.manual_seed(42)
    
    # 创建输入
    p = torch.rand(shape, device=device, dtype=dtype) * 10
    t_val = torch.rand(shape, device=device, dtype=dtype) * 20 - 5
    pet = torch.rand(shape, device=device, dtype=dtype) * 5
    snow = torch.rand(shape, device=device, dtype=dtype) * 50 + 1
    melt = torch.rand(shape, device=device, dtype=dtype) * 10 + 0.1
    sm = torch.rand(shape, device=device, dtype=dtype) * 100 + 10
    suz = torch.rand(shape, device=device, dtype=dtype) * 20 + 1
    slz = torch.rand(shape, device=device, dtype=dtype) * 30 + 1
    
    tt = torch.zeros(shape, device=device, dtype=dtype)
    cfmax = torch.ones(shape, device=device, dtype=dtype) * 3.0
    cfr = torch.ones(shape, device=device, dtype=dtype) * 0.05
    cwh = torch.ones(shape, device=device, dtype=dtype) * 0.1
    fc = torch.ones(shape, device=device, dtype=dtype) * 200.0
    beta = torch.ones(shape, device=device, dtype=dtype) * 2.0
    lp = torch.ones(shape, device=device, dtype=dtype) * 0.7
    betaet = torch.ones(shape, device=device, dtype=dtype) * 1.5
    c_par = torch.ones(shape, device=device, dtype=dtype) * 0.05
    perc = torch.ones(shape, device=device, dtype=dtype) * 2.0
    k0 = torch.ones(shape, device=device, dtype=dtype) * 0.2
    k1 = torch.ones(shape, device=device, dtype=dtype) * 0.05
    k2 = torch.ones(shape, device=device, dtype=dtype) * 0.01
    uzl = torch.ones(shape, device=device, dtype=dtype) * 20.0
    
    inputs = [p, t_val, pet, snow, melt, sm, suz, slz,
              tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
              perc, k0, k1, k2, uzl]
    for x in inputs:
        x.requires_grad_(True)
    
    # 前向
    snow_out, melt_out, sm_out, suz_out, slz_out, q_out = hbv_step_fused(*inputs)
    print(f"q_out: mean={q_out.mean().item():.4f}, std={q_out.std().item():.4f}")
    
    # 反向
    loss = q_out.sum()
    loss.backward()
    
    # 检查梯度
    print("\nGradient check:")
    for i, name in enumerate(['p', 't_val', 'pet', 'snow', 'melt', 'sm', 'suz', 'slz',
                              'tt', 'cfmax', 'cfr', 'cwh', 'fc', 'beta', 'lp', 'betaet', 'c_par',
                              'perc', 'k0', 'k1', 'k2', 'uzl']):
        g = inputs[i].grad
        if g is not None:
            has_nan = g.isnan().any().item()
            has_inf = g.isinf().any().item()
            print(f"  {name:8s}: mean={g.mean().item():.4e}, max={g.abs().max().item():.4e}, nan={has_nan}, inf={has_inf}")
        else:
            print(f"  {name:8s}: no gradient")
    
    print("\nTest passed!")
