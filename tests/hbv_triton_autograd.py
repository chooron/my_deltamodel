"""
HBV 水文模型 Triton 加速实现
支持 PyTorch 反向传播 (autograd.Function)

基于 test_hbv_gradient.py 中验证正确的手动梯度实现
使用 Triton 编译提升计算效率

Author: Generated based on verified PyTorch gradient implementation
Date: 2025-12-10
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# ==========================================
# Triton 辅助函数
# ==========================================

@triton.jit
def _tanh_val(x):
    """tanh(5x) for smooth step, 使用 tanh(x) = 2*sigmoid(2x) - 1"""
    # Triton 3.x 直接提供 tl.sigmoid
    return 2.0 * tl.sigmoid(10.0 * x) - 1.0


@triton.jit
def _smooth_step(x):
    """平滑阶跃函数: 0.5 * (tanh(5x) + 1)"""
    return 0.5 * (_tanh_val(x) + 1.0)


@triton.jit
def _smooth_relu(x):
    """平滑 ReLU: x * smooth_step(x)"""
    return x * _smooth_step(x)


@triton.jit
def _d_smooth_relu(x):
    """
    smooth_relu 的导数
    Forward: y = x * 0.5 * (tanh(5x) + 1)
    Derivative: 0.5 * (tanh(5x) + 1) + 2.5 * x * (1 - tanh^2(5x))
    返回 dy/dx
    """
    t = _tanh_val(x)
    s = 0.5 * (t + 1.0)  # smooth_step
    ds = 2.5 * (1.0 - t * t)  # d(smooth_step)/dx
    return s + x * ds


@triton.jit
def _d_smooth_step(x):
    """
    smooth_step 的导数
    Forward: y = 0.5 * (tanh(5x) + 1)
    Derivative: 2.5 * (1 - tanh^2(5x))
    返回 dy/dx
    """
    t = _tanh_val(x)
    return 2.5 * (1.0 - t * t)


# ==========================================
# Triton 前向传播 Kernel
# ==========================================

@triton.jit
def hbv_forward_kernel(
    # --- 输入指针 ---
    p_ptr, t_val_ptr, pet_ptr,
    snow_in_ptr, melt_in_ptr, sm_in_ptr, suz_in_ptr, slz_in_ptr,
    # --- 参数指针 ---
    tt_ptr, cfmax_ptr, cfr_ptr, cwh_ptr, fc_ptr, beta_ptr, lp_ptr, betaet_ptr,
    c_par_ptr, perc_ptr, k0_ptr, k1_ptr, k2_ptr, uzl_ptr,
    # --- 输出指针 ---
    snow_out_ptr, melt_out_ptr, sm_out_ptr, suz_out_ptr, slz_out_ptr, q_out_ptr,
    # --- 配置 ---
    nearzero: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """HBV 前向传播 Triton Kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    t_val = tl.load(t_val_ptr + offsets, mask=mask, other=0.0)
    pet = tl.load(pet_ptr + offsets, mask=mask, other=0.0)
    
    snow = tl.load(snow_in_ptr + offsets, mask=mask, other=0.0)
    melt = tl.load(melt_in_ptr + offsets, mask=mask, other=0.0)
    sm = tl.load(sm_in_ptr + offsets, mask=mask, other=0.0)
    suz = tl.load(suz_in_ptr + offsets, mask=mask, other=0.0)
    slz = tl.load(slz_in_ptr + offsets, mask=mask, other=0.0)
    
    tt = tl.load(tt_ptr + offsets, mask=mask, other=0.0)
    cfmax = tl.load(cfmax_ptr + offsets, mask=mask, other=0.0)
    cfr = tl.load(cfr_ptr + offsets, mask=mask, other=0.0)
    cwh = tl.load(cwh_ptr + offsets, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    lp = tl.load(lp_ptr + offsets, mask=mask, other=0.0)
    betaet = tl.load(betaet_ptr + offsets, mask=mask, other=0.0)
    c_par = tl.load(c_par_ptr + offsets, mask=mask, other=0.0)
    perc = tl.load(perc_ptr + offsets, mask=mask, other=0.0)
    k0 = tl.load(k0_ptr + offsets, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offsets, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offsets, mask=mask, other=0.0)
    uzl = tl.load(uzl_ptr + offsets, mask=mask, other=0.0)

    # =========================================================
    # 2. FORWARD COMPUTATION
    # =========================================================
    eps = 1e-6
    
    # --- 1. 降水分离 ---
    temp_diff = t_val - tt
    is_warm = _smooth_step(temp_diff)
    
    rain = p * is_warm
    snow_input = p * (1.0 - is_warm)

    # --- 2. 积雪模块 ---
    snow = snow + snow_input
    
    pot_melt = cfmax * _smooth_relu(temp_diff)
    melt_amount = tl.minimum(pot_melt, snow)
    
    snow = snow - melt_amount
    melt = melt + melt_amount

    # 再冻结
    pot_refreeze = cfr * cfmax * _smooth_relu(-temp_diff)
    refreezing = tl.minimum(pot_refreeze, melt)
    
    snow = snow + refreezing
    melt = melt - refreezing

    # 融雪出流
    tosoil = _smooth_relu(melt - cwh * snow)
    melt = melt - tosoil

    # --- 3. 土壤模块 ---
    ratio = sm / (fc + eps)
    ratio_safe = tl.minimum(tl.maximum(ratio, 0.0), 1.0)
    
    soil_wetness = tl.exp(beta * tl.log(ratio_safe + eps))
    
    recharge = (rain + tosoil) * soil_wetness
    sm = sm + rain + tosoil - recharge

    # 超渗
    excess = _smooth_relu(sm - fc)
    sm = sm - excess

    # 蒸发
    limit_val = lp * fc
    evap_ratio = sm / (limit_val + eps)
    evap_ratio_safe = tl.minimum(tl.maximum(evap_ratio, 0.0), 1.0)
    
    evapfactor = tl.exp(betaet * tl.log(evap_ratio_safe + eps))
    
    potential_et = pet * evapfactor
    etact = sm - _smooth_relu(sm - potential_et)
    
    sm = tl.maximum(sm - etact, nearzero)

    # --- 4. 汇流模块 ---
    capillary = tl.minimum(slz, c_par * slz * (1.0 - ratio_safe))
    sm = sm + capillary
    slz = slz - capillary

    suz = suz + recharge + excess
    perc_flux = tl.minimum(suz, perc)
    suz = suz - perc_flux

    # Q0
    q0 = k0 * _smooth_relu(suz - uzl)
    suz = suz - q0

    q1 = k1 * suz
    suz = suz - q1

    slz = slz + perc_flux
    q2 = k2 * slz
    slz = slz - q2

    q_total = q0 + q1 + q2

    # =========================================================
    # 3. STORE OUTPUTS
    # =========================================================
    tl.store(snow_out_ptr + offsets, snow, mask=mask)
    tl.store(melt_out_ptr + offsets, melt, mask=mask)
    tl.store(sm_out_ptr + offsets, sm, mask=mask)
    tl.store(suz_out_ptr + offsets, suz, mask=mask)
    tl.store(slz_out_ptr + offsets, slz, mask=mask)
    tl.store(q_out_ptr + offsets, q_total, mask=mask)


# ==========================================
# Triton 反向传播 Kernel
# ==========================================

@triton.jit
def hbv_backward_kernel(
    # --- Forward 输入指针 (用于重计算) ---
    p_ptr, t_val_ptr, pet_ptr,
    snow_prev_ptr, melt_prev_ptr, sm_prev_ptr, suz_prev_ptr, slz_prev_ptr,
    # --- 参数指针 ---
    tt_ptr, cfmax_ptr, cfr_ptr, cwh_ptr, fc_ptr, beta_ptr, lp_ptr, betaet_ptr,
    c_par_ptr, perc_ptr, k0_ptr, k1_ptr, k2_ptr, uzl_ptr,
    # --- 梯度输入指针 (来自下一时刻或 Loss) ---
    d_q_total_ptr, d_snow_next_ptr, d_melt_next_ptr, d_sm_next_ptr, d_suz_next_ptr, d_slz_next_ptr,
    # --- 梯度输出指针 ---
    d_p_out_ptr, d_t_val_out_ptr, d_pet_out_ptr,
    d_snow_prev_out_ptr, d_melt_prev_out_ptr, d_sm_prev_out_ptr, d_suz_prev_out_ptr, d_slz_prev_out_ptr,
    d_tt_out_ptr, d_cfmax_out_ptr, d_cfr_out_ptr, d_cwh_out_ptr, d_fc_out_ptr,
    d_beta_out_ptr, d_lp_out_ptr, d_betaet_out_ptr, d_c_par_out_ptr, d_perc_out_ptr,
    d_k0_out_ptr, d_k1_out_ptr, d_k2_out_ptr, d_uzl_out_ptr,
    # --- 配置 ---
    nearzero: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """HBV 反向传播 Triton Kernel - 基于验证过的 PyTorch 手动梯度实现"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    eps = 1e-6

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    # Load inputs
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    t_val = tl.load(t_val_ptr + offsets, mask=mask, other=0.0)
    pet = tl.load(pet_ptr + offsets, mask=mask, other=0.0)
    
    snow_prev = tl.load(snow_prev_ptr + offsets, mask=mask, other=0.0)
    melt_prev = tl.load(melt_prev_ptr + offsets, mask=mask, other=0.0)
    sm_prev = tl.load(sm_prev_ptr + offsets, mask=mask, other=0.0)
    suz_prev = tl.load(suz_prev_ptr + offsets, mask=mask, other=0.0)
    slz_prev = tl.load(slz_prev_ptr + offsets, mask=mask, other=0.0)

    # Load parameters
    tt = tl.load(tt_ptr + offsets, mask=mask, other=0.0)
    cfmax = tl.load(cfmax_ptr + offsets, mask=mask, other=0.0)
    cfr = tl.load(cfr_ptr + offsets, mask=mask, other=0.0)
    cwh = tl.load(cwh_ptr + offsets, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    lp = tl.load(lp_ptr + offsets, mask=mask, other=0.0)
    betaet = tl.load(betaet_ptr + offsets, mask=mask, other=0.0)
    c_par = tl.load(c_par_ptr + offsets, mask=mask, other=0.0)
    perc = tl.load(perc_ptr + offsets, mask=mask, other=0.0)
    k0 = tl.load(k0_ptr + offsets, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offsets, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offsets, mask=mask, other=0.0)
    uzl = tl.load(uzl_ptr + offsets, mask=mask, other=0.0)

    # Load incoming gradients
    d_q_total = tl.load(d_q_total_ptr + offsets, mask=mask, other=0.0)
    d_snow_next = tl.load(d_snow_next_ptr + offsets, mask=mask, other=0.0)
    d_melt_next = tl.load(d_melt_next_ptr + offsets, mask=mask, other=0.0)
    d_sm_next = tl.load(d_sm_next_ptr + offsets, mask=mask, other=0.0)
    d_suz_next = tl.load(d_suz_next_ptr + offsets, mask=mask, other=0.0)
    d_slz_next = tl.load(d_slz_next_ptr + offsets, mask=mask, other=0.0)

    # =========================================================
    # 2. PHASE 1: FORWARD RECOMPUTE
    # =========================================================
    # [1. Precip]
    temp_diff = t_val - tt
    smooth_step_val = _smooth_step(temp_diff)
    rain = p * smooth_step_val
    snow_input = p * (1.0 - smooth_step_val)

    # [2. Snow]
    snow_st1 = snow_prev + snow_input
    pot_melt = cfmax * _smooth_relu(temp_diff)
    is_melt_limited = pot_melt < snow_st1
    melt_amt = tl.where(is_melt_limited, pot_melt, snow_st1)

    snow_st2 = snow_st1 - melt_amt
    melt_st1 = melt_prev + melt_amt

    pot_refreeze = cfr * cfmax * _smooth_relu(-temp_diff)
    is_refreeze_limited = pot_refreeze < melt_st1
    refreeze_amt = tl.where(is_refreeze_limited, pot_refreeze, melt_st1)

    snow_st3 = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt

    tosoil = _smooth_relu(melt_st2 - cwh * snow_st3)

    # [3. Soil]
    ratio = sm_prev / (fc + eps)
    ratio_safe = tl.minimum(tl.maximum(ratio, 0.0), 1.0)
    soil_wetness = tl.exp(beta * tl.log(ratio_safe + eps))

    recharge = (rain + tosoil) * soil_wetness
    sm_st1 = sm_prev + rain + tosoil - recharge

    excess = _smooth_relu(sm_st1 - fc)
    sm_st2 = sm_st1 - excess

    limit_val = lp * fc
    evap_ratio = sm_st2 / (limit_val + eps)
    evap_ratio_safe = tl.minimum(tl.maximum(evap_ratio, 0.0), 1.0)
    evapfactor = tl.exp(betaet * tl.log(evap_ratio_safe + eps))

    pot_et = pet * evapfactor
    sm_temp = _smooth_relu(sm_st2 - pot_et)
    # 注意：forward 中是先 clamp 再加 capillary
    sm_after_et = tl.maximum(sm_temp, nearzero)

    # [4. Capillary & GW]
    cap_threshold = c_par * slz_prev * (1.0 - ratio_safe)
    capillary = tl.minimum(slz_prev, cap_threshold)
    is_cap_limited = slz_prev < cap_threshold

    sm_final_fwd = sm_after_et + capillary  # forward: sm = max(...) + capillary
    slz_st1 = slz_prev - capillary

    suz_st1 = suz_prev + recharge + excess
    perc_flux = tl.minimum(suz_st1, perc)
    is_perc_limited = suz_st1 < perc

    suz_st2 = suz_st1 - perc_flux
    slz_st2 = slz_st1 + perc_flux

    q0 = k0 * _smooth_relu(suz_st2 - uzl)
    suz_st3 = suz_st2 - q0

    q1 = k1 * suz_st3
    # suz_final = suz_st3 - q1

    q2 = k2 * slz_st2
    # slz_final = slz_st2 - q2

    # =========================================================
    # 3. PHASE 2: GRADIENT BACKPROP
    # =========================================================
    
    # --- 初始化梯度累加器 ---
    d_snow = d_snow_next
    d_melt = d_melt_next
    d_sm = d_sm_next
    d_suz = d_suz_next
    d_slz = d_slz_next

    # 参数梯度初始化
    d_tt = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_cfmax = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_cfr = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_cwh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_fc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_beta = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_lp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_betaet = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_c_par = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_perc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_k0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_k1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_k2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_uzl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # --- Q Total 分发 ---
    d_q0 = d_q_total
    d_q1 = d_q_total
    d_q2 = d_q_total

    # --- Q2: q2 = k2 * slz_st2, slz_final = slz_st2 - q2 ---
    d_k2 = d_k2 + d_q2 * slz_st2
    d_slz_st2 = d_slz * (1.0 - k2) + d_q2 * k2

    # --- Q1: q1 = k1 * suz_st3, suz_final = suz_st3 - q1 ---
    d_k1 = d_k1 + d_q1 * suz_st3
    d_suz_st3 = d_suz * (1.0 - k1) + d_q1 * k1

    # --- Q0: q0 = k0 * smooth_relu(suz_st2 - uzl), suz_st3 = suz_st2 - q0 ---
    val_sr_q0 = _smooth_relu(suz_st2 - uzl)
    d_sr_q0 = _d_smooth_relu(suz_st2 - uzl)
    
    d_k0 = d_k0 + d_q0 * val_sr_q0
    d_uzl = d_uzl + (d_q0 - d_suz_st3) * k0 * d_sr_q0 * (-1.0)
    d_suz_st2 = d_suz_st3 + (d_q0 - d_suz_st3) * k0 * d_sr_q0

    # --- Perc & slz_st2 ---
    d_perc_flux = d_slz_st2 - d_suz_st2
    d_suz_st1 = d_suz_st2 + tl.where(is_perc_limited, d_perc_flux, 0.0)
    d_perc = d_perc + tl.where(is_perc_limited, 0.0, d_perc_flux)
    d_slz_st1 = d_slz_st2

    # --- suz_st1 = suz_prev + recharge + excess ---
    d_suz_prev = d_suz_st1

    # --- Capillary ---
    d_capillary = d_sm - d_slz_st1
    d_term2 = tl.where(is_cap_limited, 0.0, d_capillary)
    d_slz_prev_cap = tl.where(is_cap_limited, d_capillary, 0.0)

    d_c_par = d_c_par + d_term2 * slz_prev * (1.0 - ratio_safe)
    d_slz_prev_cap = d_slz_prev_cap + d_term2 * c_par * (1.0 - ratio_safe)
    d_ratio_cap = d_term2 * c_par * slz_prev * (-1.0)
    
    d_slz_prev = d_slz_st1 + d_slz_prev_cap

    # --- Soil Backprop ---
    # sm_final = sm_after_et + capillary (capillary 梯度已在上面处理)
    # sm_after_et = max(sm_temp, nearzero)
    # sm_temp = smooth_relu(sm_st2 - pot_et)
    d_sm_after_et = d_sm  # d_sm 直接传给 sm_after_et（capillary 的贡献已处理）
    
    # sm_after_et = max(sm_temp, nearzero)
    # 当 sm_temp > nearzero 时梯度传递，否则为 0
    d_sm_temp = tl.where(sm_temp > nearzero, d_sm_after_et, 0.0)
    
    # sm_temp = smooth_relu(sm_st2 - pot_et)
    d_sr_et = _d_smooth_relu(sm_st2 - pot_et) * d_sm_temp
    d_sm_st2 = d_sr_et
    d_pot_et = -d_sr_et

    d_evapfactor = d_pot_et * pet

    base_evap = evap_ratio_safe + eps
    d_base_evap = d_evapfactor * betaet * tl.exp((betaet - 1.0) * tl.log(base_evap))
    d_betaet = d_betaet + d_evapfactor * evapfactor * tl.log(base_evap)

    mask_evap_range = (evap_ratio > 0.0) & (evap_ratio < 1.0)
    d_evap_ratio = tl.where(mask_evap_range, d_base_evap, 0.0)

    d_sm_st2 = d_sm_st2 + d_evap_ratio * (1.0 / (limit_val + eps))
    d_lp = d_lp + d_evap_ratio * sm_st2 * (-1.0 / (limit_val * lp + 1e-10))
    d_fc = d_fc + d_evap_ratio * sm_st2 * (-1.0 / (limit_val * fc + 1e-10))

    # --- Excess ---
    d_excess = d_suz_st1 - d_sm_st2
    d_sr_ex = _d_smooth_relu(sm_st1 - fc) * d_excess
    d_sm_st1 = d_sm_st2 + d_sr_ex
    d_fc = d_fc - d_sr_ex

    # --- Recharge ---
    d_recharge_total = d_suz_st1 - d_sm_st1
    
    d_soil_wetness = d_recharge_total * (rain + tosoil)
    d_rain = d_sm_st1 + d_recharge_total * soil_wetness
    d_tosoil = d_sm_st1 + d_recharge_total * soil_wetness
    
    d_sm_prev = d_sm_st1

    # --- Soil Wetness ---
    base_sw = ratio_safe + eps
    d_base_sw = d_soil_wetness * beta * tl.exp((beta - 1.0) * tl.log(base_sw))
    d_beta = d_beta + d_soil_wetness * soil_wetness * tl.log(base_sw)

    mask_sw_range = (ratio > 0.0) & (ratio < 1.0)
    d_ratio = tl.where(mask_sw_range, d_base_sw, 0.0) + d_ratio_cap

    d_sm_prev = d_sm_prev + d_ratio * (1.0 / (fc + eps))
    d_fc = d_fc + d_ratio * sm_prev * (-1.0 / ((fc + eps) * (fc + eps)))

    # --- Snow Backprop ---
    d_sr_tosoil = _d_smooth_relu(melt_st2 - cwh * snow_st3) * d_tosoil
    
    d_melt_st2 = d_melt + d_sr_tosoil
    d_snow_st3 = d_snow + d_sr_tosoil * (-cwh)
    d_cwh = d_cwh + d_sr_tosoil * (-snow_st3)

    # --- Refreeze ---
    d_refreeze_amt = d_snow_st3 - d_melt_st2
    d_pot_ref = tl.where(is_refreeze_limited, d_refreeze_amt, 0.0)
    d_melt_st1_from_refreeze = tl.where(is_refreeze_limited, 0.0, d_refreeze_amt)
    
    d_melt_st1 = d_melt_st2 + d_melt_st1_from_refreeze
    d_snow_st2 = d_snow_st3

    sr_neg = _smooth_relu(-temp_diff)
    d_cfr = d_cfr + d_pot_ref * cfmax * sr_neg
    d_cfmax = d_cfmax + d_pot_ref * cfr * sr_neg
    d_sr_neg = _d_smooth_relu(-temp_diff) * (d_pot_ref * cfr * cfmax)
    d_temp_diff = -d_sr_neg

    # --- Melt ---
    d_melt_amt = d_melt_st1 - d_snow_st2
    d_pot_melt = tl.where(is_melt_limited, d_melt_amt, 0.0)
    d_snow_st1_from_melt = tl.where(is_melt_limited, 0.0, d_melt_amt)
    
    d_snow_st1 = d_snow_st2 + d_snow_st1_from_melt
    d_melt_prev = d_melt_st1

    sr_pos = _smooth_relu(temp_diff)
    d_cfmax = d_cfmax + d_pot_melt * sr_pos
    d_temp_diff = d_temp_diff + _d_smooth_relu(temp_diff) * (d_pot_melt * cfmax)

    # --- Snow Input ---
    d_snow_prev = d_snow_st1
    d_snow_input = d_snow_st1

    # --- Rain/Snow Split ---
    s = smooth_step_val
    d_p = d_rain * s + d_snow_input * (1.0 - s)
    d_s = (d_rain - d_snow_input) * p
    d_temp_diff = d_temp_diff + _d_smooth_step(temp_diff) * d_s

    d_t_val = d_temp_diff
    d_tt = d_tt - d_temp_diff

    # --- PET gradient ---
    d_pet = d_pot_et * (-evapfactor)

    # =========================================================
    # 4. STORE GRADIENTS
    # =========================================================
    tl.store(d_p_out_ptr + offsets, d_p, mask=mask)
    tl.store(d_t_val_out_ptr + offsets, d_t_val, mask=mask)
    tl.store(d_pet_out_ptr + offsets, d_pet, mask=mask)
    
    tl.store(d_snow_prev_out_ptr + offsets, d_snow_prev, mask=mask)
    tl.store(d_melt_prev_out_ptr + offsets, d_melt_prev, mask=mask)
    tl.store(d_sm_prev_out_ptr + offsets, d_sm_prev, mask=mask)
    tl.store(d_suz_prev_out_ptr + offsets, d_suz_prev, mask=mask)
    tl.store(d_slz_prev_out_ptr + offsets, d_slz_prev, mask=mask)
    
    tl.store(d_tt_out_ptr + offsets, d_tt, mask=mask)
    tl.store(d_cfmax_out_ptr + offsets, d_cfmax, mask=mask)
    tl.store(d_cfr_out_ptr + offsets, d_cfr, mask=mask)
    tl.store(d_cwh_out_ptr + offsets, d_cwh, mask=mask)
    tl.store(d_fc_out_ptr + offsets, d_fc, mask=mask)
    tl.store(d_beta_out_ptr + offsets, d_beta, mask=mask)
    tl.store(d_lp_out_ptr + offsets, d_lp, mask=mask)
    tl.store(d_betaet_out_ptr + offsets, d_betaet, mask=mask)
    tl.store(d_c_par_out_ptr + offsets, d_c_par, mask=mask)
    tl.store(d_perc_out_ptr + offsets, d_perc, mask=mask)
    tl.store(d_k0_out_ptr + offsets, d_k0, mask=mask)
    tl.store(d_k1_out_ptr + offsets, d_k1, mask=mask)
    tl.store(d_k2_out_ptr + offsets, d_k2, mask=mask)
    tl.store(d_uzl_out_ptr + offsets, d_uzl, mask=mask)


# ==========================================
# PyTorch 包装函数
# ==========================================

def _hbv_forward_triton(
    p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
    snow: torch.Tensor, melt: torch.Tensor, sm: torch.Tensor,
    suz: torch.Tensor, slz: torch.Tensor,
    tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor,
    cwh: torch.Tensor, fc: torch.Tensor, beta: torch.Tensor,
    lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
    perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor,
    k2: torch.Tensor, uzl: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """Triton 前向传播包装函数"""
    n_elements = p.numel()
    
    # 确保所有输入在同一设备上
    device = p.device
    dtype = p.dtype
    
    # 分配输出张量
    snow_out = torch.empty_like(snow)
    melt_out = torch.empty_like(melt)
    sm_out = torch.empty_like(sm)
    suz_out = torch.empty_like(suz)
    slz_out = torch.empty_like(slz)
    q_out = torch.empty_like(p)
    
    # 计算 grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动 kernel
    hbv_forward_kernel[grid](
        p, t_val, pet,
        snow, melt, sm, suz, slz,
        tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
        c_par, perc, k0, k1, k2, uzl,
        snow_out, melt_out, sm_out, suz_out, slz_out, q_out,
        nearzero,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return snow_out, melt_out, sm_out, suz_out, slz_out, q_out


def _hbv_backward_triton(
    p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
    snow_prev: torch.Tensor, melt_prev: torch.Tensor, sm_prev: torch.Tensor,
    suz_prev: torch.Tensor, slz_prev: torch.Tensor,
    tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor,
    cwh: torch.Tensor, fc: torch.Tensor, beta: torch.Tensor,
    lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
    perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor,
    k2: torch.Tensor, uzl: torch.Tensor,
    d_q_total: torch.Tensor, d_snow_next: torch.Tensor, d_melt_next: torch.Tensor,
    d_sm_next: torch.Tensor, d_suz_next: torch.Tensor, d_slz_next: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """Triton 反向传播包装函数"""
    n_elements = p.numel()
    device = p.device
    dtype = p.dtype
    
    # 分配梯度输出张量
    d_p = torch.empty_like(p)
    d_t_val = torch.empty_like(t_val)
    d_pet = torch.empty_like(pet)
    d_snow_prev = torch.empty_like(snow_prev)
    d_melt_prev = torch.empty_like(melt_prev)
    d_sm_prev = torch.empty_like(sm_prev)
    d_suz_prev = torch.empty_like(suz_prev)
    d_slz_prev = torch.empty_like(slz_prev)
    
    d_tt = torch.empty_like(tt)
    d_cfmax = torch.empty_like(cfmax)
    d_cfr = torch.empty_like(cfr)
    d_cwh = torch.empty_like(cwh)
    d_fc = torch.empty_like(fc)
    d_beta = torch.empty_like(beta)
    d_lp = torch.empty_like(lp)
    d_betaet = torch.empty_like(betaet)
    d_c_par = torch.empty_like(c_par)
    d_perc = torch.empty_like(perc)
    d_k0 = torch.empty_like(k0)
    d_k1 = torch.empty_like(k1)
    d_k2 = torch.empty_like(k2)
    d_uzl = torch.empty_like(uzl)
    
    # 计算 grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动 kernel
    hbv_backward_kernel[grid](
        p, t_val, pet,
        snow_prev, melt_prev, sm_prev, suz_prev, slz_prev,
        tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
        c_par, perc, k0, k1, k2, uzl,
        d_q_total, d_snow_next, d_melt_next, d_sm_next, d_suz_next, d_slz_next,
        d_p, d_t_val, d_pet,
        d_snow_prev, d_melt_prev, d_sm_prev, d_suz_prev, d_slz_prev,
        d_tt, d_cfmax, d_cfr, d_cwh, d_fc,
        d_beta, d_lp, d_betaet, d_c_par, d_perc,
        d_k0, d_k1, d_k2, d_uzl,
        nearzero,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (d_p, d_t_val, d_pet,
            d_snow_prev, d_melt_prev, d_sm_prev, d_suz_prev, d_slz_prev,
            d_tt, d_cfmax, d_cfr, d_cwh, d_fc,
            d_beta, d_lp, d_betaet, d_c_par, d_perc,
            d_k0, d_k1, d_k2, d_uzl)


# ==========================================
# PyTorch Autograd Function
# ==========================================

class HBVTritonFunction(torch.autograd.Function):
    """
    HBV 模型 Triton 加速实现，支持 PyTorch 自动微分
    
    前向传播使用 Triton kernel 计算
    反向传播使用手动实现的 Triton kernel（基于验证过的梯度）
    """
    
    @staticmethod
    def forward(
        ctx,
        p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
        snow: torch.Tensor, melt: torch.Tensor, sm: torch.Tensor,
        suz: torch.Tensor, slz: torch.Tensor,
        tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor,
        cwh: torch.Tensor, fc: torch.Tensor, beta: torch.Tensor,
        lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
        perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor,
        k2: torch.Tensor, uzl: torch.Tensor,
        nearzero: float = 1e-6,
    ) -> Tuple[torch.Tensor, ...]:
        """前向传播"""
        # 确保输入是连续的
        p = p.contiguous()
        t_val = t_val.contiguous()
        pet = pet.contiguous()
        snow = snow.contiguous()
        melt = melt.contiguous()
        sm = sm.contiguous()
        suz = suz.contiguous()
        slz = slz.contiguous()
        tt = tt.contiguous()
        cfmax = cfmax.contiguous()
        cfr = cfr.contiguous()
        cwh = cwh.contiguous()
        fc = fc.contiguous()
        beta = beta.contiguous()
        lp = lp.contiguous()
        betaet = betaet.contiguous()
        c_par = c_par.contiguous()
        perc = perc.contiguous()
        k0 = k0.contiguous()
        k1 = k1.contiguous()
        k2 = k2.contiguous()
        uzl = uzl.contiguous()
        
        # 调用 Triton 前向
        snow_out, melt_out, sm_out, suz_out, slz_out, q_out = _hbv_forward_triton(
            p, t_val, pet,
            snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
            c_par, perc, k0, k1, k2, uzl,
            nearzero,
        )
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(
            p, t_val, pet,
            snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
            c_par, perc, k0, k1, k2, uzl,
        )
        ctx.nearzero = nearzero
        
        return snow_out, melt_out, sm_out, suz_out, slz_out, q_out
    
    @staticmethod
    def backward(
        ctx,
        d_snow_next: torch.Tensor, d_melt_next: torch.Tensor,
        d_sm_next: torch.Tensor, d_suz_next: torch.Tensor,
        d_slz_next: torch.Tensor, d_q_total: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """反向传播"""
        # 恢复保存的张量
        (p, t_val, pet,
         snow_prev, melt_prev, sm_prev, suz_prev, slz_prev,
         tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
         c_par, perc, k0, k1, k2, uzl) = ctx.saved_tensors
        nearzero = ctx.nearzero
        
        # 处理 None 梯度
        device = p.device
        dtype = p.dtype
        shape = p.shape
        
        if d_snow_next is None:
            d_snow_next = torch.zeros(shape, device=device, dtype=dtype)
        if d_melt_next is None:
            d_melt_next = torch.zeros(shape, device=device, dtype=dtype)
        if d_sm_next is None:
            d_sm_next = torch.zeros(shape, device=device, dtype=dtype)
        if d_suz_next is None:
            d_suz_next = torch.zeros(shape, device=device, dtype=dtype)
        if d_slz_next is None:
            d_slz_next = torch.zeros(shape, device=device, dtype=dtype)
        if d_q_total is None:
            d_q_total = torch.zeros(shape, device=device, dtype=dtype)
        
        # 确保梯度是连续的
        d_snow_next = d_snow_next.contiguous()
        d_melt_next = d_melt_next.contiguous()
        d_sm_next = d_sm_next.contiguous()
        d_suz_next = d_suz_next.contiguous()
        d_slz_next = d_slz_next.contiguous()
        d_q_total = d_q_total.contiguous()
        
        # 调用 Triton 反向
        grads = _hbv_backward_triton(
            p, t_val, pet,
            snow_prev, melt_prev, sm_prev, suz_prev, slz_prev,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
            c_par, perc, k0, k1, k2, uzl,
            d_q_total, d_snow_next, d_melt_next, d_sm_next, d_suz_next, d_slz_next,
            nearzero,
        )
        
        (d_p, d_t_val, d_pet,
         d_snow_prev, d_melt_prev, d_sm_prev, d_suz_prev, d_slz_prev,
         d_tt, d_cfmax, d_cfr, d_cwh, d_fc,
         d_beta, d_lp, d_betaet, d_c_par, d_perc,
         d_k0, d_k1, d_k2, d_uzl) = grads
        
        # 返回所有输入参数的梯度 (nearzero 不需要梯度)
        return (d_p, d_t_val, d_pet,
                d_snow_prev, d_melt_prev, d_sm_prev, d_suz_prev, d_slz_prev,
                d_tt, d_cfmax, d_cfr, d_cwh, d_fc,
                d_beta, d_lp, d_betaet, d_c_par, d_perc,
                d_k0, d_k1, d_k2, d_uzl,
                None)  # nearzero


# ==========================================
# 用户友好的 API
# ==========================================

def hbv_step_triton(
    p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
    snow: torch.Tensor, melt: torch.Tensor, sm: torch.Tensor,
    suz: torch.Tensor, slz: torch.Tensor,
    tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor,
    cwh: torch.Tensor, fc: torch.Tensor, beta: torch.Tensor,
    lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
    perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor,
    k2: torch.Tensor, uzl: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HBV 模型单步计算 (Triton 加速版本)
    
    支持 PyTorch 自动微分，可以直接用于训练。
    
    参数:
        p: 降水量 [batch_size] 或 [batch_size, n_grids]
        t_val: 温度 [batch_size] 或 [batch_size, n_grids]
        pet: 潜在蒸发量 [batch_size] 或 [batch_size, n_grids]
        snow: 积雪状态 [batch_size] 或 [batch_size, n_grids]
        melt: 融雪状态 [batch_size] 或 [batch_size, n_grids]
        sm: 土壤含水量 [batch_size] 或 [batch_size, n_grids]
        suz: 上层蓄水量 [batch_size] 或 [batch_size, n_grids]
        slz: 下层蓄水量 [batch_size] 或 [batch_size, n_grids]
        tt: 温度阈值参数
        cfmax: 融雪系数
        cfr: 再冻结系数
        cwh: 持水系数
        fc: 田间持水量
        beta: 土壤参数
        lp: 蒸发比例参数
        betaet: 蒸发指数
        c_par: 毛管上升系数
        perc: 渗透参数
        k0: 快速响应系数
        k1: 中间响应系数
        k2: 慢速响应系数
        uzl: 上层阈值
        nearzero: 数值稳定性小量
    
    返回:
        snow_out: 输出积雪状态
        melt_out: 输出融雪状态
        sm_out: 输出土壤含水量
        suz_out: 输出上层蓄水量
        slz_out: 输出下层蓄水量
        q_total: 总出流量
    """
    return HBVTritonFunction.apply(
        p, t_val, pet,
        snow, melt, sm, suz, slz,
        tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
        c_par, perc, k0, k1, k2, uzl,
        nearzero,
    )


# ==========================================
# 时间序列循环计算
# ==========================================

def hbv_run_triton(
    precip: torch.Tensor,  # [T, batch] or [T, batch, n_grids]
    temp: torch.Tensor,    # [T, batch] or [T, batch, n_grids]
    pet: torch.Tensor,     # [T, batch] or [T, batch, n_grids]
    params: dict,          # 包含所有 HBV 参数的字典
    init_states: Optional[dict] = None,  # 初始状态
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """
    HBV 模型时间序列循环计算 (Triton 加速版本)
    
    参数:
        precip: 降水时间序列 [T, ...]
        temp: 温度时间序列 [T, ...]
        pet: 潜在蒸发时间序列 [T, ...]
        params: HBV 参数字典，包含 tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par, perc, k0, k1, k2, uzl
        init_states: 初始状态字典，包含 snow, melt, sm, suz, slz
        nearzero: 数值稳定性小量
    
    返回:
        q_series: 出流时间序列 [T, ...]
        final_states: 最终状态字典
    """
    T = precip.shape[0]
    device = precip.device
    dtype = precip.dtype
    spatial_shape = precip.shape[1:]
    
    # 初始化状态
    if init_states is None:
        snow = torch.zeros(spatial_shape, device=device, dtype=dtype)
        melt = torch.zeros(spatial_shape, device=device, dtype=dtype)
        sm = torch.ones(spatial_shape, device=device, dtype=dtype) * 50.0
        suz = torch.zeros(spatial_shape, device=device, dtype=dtype)
        slz = torch.ones(spatial_shape, device=device, dtype=dtype) * 10.0
    else:
        snow = init_states['snow']
        melt = init_states['melt']
        sm = init_states['sm']
        suz = init_states['suz']
        slz = init_states['slz']
    
    # 提取参数
    tt = params['tt']
    cfmax = params['cfmax']
    cfr = params['cfr']
    cwh = params['cwh']
    fc = params['fc']
    beta = params['beta']
    lp = params['lp']
    betaet = params['betaet']
    c_par = params['c_par']
    perc = params['perc']
    k0 = params['k0']
    k1 = params['k1']
    k2 = params['k2']
    uzl = params['uzl']
    
    # 存储输出
    q_series = []
    
    # 时间循环
    for t in range(T):
        p_t = precip[t]
        temp_t = temp[t]
        pet_t = pet[t]
        
        snow, melt, sm, suz, slz, q = hbv_step_triton(
            p_t, temp_t, pet_t,
            snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet,
            c_par, perc, k0, k1, k2, uzl,
            nearzero,
        )
        
        q_series.append(q)
    
    # 堆叠输出
    q_series = torch.stack(q_series, dim=0)
    
    final_states = {
        'snow': snow,
        'melt': melt,
        'sm': sm,
        'suz': suz,
        'slz': slz,
    }
    
    return q_series, final_states
