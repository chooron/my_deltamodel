"""
HBV hydrological model — Triton implementation with hand-written backward.
This file mirrors the manually derived gradients in hbv_manual_core.py but runs
both forward and backward with Triton kernels. Each physical block (snow, soil,
routing) is exposed as an individual autograd.Function to aid debugging.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# =============================
# Snow block (forward/backward)
# =============================


@triton.jit
def _snow_forward_kernel(
    p_ptr, t_ptr, snow_ptr, melt_ptr, tt_ptr, cfmax_ptr, cfr_ptr, cwh_ptr,
    snow_out_ptr, melt_out_ptr, tosoil_ptr, rain_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    t_val = tl.load(t_ptr + offs, mask=mask, other=0.0)
    snow = tl.load(snow_ptr + offs, mask=mask, other=0.0)
    melt = tl.load(melt_ptr + offs, mask=mask, other=0.0)
    tt = tl.load(tt_ptr + offs, mask=mask, other=0.0)
    cfmax = tl.load(cfmax_ptr + offs, mask=mask, other=0.0)
    cfr = tl.load(cfr_ptr + offs, mask=mask, other=0.0)
    cwh = tl.load(cwh_ptr + offs, mask=mask, other=0.0)

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
    snow_st3 = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt

    arg_tosoil = melt_st2 - cwh * snow_st3
    mask_tosoil = arg_tosoil > 0.0
    tosoil = tl.where(mask_tosoil, arg_tosoil, 0.0)
    melt_out = melt_st2 - tosoil

    tl.store(snow_out_ptr + offs, snow_st3, mask=mask)
    tl.store(melt_out_ptr + offs, melt_out, mask=mask)
    tl.store(tosoil_ptr + offs, tosoil, mask=mask)
    tl.store(rain_ptr + offs, rain, mask=mask)


@triton.jit
def _snow_backward_kernel(
    p_ptr, t_ptr, snow_ptr, melt_ptr, tt_ptr, cfmax_ptr, cfr_ptr, cwh_ptr,
    g_snow_out_ptr, g_melt_out_ptr, g_tosoil_ptr, g_rain_ptr,
    gp_ptr, gt_ptr, gsnow_ptr, gmelt_ptr, gtt_ptr, gcfmax_ptr, gcfr_ptr, gcwh_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load forward inputs
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    t_val = tl.load(t_ptr + offs, mask=mask, other=0.0)
    snow = tl.load(snow_ptr + offs, mask=mask, other=0.0)
    melt = tl.load(melt_ptr + offs, mask=mask, other=0.0)
    tt = tl.load(tt_ptr + offs, mask=mask, other=0.0)
    cfmax = tl.load(cfmax_ptr + offs, mask=mask, other=0.0)
    cfr = tl.load(cfr_ptr + offs, mask=mask, other=0.0)
    cwh = tl.load(cwh_ptr + offs, mask=mask, other=0.0)

    g_snow_out = tl.load(g_snow_out_ptr + offs, mask=mask, other=0.0)
    g_melt_out = tl.load(g_melt_out_ptr + offs, mask=mask, other=0.0)
    g_tosoil = tl.load(g_tosoil_ptr + offs, mask=mask, other=0.0)
    g_rain = tl.load(g_rain_ptr + offs, mask=mask, other=0.0)

    # Recompute forward
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
    snow_st3 = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt

    arg_tosoil = melt_st2 - cwh * snow_st3
    mask_tosoil = arg_tosoil > 0.0
    tosoil = tl.where(mask_tosoil, arg_tosoil, 0.0)

    # Init grads
    g_melt_st2 = g_melt_out
    g_tosoil_tot = g_tosoil - g_melt_out
    g_snow_st3 = g_snow_out
    g_temp_diff = tl.zeros_like(temp_diff)

    # tosoil
    g_arg = tl.where(mask_tosoil, g_tosoil_tot, 0.0)
    g_melt_st2 += g_arg
    g_snow_st3 += g_arg * (-cwh)
    gcwh = g_arg * (-snow_st3)

    # melt_st2 = melt_st1 - refreeze
    g_melt_st1 = g_melt_st2
    g_refreeze = -g_melt_st2 + g_snow_st3

    # snow_st3 = snow_st2 + refreeze
    g_snow_st2 = g_snow_st3

    # refreeze = min(pot_refreeze, melt_st1)
    g_pot_ref = tl.where(mask_refreeze, g_refreeze, 0.0)
    g_melt_st1 += tl.where(mask_refreeze, 0.0, g_refreeze)

    mask_cold = temp_diff < 0.0
    relu_neg = tl.maximum(-temp_diff, 0.0)
    gcfr = g_pot_ref * cfmax * relu_neg
    gcfmax_ref = g_pot_ref * cfr * relu_neg
    g_temp_diff += g_pot_ref * cfr * cfmax * (-1.0) * mask_cold

    # melt_st1 = melt + melt_amount
    gmelt = g_melt_st1
    g_melt_amt = g_melt_st1

    # snow_st2 = snow_st1 - melt_amount
    g_snow_st1 = g_snow_st2
    g_melt_amt += -g_snow_st2

    # melt_amount = min(pot_melt, snow_st1)
    g_pot_melt = tl.where(mask_melt, g_melt_amt, 0.0)
    g_snow_st1 += tl.where(mask_melt, 0.0, g_melt_amt)

    mask_warm = temp_diff > 0.0
    relu_pos = tl.maximum(temp_diff, 0.0)
    gcfmax = g_pot_melt * relu_pos + gcfmax_ref
    g_temp_diff += g_pot_melt * cfmax * mask_warm

    gsnow = g_snow_st1
    g_snow_input_tot = g_snow_st1

    gp = tl.where(is_rain, g_rain, g_snow_input_tot)

    gt = g_temp_diff
    gtt = -g_temp_diff

    tl.store(gp_ptr + offs, gp, mask=mask)
    tl.store(gt_ptr + offs, gt, mask=mask)
    tl.store(gsnow_ptr + offs, gsnow, mask=mask)
    tl.store(gmelt_ptr + offs, gmelt, mask=mask)
    tl.store(gtt_ptr + offs, gtt, mask=mask)
    tl.store(gcfmax_ptr + offs, gcfmax, mask=mask)
    tl.store(gcfr_ptr + offs, gcfr, mask=mask)
    tl.store(gcwh_ptr + offs, gcwh, mask=mask)


class SnowBlockTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, t_val, snow, melt, tt, cfmax, cfr, cwh):
        n = p.numel()
        snow_out = torch.empty_like(snow)
        melt_out = torch.empty_like(melt)
        tosoil = torch.empty_like(p)
        rain = torch.empty_like(p)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _snow_forward_kernel[grid](
            p, t_val, snow, melt, tt, cfmax, cfr, cwh,
            snow_out, melt_out, tosoil, rain,
            n,
            BLOCK_SIZE=256,
        )
        ctx.save_for_backward(p, t_val, snow, melt, tt, cfmax, cfr, cwh)
        ctx.n = n
        return snow_out, melt_out, tosoil, rain

    @staticmethod
    def backward(ctx, g_snow_out, g_melt_out, g_tosoil, g_rain):
        p, t_val, snow, melt, tt, cfmax, cfr, cwh = ctx.saved_tensors
        n = ctx.n
        gp = torch.empty_like(p)
        gt = torch.empty_like(t_val)
        gsnow = torch.empty_like(snow)
        gmelt = torch.empty_like(melt)
        gtt = torch.empty_like(tt)
        gcfmax = torch.empty_like(cfmax)
        gcfr = torch.empty_like(cfr)
        gcwh = torch.empty_like(cwh)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _snow_backward_kernel[grid](
            p, t_val, snow, melt, tt, cfmax, cfr, cwh,
            g_snow_out, g_melt_out, g_tosoil, g_rain,
            gp, gt, gsnow, gmelt, gtt, gcfmax, gcfr, gcwh,
            n,
            BLOCK_SIZE=256,
        )
        return gp, gt, gsnow, gmelt, gtt, gcfmax, gcfr, gcwh


# =============================
# Soil block (forward/backward)
# =============================


@triton.jit
def _soil_forward_kernel(
    sm_ptr, slz_ptr, rain_ptr, tosoil_ptr, pet_ptr,
    fc_ptr, beta_ptr, lp_ptr, betaet_ptr, c_ptr,
    sm_out_ptr, slz_out_ptr, recharge_ptr, excess_ptr,
    soil_wet_ptr, evapfactor_ptr, capillary_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    eps = 1e-6

    sm = tl.load(sm_ptr + offs, mask=mask, other=0.0)
    slz = tl.load(slz_ptr + offs, mask=mask, other=0.0)
    rain = tl.load(rain_ptr + offs, mask=mask, other=0.0)
    tosoil = tl.load(tosoil_ptr + offs, mask=mask, other=0.0)
    pet = tl.load(pet_ptr + offs, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    lp = tl.load(lp_ptr + offs, mask=mask, other=0.0)
    betaet = tl.load(betaet_ptr + offs, mask=mask, other=0.0)
    c_par = tl.load(c_ptr + offs, mask=mask, other=0.0)

    # Triton does not support Python's ** on tensors; use tl.exp/tl.log for pow
    soil_ratio = sm / fc
    soil_wet = tl.minimum(tl.maximum(tl.exp(beta * tl.log(tl.maximum(soil_ratio, eps))), 0.0), 1.0)
    recharge = (rain + tosoil) * soil_wet

    sm_st1 = sm + rain + tosoil - recharge
    excess = tl.maximum(sm_st1 - fc, 0.0)
    sm_st2 = sm_st1 - excess

    evapfactor = sm_st2 / (lp * fc)
    evapfactor = tl.minimum(tl.maximum(evapfactor, 0.0), 1.0)
    evapfactor = tl.minimum(tl.maximum(tl.exp(betaet * tl.log(tl.maximum(evapfactor, eps))), 0.0), 1.0)

    etact = tl.minimum(pet * evapfactor, sm_st2)
    sm_after_evap = tl.maximum(sm_st2 - etact, eps)

    sm_ratio = sm_after_evap / fc
    sm_ratio = tl.minimum(sm_ratio, 1.0)
    capillary = tl.minimum(slz, c_par * slz * (1.0 - sm_ratio))
    sm_out = tl.maximum(sm_after_evap + capillary, eps)
    slz_out = tl.maximum(slz - capillary, eps)

    tl.store(sm_out_ptr + offs, sm_out, mask=mask)
    tl.store(slz_out_ptr + offs, slz_out, mask=mask)
    tl.store(recharge_ptr + offs, recharge, mask=mask)
    tl.store(excess_ptr + offs, excess, mask=mask)
    tl.store(soil_wet_ptr + offs, soil_wet, mask=mask)
    tl.store(evapfactor_ptr + offs, evapfactor, mask=mask)
    tl.store(capillary_ptr + offs, capillary, mask=mask)


@triton.jit
def _soil_backward_kernel(
    sm_ptr, slz_ptr, rain_ptr, tosoil_ptr, pet_ptr,
    fc_ptr, beta_ptr, lp_ptr, betaet_ptr, c_ptr,
    g_sm_out_ptr, g_slz_out_ptr, g_recharge_ptr, g_excess_ptr,
    g_soil_wet_ptr, g_evapfactor_ptr, g_capillary_ptr,
    g_sm_ptr, g_slz_ptr, g_rain_ptr, g_tosoil_ptr, g_pet_ptr,
    g_fc_ptr, g_beta_ptr, g_lp_ptr, g_betaet_ptr, g_c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    eps = 1e-6
    LEAK = 1e-4

    sm = tl.load(sm_ptr + offs, mask=mask, other=0.0)
    slz = tl.load(slz_ptr + offs, mask=mask, other=0.0)
    rain = tl.load(rain_ptr + offs, mask=mask, other=0.0)
    tosoil = tl.load(tosoil_ptr + offs, mask=mask, other=0.0)
    pet = tl.load(pet_ptr + offs, mask=mask, other=0.0)
    fc = tl.load(fc_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    lp = tl.load(lp_ptr + offs, mask=mask, other=0.0)
    betaet = tl.load(betaet_ptr + offs, mask=mask, other=0.0)
    c_par = tl.load(c_ptr + offs, mask=mask, other=0.0)

    g_sm_out = tl.load(g_sm_out_ptr + offs, mask=mask, other=0.0)
    g_slz_out = tl.load(g_slz_out_ptr + offs, mask=mask, other=0.0)
    g_recharge = tl.load(g_recharge_ptr + offs, mask=mask, other=0.0)
    g_excess = tl.load(g_excess_ptr + offs, mask=mask, other=0.0)
    g_soil_wet = tl.load(g_soil_wet_ptr + offs, mask=mask, other=0.0)
    g_evapfactor = tl.load(g_evapfactor_ptr + offs, mask=mask, other=0.0)
    g_capillary = tl.load(g_capillary_ptr + offs, mask=mask, other=0.0)

    soil_ratio = sm / fc
    soil_wet = tl.minimum(tl.maximum(tl.exp(beta * tl.log(tl.maximum(soil_ratio, eps))), 0.0), 1.0)
    recharge = (rain + tosoil) * soil_wet
    sm_st1 = sm + rain + tosoil - recharge
    excess = tl.maximum(sm_st1 - fc, 0.0)
    mask_excess = sm_st1 > fc
    sm_st2 = sm_st1 - excess

    ef1 = sm_st2 / (lp * fc)
    mask_ef1 = (ef1 > 0.0) & (ef1 < 1.0)
    ef1 = tl.minimum(tl.maximum(ef1, 0.0), 1.0)
    ef1_base = tl.maximum(ef1, eps)
    ef2 = tl.exp(betaet * tl.log(ef1_base))
    evapfactor = tl.minimum(tl.maximum(ef2, 0.0), 1.0)
    mask_ef2 = (evapfactor > 0.0) & (evapfactor < 1.0)

    pet_prod = pet * evapfactor
    mask_et = pet_prod < sm_st2
    etact = tl.where(mask_et, pet_prod, sm_st2)
    sm_after_evap = sm_st2 - etact

    sm_ratio = sm_after_evap / fc
    sm_ratio = tl.minimum(sm_ratio, 1.0)
    mask_smratio = sm_ratio < 1.0
    cap_expr = c_par * slz * (1.0 - sm_ratio)
    capillary = tl.minimum(slz, cap_expr)
    sm_out = sm_after_evap + capillary
    slz_out = slz - capillary

    mask_slz_out = slz_out > eps
    g_slz_out_mask = g_slz_out * mask_slz_out

    g_cap_total = g_capillary + g_sm_out - g_slz_out_mask
    g_sm_after_evap = g_sm_out
    g_slz = g_slz_out_mask

    mask_cap_expr = cap_expr < slz
    g_cap_expr = tl.where(mask_cap_expr, g_cap_total, 0.0)
    g_slz += tl.where(mask_cap_expr, 0.0, g_cap_total)
    g_slz += g_cap_expr * c_par * (1.0 - sm_ratio)
    g_c_par = g_cap_expr * slz * (1.0 - sm_ratio)
    g_sm_after_evap += g_cap_expr * (-c_par * slz * (1.0 / fc)) * mask_smratio
    g_fc_cap = g_cap_expr * (c_par * slz * (sm_after_evap / (fc * fc))) * mask_smratio

    mask_sm3 = sm_out > eps
    g_sm_st2 = tl.where(mask_sm3, g_sm_after_evap, 0.0)
    g_etact = tl.where(mask_sm3, -g_sm_after_evap, 0.0)

    g_pet_prod = tl.where(mask_et, g_etact, g_etact * LEAK)
    g_sm_st2 += tl.where(mask_et, 0.0, g_etact)
    g_evapfactor_tot = g_evapfactor + g_pet_prod * pet
    g_pet = g_pet_prod * evapfactor

    g_ef2 = tl.where(mask_ef2, g_evapfactor_tot, 0.0)
    log_ef1 = tl.log(ef1_base)
    g_betaet = g_ef2 * ef2 * log_ef1
    g_ef1 = g_ef2 * betaet * tl.exp((betaet - 1.0) * log_ef1) * mask_ef1

    denom = lp * fc
    g_sm_st2 += g_ef1 * (1.0 / denom)
    g_lp = g_ef1 * (-sm_st2 / (denom * lp))
    g_fc = g_ef1 * (-sm_st2 / (denom * fc)) + g_fc_cap

    g_excess_tot = g_excess - g_sm_st2
    g_sm_st1 = g_sm_st2 + tl.where(mask_excess, g_excess_tot, g_excess_tot * LEAK)
    g_fc += tl.where(mask_excess, -g_excess_tot, -g_excess_tot * LEAK)

    g_recharge_tot = g_recharge + (-g_sm_st1)
    g_soil_wet_tot = g_soil_wet + g_recharge_tot * (rain + tosoil)
    g_rain = g_sm_st1 + g_recharge_tot * soil_wet
    g_tosoil = g_sm_st1 + g_recharge_tot * soil_wet

    sw_base = sm / fc
    safe_sw = tl.maximum(sw_base, eps)
    sw_pow = tl.exp(beta * tl.log(safe_sw))
    mask_sw = (sw_pow > 0.0) & (sw_pow < 1.0)
    g_sw_pow = tl.where(mask_sw, g_soil_wet_tot, g_soil_wet_tot * LEAK)

    g_beta = g_sw_pow * sw_pow * tl.log(safe_sw)
    pow_sw = tl.exp((beta - 1.0) * tl.log(safe_sw))
    g_sm = g_sw_pow * beta * pow_sw * (1.0 / fc)
    g_fc += g_sw_pow * beta * pow_sw * (-sm / (fc * fc))
    g_sm += g_sm_st1

    tl.store(g_sm_ptr + offs, g_sm, mask=mask)
    tl.store(g_slz_ptr + offs, g_slz, mask=mask)
    tl.store(g_rain_ptr + offs, g_rain, mask=mask)
    tl.store(g_tosoil_ptr + offs, g_tosoil, mask=mask)
    tl.store(g_pet_ptr + offs, g_pet, mask=mask)
    tl.store(g_fc_ptr + offs, g_fc, mask=mask)
    tl.store(g_beta_ptr + offs, g_beta, mask=mask)
    tl.store(g_lp_ptr + offs, g_lp, mask=mask)
    tl.store(g_betaet_ptr + offs, g_betaet, mask=mask)
    tl.store(g_c_ptr + offs, g_c_par, mask=mask)


class SoilBlockTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par):
        n = sm.numel()
        sm_out = torch.empty_like(sm)
        slz_out = torch.empty_like(slz)
        recharge = torch.empty_like(sm)
        excess = torch.empty_like(sm)
        soil_wet = torch.empty_like(sm)
        evapfactor = torch.empty_like(sm)
        capillary = torch.empty_like(sm)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _soil_forward_kernel[grid](
            sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par,
            sm_out, slz_out, recharge, excess, soil_wet, evapfactor, capillary,
            n,
            BLOCK_SIZE=256,
        )
        ctx.save_for_backward(sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par)
        ctx.n = n
        return sm_out, slz_out, recharge, excess, soil_wet, evapfactor, capillary

    @staticmethod
    def backward(ctx, g_sm_out, g_slz_out, g_recharge, g_excess, g_soil_wet, g_evapfactor, g_capillary):
        sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par = ctx.saved_tensors
        n = ctx.n
        g_sm = torch.empty_like(sm)
        g_slz = torch.empty_like(slz)
        g_rain = torch.empty_like(rain)
        g_tosoil = torch.empty_like(tosoil)
        g_pet = torch.empty_like(pet)
        g_fc = torch.empty_like(fc)
        g_beta = torch.empty_like(beta)
        g_lp = torch.empty_like(lp)
        g_betaet = torch.empty_like(betaet)
        g_c = torch.empty_like(c_par)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _soil_backward_kernel[grid](
            sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par,
            g_sm_out, g_slz_out, g_recharge, g_excess, g_soil_wet, g_evapfactor, g_capillary,
            g_sm, g_slz, g_rain, g_tosoil, g_pet, g_fc, g_beta, g_lp, g_betaet, g_c,
            n,
            BLOCK_SIZE=256,
        )
        return g_sm, g_slz, g_rain, g_tosoil, g_pet, g_fc, g_beta, g_lp, g_betaet, g_c


# ================================
# Routing block (forward/backward)
# ================================


@triton.jit
def _routing_forward_kernel(
    sm_after_ptr, suz_ptr, slz_ptr, recharge_ptr, excess_ptr,
    perc_ptr, k0_ptr, k1_ptr, k2_ptr, uzl_ptr,
    sm_out_ptr, suz_out_ptr, slz_out_ptr, q_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    sm_after = tl.load(sm_after_ptr + offs, mask=mask, other=0.0)
    suz = tl.load(suz_ptr + offs, mask=mask, other=0.0)
    slz = tl.load(slz_ptr + offs, mask=mask, other=0.0)
    recharge = tl.load(recharge_ptr + offs, mask=mask, other=0.0)
    excess = tl.load(excess_ptr + offs, mask=mask, other=0.0)
    perc = tl.load(perc_ptr + offs, mask=mask, other=0.0)
    k0 = tl.load(k0_ptr + offs, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offs, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offs, mask=mask, other=0.0)
    uzl = tl.load(uzl_ptr + offs, mask=mask, other=0.0)

    suz_st1 = suz + recharge + excess
    perc_flux = tl.minimum(suz_st1, perc)
    suz_st2 = suz_st1 - perc_flux
    slz_st1 = slz + perc_flux

    q0 = k0 * tl.maximum(suz_st2 - uzl, 0.0)
    suz_st3 = suz_st2 - q0
    q1 = k1 * suz_st3
    suz_out = suz_st3 - q1

    q2 = k2 * slz_st1
    slz_out = slz_st1 - q2

    q_total = q0 + q1 + q2

    tl.store(sm_out_ptr + offs, sm_after, mask=mask)
    tl.store(suz_out_ptr + offs, suz_out, mask=mask)
    tl.store(slz_out_ptr + offs, slz_out, mask=mask)
    tl.store(q_out_ptr + offs, q_total, mask=mask)


@triton.jit
def _routing_backward_kernel(
    sm_after_ptr, suz_ptr, slz_ptr, recharge_ptr, excess_ptr,
    perc_ptr, k0_ptr, k1_ptr, k2_ptr, uzl_ptr,
    g_sm_after_ptr, g_suz_out_ptr, g_slz_out_ptr, g_q_total_ptr,
    g_sm_ptr, g_suz_ptr, g_slz_ptr, g_recharge_ptr, g_excess_ptr,
    g_perc_ptr, g_k0_ptr, g_k1_ptr, g_k2_ptr, g_uzl_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    sm_after = tl.load(sm_after_ptr + offs, mask=mask, other=0.0)
    suz = tl.load(suz_ptr + offs, mask=mask, other=0.0)
    slz = tl.load(slz_ptr + offs, mask=mask, other=0.0)
    recharge = tl.load(recharge_ptr + offs, mask=mask, other=0.0)
    excess = tl.load(excess_ptr + offs, mask=mask, other=0.0)
    perc = tl.load(perc_ptr + offs, mask=mask, other=0.0)
    k0 = tl.load(k0_ptr + offs, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offs, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offs, mask=mask, other=0.0)
    uzl = tl.load(uzl_ptr + offs, mask=mask, other=0.0)

    g_sm_after = tl.load(g_sm_after_ptr + offs, mask=mask, other=0.0)
    g_suz_out = tl.load(g_suz_out_ptr + offs, mask=mask, other=0.0)
    g_slz_out = tl.load(g_slz_out_ptr + offs, mask=mask, other=0.0)
    g_q_total = tl.load(g_q_total_ptr + offs, mask=mask, other=0.0)

    suz_st1 = suz + recharge + excess
    perc_flux = tl.minimum(suz_st1, perc)
    mask_perc = suz_st1 < perc
    suz_st2 = suz_st1 - perc_flux
    slz_st1 = slz + perc_flux

    q0_arg = suz_st2 - uzl
    mask_q0 = q0_arg > 0.0
    q0 = k0 * tl.maximum(q0_arg, 0.0)
    suz_st3 = suz_st2 - q0
    q1 = k1 * suz_st3
    suz_out = suz_st3 - q1

    q2 = k2 * slz_st1
    slz_out = slz_st1 - q2

    g_q0 = g_q_total
    g_q1 = g_q_total
    g_q2 = g_q_total

    g_suz_st3 = g_suz_out
    g_q1 = g_q1 - g_suz_out

    g_k1 = g_q1 * suz_st3
    g_suz_st3 += g_q1 * k1

    g_suz_st2 = g_suz_st3
    g_q0 = g_q0 - g_suz_st3

    g_k0 = g_q0 * tl.maximum(q0_arg, 0.0)
    g_suz_st2 += g_q0 * k0 * mask_q0
    g_uzl = -g_q0 * k0 * mask_q0

    g_slz_st1 = g_slz_out
    g_q2 = g_q2 - g_slz_out

    g_k2 = g_q2 * slz_st1
    g_slz_st1 += g_q2 * k2

    g_suz_st2_eff = tl.where(mask_perc, 0.0, g_suz_st2)
    g_perc_flux = g_slz_st1 - g_suz_st2_eff
    g_suz_st1 = tl.where(mask_perc, g_perc_flux, g_suz_st2_eff)
    g_perc = tl.where(mask_perc, 0.0, g_perc_flux)

    g_suz = g_suz_st1
    g_recharge = g_suz_st1
    g_excess = g_suz_st1
    g_slz = g_slz_st1
    g_sm = g_sm_after

    tl.store(g_sm_ptr + offs, g_sm, mask=mask)
    tl.store(g_suz_ptr + offs, g_suz, mask=mask)
    tl.store(g_slz_ptr + offs, g_slz, mask=mask)
    tl.store(g_recharge_ptr + offs, g_recharge, mask=mask)
    tl.store(g_excess_ptr + offs, g_excess, mask=mask)
    tl.store(g_perc_ptr + offs, g_perc, mask=mask)
    tl.store(g_k0_ptr + offs, g_k0, mask=mask)
    tl.store(g_k1_ptr + offs, g_k1, mask=mask)
    tl.store(g_k2_ptr + offs, g_k2, mask=mask)
    tl.store(g_uzl_ptr + offs, g_uzl, mask=mask)


class RoutingBlockTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl):
        n = sm_after_evap.numel()
        sm_out = torch.empty_like(sm_after_evap)
        suz_out = torch.empty_like(suz)
        slz_out = torch.empty_like(slz)
        q_out = torch.empty_like(sm_after_evap)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _routing_forward_kernel[grid](
            sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl,
            sm_out, suz_out, slz_out, q_out,
            n,
            BLOCK_SIZE=256,
        )
        ctx.save_for_backward(sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl)
        ctx.n = n
        return sm_out, suz_out, slz_out, q_out

    @staticmethod
    def backward(ctx, g_sm_out, g_suz_out, g_slz_out, g_q_out):
        sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl = ctx.saved_tensors
        n = ctx.n
        g_sm = torch.empty_like(sm_after_evap)
        g_suz = torch.empty_like(suz)
        g_slz = torch.empty_like(slz)
        g_recharge = torch.empty_like(recharge)
        g_excess = torch.empty_like(excess)
        g_perc = torch.empty_like(perc)
        g_k0 = torch.empty_like(k0)
        g_k1 = torch.empty_like(k1)
        g_k2 = torch.empty_like(k2)
        g_uzl = torch.empty_like(uzl)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _routing_backward_kernel[grid](
            sm_after_evap, suz, slz, recharge, excess,
            perc, k0, k1, k2, uzl,
            g_sm_out, g_suz_out, g_slz_out, g_q_out,
            g_sm, g_suz, g_slz, g_recharge, g_excess,
            g_perc, g_k0, g_k1, g_k2, g_uzl,
            n,
            BLOCK_SIZE=256,
        )
        return g_sm, g_suz, g_slz, g_recharge, g_excess, g_perc, g_k0, g_k1, g_k2, g_uzl


# ==================
# High-level helpers
# ==================


def hbv_step_triton(
    p: torch.Tensor, t_val: torch.Tensor, pet: torch.Tensor,
    snow: torch.Tensor, melt: torch.Tensor, sm: torch.Tensor,
    suz: torch.Tensor, slz: torch.Tensor,
    tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor, cwh: torch.Tensor,
    fc: torch.Tensor, beta: torch.Tensor, lp: torch.Tensor, betaet: torch.Tensor, c_par: torch.Tensor,
    perc: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, uzl: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    snow_out, melt_out, tosoil, rain = SnowBlockTriton.apply(p, t_val, snow, melt, tt, cfmax, cfr, cwh)
    sm_out, slz_after_cap, recharge, excess, _, _, _ = SoilBlockTriton.apply(sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par)
    sm_after, suz_out, slz_out, q_out = RoutingBlockTriton.apply(
        sm_out, suz, slz_after_cap, recharge, excess, perc, k0, k1, k2, uzl
    )
    return snow_out, melt_out, sm_after, suz_out, slz_out, q_out


def hbv_run_triton(
    precip: torch.Tensor,
    temp: torch.Tensor,
    pet: torch.Tensor,
    params: dict,
    init_states: Optional[dict] = None,
) -> Tuple[torch.Tensor, dict]:
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
        snow, melt, sm, suz, slz, q = hbv_step_triton(
            precip[t], temp[t], pet[t], snow, melt, sm, suz, slz,
            tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par, perc, k0, k1, k2, uzl,
        )
        q_series.append(q)

    q_series = torch.stack(q_series, dim=0)
    final_states = {"snow": snow, "melt": melt, "sm": sm, "suz": suz, "slz": slz}
    return q_series, final_states


def run_block_grad_tests(device: Optional[str] = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.manual_seed(0)

    def _check(fn, inputs, name: str):
        test_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]
        ok = torch.autograd.gradcheck(lambda *args: fn.apply(*args), test_inputs, eps=1e-4, atol=1e-6, rtol=1e-6)
        print(f"[gradcheck] {name}: {ok}")

    p = torch.rand(8, device=device, dtype=dtype)
    t_val = torch.rand(8, device=device, dtype=dtype) * 4 + 1.0
    snow = torch.rand(8, device=device, dtype=dtype) * 10
    melt = torch.rand(8, device=device, dtype=dtype) * 2
    tt = torch.zeros(8, device=device, dtype=dtype)
    cfmax = torch.ones(8, device=device, dtype=dtype) * 2.0
    cfr = torch.ones(8, device=device, dtype=dtype) * 0.05
    cwh = torch.ones(8, device=device, dtype=dtype) * 0.1
    _check(SnowBlockTriton, [p, t_val, snow, melt, tt, cfmax, cfr, cwh], "snow")

    sm = torch.rand(8, device=device, dtype=dtype) * 60 + 20.0
    slz = torch.rand(8, device=device, dtype=dtype) * 30 + 10.0
    rain = torch.rand(8, device=device, dtype=dtype)
    tosoil = torch.rand(8, device=device, dtype=dtype)
    pet = torch.rand(8, device=device, dtype=dtype)
    fc = torch.ones(8, device=device, dtype=dtype) * 150.0
    beta = torch.ones(8, device=device, dtype=dtype) * 1.5
    lp = torch.ones(8, device=device, dtype=dtype) * 0.7
    betaet = torch.ones(8, device=device, dtype=dtype) * 1.2
    c_par = torch.ones(8, device=device, dtype=dtype) * 0.05
    _check(SoilBlockTriton, [sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par], "soil")

    sm_after_evap = torch.rand(8, device=device, dtype=dtype) * 50 + 1.0
    suz = torch.rand(8, device=device, dtype=dtype) * 30 + 1.0
    slz = torch.rand(8, device=device, dtype=dtype) * 30 + 1.0
    recharge = torch.rand(8, device=device, dtype=dtype)
    excess = torch.rand(8, device=device, dtype=dtype)
    perc = torch.ones(8, device=device, dtype=dtype) * 3.0
    k0 = torch.ones(8, device=device, dtype=dtype) * 0.25
    k1 = torch.ones(8, device=device, dtype=dtype) * 0.05
    k2 = torch.ones(8, device=device, dtype=dtype) * 0.01
    uzl = torch.ones(8, device=device, dtype=dtype) * 5.0
    _check(RoutingBlockTriton, [sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl], "routing")


if __name__ == "__main__":
    run_block_grad_tests()