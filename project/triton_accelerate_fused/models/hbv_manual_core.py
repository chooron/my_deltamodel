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
# 模块化公式（Torch 端）用于逐块梯度校验
# ==========================================


def _snow_block_formula(p: torch.Tensor, t_val: torch.Tensor, snow: torch.Tensor, melt: torch.Tensor,
                        tt: torch.Tensor, cfmax: torch.Tensor, cfr: torch.Tensor, cwh: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """前向：降水分离 + 积雪/融雪/再冻结 + 融雪入土 (纯 Torch 版本)。"""
    eps = 1e-6
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
    snow_st3 = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt

    tosoil = torch.clamp(melt_st2 - cwh * snow_st3, min=0.0)
    melt_out = melt_st2 - tosoil

    return snow_st3, melt_out, tosoil, rain, snow_input, temp_diff


class SnowBlock(torch.autograd.Function):
    """自定义 Autograd：只覆盖前向与手工反向测试接口。"""

    @staticmethod
    def forward(ctx, p, t_val, snow, melt, tt, cfmax, cfr, cwh):
        outputs = _snow_block_formula(p, t_val, snow, melt, tt, cfmax, cfr, cwh)
        ctx.save_for_backward(p, t_val, snow, melt, tt, cfmax, cfr, cwh)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        (p, t_val, snow, melt, tt, cfmax, cfr, cwh) = ctx.saved_tensors
        (g_snow3, g_melt_out, g_tosoil, g_rain, g_snow_input, g_temp_diff_out) = grad_outputs

        # 初始化梯度
        gp = torch.zeros_like(p)
        gt = torch.zeros_like(t_val)
        gsnow = torch.zeros_like(snow)
        gmelt = torch.zeros_like(melt)
        gtt = torch.zeros_like(tt)
        gcfmax = torch.zeros_like(cfmax)
        gcfr = torch.zeros_like(cfr)
        gcwh = torch.zeros_like(cwh)

        with torch.no_grad():
            temp_diff = t_val - tt
            is_rain = temp_diff > 0.0
            rain = torch.where(is_rain, p, torch.zeros_like(p))
            snow_input = torch.where(is_rain, torch.zeros_like(p), p)

            snow_st1 = snow + snow_input
            pot_melt = cfmax * torch.clamp(temp_diff, min=0.0)
            mask_melt = pot_melt < snow_st1
            melt_amount = torch.where(mask_melt, pot_melt, snow_st1)
            snow_st2 = snow_st1 - melt_amount
            melt_st1 = melt + melt_amount

            pot_refreeze = cfr * cfmax * torch.clamp(-temp_diff, min=0.0)
            mask_refreeze = pot_refreeze < melt_st1
            refreeze_amt = torch.where(mask_refreeze, pot_refreeze, melt_st1)
            snow_st3 = snow_st2 + refreeze_amt
            melt_st2 = melt_st1 - refreeze_amt

            arg_tosoil = melt_st2 - cwh * snow_st3
            mask_tosoil = arg_tosoil > 0.0
            tosoil = torch.where(mask_tosoil, arg_tosoil, torch.zeros_like(arg_tosoil))
            melt_out = melt_st2 - tosoil

            # 来自输出的初始梯度
            g_melt_st2 = torch.zeros_like(g_melt_out)
            g_tosoil_tot = g_tosoil.clone()
            g_snow_st3 = g_snow3.clone()
            g_temp_diff = g_temp_diff_out.clone()

            # melt_out = melt_st2 - tosoil
            g_melt_st2 += g_melt_out
            g_tosoil_tot += -g_melt_out

            # tosoil = relu(arg)
            g_arg = torch.where(mask_tosoil, g_tosoil_tot, torch.zeros_like(g_tosoil_tot))
            g_melt_st2 += g_arg
            g_snow_st3 += g_arg * (-cwh)
            gcwh += g_arg * (-snow_st3)

            # melt_st2 = melt_st1 - refreeze_amt
            g_melt_st1 = g_melt_st2.clone()
            g_refreeze = -g_melt_st2

            # snow_st3 = snow_st2 + refreeze_amt
            g_snow_st2 = g_snow_st3.clone()
            g_refreeze += g_snow_st3

            # refreeze = min(pot_refreeze, melt_st1)
            g_pot_ref = torch.where(mask_refreeze, g_refreeze, torch.zeros_like(g_refreeze))
            g_melt_st1 += torch.where(mask_refreeze, torch.zeros_like(g_refreeze), g_refreeze)

            # pot_refreeze = cfr * cfmax * relu(-temp_diff)
            mask_cold = temp_diff < 0.0
            relu_neg = torch.clamp(-temp_diff, min=0.0)
            gcfr += g_pot_ref * cfmax * relu_neg
            gcfmax += g_pot_ref * cfr * relu_neg
            g_temp_diff += g_pot_ref * cfr * cfmax * (-1.0) * mask_cold

            # melt_st1 = melt + melt_amount
            gmelt += g_melt_st1
            g_melt_amt = g_melt_st1.clone()

            # snow_st2 = snow_st1 - melt_amount
            g_snow_st1 = g_snow_st2.clone()
            g_melt_amt += -g_snow_st2

            # melt_amount = min(pot_melt, snow_st1)
            g_pot_melt = torch.where(mask_melt, g_melt_amt, torch.zeros_like(g_melt_amt))
            g_snow_st1 += torch.where(mask_melt, torch.zeros_like(g_melt_amt), g_melt_amt)

            # pot_melt = cfmax * relu(temp_diff)
            mask_warm = temp_diff > 0.0
            relu_pos = torch.clamp(temp_diff, min=0.0)
            gcfmax += g_pot_melt * relu_pos
            g_temp_diff += g_pot_melt * cfmax * mask_warm

            # snow_st1 = snow + snow_input
            gsnow += g_snow_st1
            g_snow_input_tot = g_snow_st1.clone()

            # rain & snow_input split
            # p 梯度: rain 路径 or snow_input 路径
            gp = torch.where(is_rain, g_rain, g_snow_input + g_snow_input_tot)

            # temp_diff = t_val - tt
            gt += g_temp_diff
            gtt += -g_temp_diff

        return (gp, gt, gsnow, gmelt, gtt, gcfmax, gcfr, gcwh)


def _soil_block_formula(sm: torch.Tensor, rain: torch.Tensor, tosoil: torch.Tensor, pet: torch.Tensor,
                        fc: torch.Tensor, beta: torch.Tensor, lp: torch.Tensor, betaet: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """前向：土壤水分补给/超渗 + 蒸散 (贴合引用逻辑)。"""
    eps = 1e-6

    soil_wetness = torch.clamp((sm / fc) ** beta, 0.0, 1.0)
    recharge = (rain + tosoil) * soil_wetness

    sm_st1 = sm + rain + tosoil - recharge
    excess = torch.clamp(sm_st1 - fc, min=0.0)
    sm_st2 = sm_st1 - excess

    evapfactor = sm_st2 / (lp * fc)
    evapfactor = torch.clamp(evapfactor, 0.0, 1.0)
    evapfactor = torch.clamp(evapfactor ** betaet, 0.0, 1.0)

    etact = torch.minimum(pet * evapfactor, sm_st2)
    sm_st3 = torch.clamp(sm_st2 - etact, min=eps)

    return sm_st3, recharge, excess, soil_wetness, evapfactor


class SoilBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sm, rain, tosoil, pet, fc, beta, lp, betaet):
        outputs = _soil_block_formula(sm, rain, tosoil, pet, fc, beta, lp, betaet)
        ctx.save_for_backward(sm, rain, tosoil, pet, fc, beta, lp, betaet)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        sm, rain, tosoil, pet, fc, beta, lp, betaet = ctx.saved_tensors
        (g_sm3, g_recharge, g_excess, g_soil_wetness, g_evapfactor) = grad_outputs

        g_sm = torch.zeros_like(sm)
        g_rain = torch.zeros_like(rain)
        g_tosoil = torch.zeros_like(tosoil)
        g_pet = torch.zeros_like(pet)
        g_fc = torch.zeros_like(fc)
        g_beta = torch.zeros_like(beta)
        g_lp = torch.zeros_like(lp)
        g_betaet = torch.zeros_like(betaet)

        eps = 1e-6

        with torch.no_grad():
            soil_wetness = torch.clamp((sm / fc) ** beta, 0.0, 1.0)
            recharge = (rain + tosoil) * soil_wetness

            sm_st1 = sm + rain + tosoil - recharge
            excess = torch.clamp(sm_st1 - fc, min=0.0)
            mask_excess = sm_st1 > fc
            sm_st2 = sm_st1 - excess

            ef1 = sm_st2 / (lp * fc)
            mask_ef1 = (ef1 > 0.0) & (ef1 < 1.0)
            ef1 = torch.clamp(ef1, 0.0, 1.0)
            ef2 = ef1 ** betaet
            ef2_base = ef1.clamp(min=eps)
            evapfactor = torch.clamp(ef2, 0.0, 1.0)
            mask_ef2 = (ef2 > 0.0) & (ef2 < 1.0)

            pet_prod = pet * evapfactor
            mask_et = pet_prod < sm_st2
            etact = torch.where(mask_et, pet_prod, sm_st2)
            sm_st3_pre = sm_st2 - etact
            mask_sm3 = sm_st3_pre > eps

            # sm_st3 = clamp(sm_st3_pre, eps)
            g_sm_st2 = torch.where(mask_sm3, g_sm3, torch.zeros_like(g_sm3))
            g_etact = torch.where(mask_sm3, -g_sm3, torch.zeros_like(g_sm3))

            # etact = min(pet*evapfactor, sm_st2)
            g_pet_prod = torch.where(mask_et, g_etact, torch.zeros_like(g_etact))
            g_sm_st2 += torch.where(mask_et, torch.zeros_like(g_etact), g_etact)
            g_evapfactor_tot = g_evapfactor + g_pet_prod * pet
            g_pet += g_pet_prod * evapfactor

            # evapfactor clamp on ef2
            g_ef2 = torch.where(mask_ef2, g_evapfactor_tot, torch.zeros_like(g_evapfactor_tot))

            # ef2 = ef1 ** betaet
            g_betaet += g_ef2 * torch.pow(ef2_base, betaet) * torch.log(ef2_base)
            pow_term = torch.pow(ef2_base, betaet - 1.0)
            g_ef1 = g_ef2 * betaet * pow_term * mask_ef1

            # ef1 = sm_st2 / (lp*fc)
            denom = lp * fc
            g_sm_st2 += g_ef1 * (1.0 / denom)
            g_lp += g_ef1 * (-sm_st2 / (denom * lp))
            g_fc += g_ef1 * (-sm_st2 / (denom * fc))

            # sm_st2 = sm_st1 - excess
            g_excess_tot = g_excess - g_sm_st2
            g_sm_st1 = g_sm_st2 + torch.where(mask_excess, g_excess_tot, torch.zeros_like(g_excess_tot))
            g_fc += torch.where(mask_excess, -g_excess_tot, torch.zeros_like(g_excess_tot))

            # recharge = (rain + tosoil) * soil_wetness
            g_recharge_tot = g_recharge + (-g_sm_st1)
            g_soil_wetness_tot = g_soil_wetness + g_recharge_tot * (rain + tosoil)
            g_rain += g_sm_st1 + g_recharge_tot * soil_wetness
            g_tosoil += g_sm_st1 + g_recharge_tot * soil_wetness

            # soil_wetness = clamp((sm/fc)**beta, 0,1)
            sw_base = (sm / fc)
            sw_pow = sw_base.clamp(min=eps) ** beta
            mask_sw = (sw_pow > 0.0) & (sw_pow < 1.0)
            g_sw_pow = torch.where(mask_sw, g_soil_wetness_tot, torch.zeros_like(g_soil_wetness_tot))

            g_beta += g_sw_pow * sw_pow * torch.log(sw_base.clamp(min=eps))
            pow_sw = sw_base.clamp(min=eps) ** (beta - 1.0)
            g_sm += g_sw_pow * beta * pow_sw * (1.0 / fc)
            g_fc += g_sw_pow * beta * pow_sw * (-sm / (fc * fc))

            # sm_st1 = sm + rain + tosoil - recharge
            g_sm += g_sm_st1

        return (g_sm, g_rain, g_tosoil, g_pet, g_fc, g_beta, g_lp, g_betaet)


def _routing_block_formula(sm_after_evap: torch.Tensor, suz: torch.Tensor, slz: torch.Tensor,
                           recharge: torch.Tensor, excess: torch.Tensor, perc: torch.Tensor,
                           k0: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, uzl: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """前向：地下水三箱汇流 (贴合引用逻辑，无毛管上升)。"""
    suz_st1 = suz + recharge + excess
    perc_flux = torch.minimum(suz_st1, perc)
    suz_st2 = suz_st1 - perc_flux
    slz_st1 = slz + perc_flux

    q0 = k0 * torch.clamp(suz_st2 - uzl, min=0.0)
    suz_st3 = suz_st2 - q0
    q1 = k1 * suz_st3
    suz_out = suz_st3 - q1

    q2 = k2 * slz_st1
    slz_out = slz_st1 - q2

    q_total = q0 + q1 + q2
    return sm_after_evap, suz_out, slz_out, q_total


class RoutingBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl):
        outputs = _routing_block_formula(sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl)
        ctx.save_for_backward(sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl = ctx.saved_tensors
        (g_sm_after, g_suz_out, g_slz_out, g_q_total) = grad_outputs

        g_sm = torch.zeros_like(sm_after_evap)
        g_suz = torch.zeros_like(suz)
        g_slz = torch.zeros_like(slz)
        g_recharge = torch.zeros_like(recharge)
        g_excess = torch.zeros_like(excess)
        g_perc = torch.zeros_like(perc)
        g_k0 = torch.zeros_like(k0)
        g_k1 = torch.zeros_like(k1)
        g_k2 = torch.zeros_like(k2)
        g_uzl = torch.zeros_like(uzl)

        with torch.no_grad():
            suz_st1 = suz + recharge + excess
            perc_flux = torch.minimum(suz_st1, perc)
            mask_perc = suz_st1 < perc
            suz_st2 = suz_st1 - perc_flux
            slz_st1 = slz + perc_flux

            q0_arg = suz_st2 - uzl
            mask_q0 = q0_arg > 0.0
            q0 = k0 * torch.clamp(q0_arg, min=0.0)
            suz_st3 = suz_st2 - q0
            q1 = k1 * suz_st3
            suz_out = suz_st3 - q1

            q2 = k2 * slz_st1
            slz_out = slz_st1 - q2

            # 初始梯度来自输出
            g_q0 = g_q_total.clone()
            g_q1 = g_q_total.clone()
            g_q2 = g_q_total.clone()

            # suz_out = suz_st3 - q1
            g_suz_st3 = g_suz_out.clone()
            g_q1 = g_q1 - g_suz_out

            # q1 = k1 * suz_st3
            g_k1 += g_q1 * suz_st3
            g_suz_st3 = g_suz_st3 + g_q1 * k1

            # suz_st3 = suz_st2 - q0
            g_suz_st2 = g_suz_st3.clone()
            g_q0 = g_q0 - g_suz_st3

            # q0 = k0 * relu(q0_arg)
            g_k0 += g_q0 * torch.clamp(q0_arg, min=0.0)
            g_suz_st2 += g_q0 * k0 * mask_q0
            g_uzl += -g_q0 * k0 * mask_q0

            # slz_out = slz_st1 - q2
            g_slz_st1 = g_slz_out.clone()
            g_q2 = g_q2 - g_slz_out

            # q2 = k2 * slz_st1
            g_k2 += g_q2 * slz_st1
            g_slz_st1 = g_slz_st1 + g_q2 * k2

            # perc_flux path
            g_suz_st2_eff = torch.where(mask_perc, torch.zeros_like(g_suz_st2), g_suz_st2)
            g_perc_flux = g_slz_st1 - g_suz_st2_eff
            g_suz_st1 = torch.where(mask_perc, g_perc_flux, g_suz_st2_eff)
            g_perc += torch.where(mask_perc, torch.zeros_like(g_perc_flux), g_perc_flux)

            # suz_st1 = suz + recharge + excess
            g_suz += g_suz_st1
            g_recharge += g_suz_st1
            g_excess += g_suz_st1

            # slz_st1 = slz + perc_flux
            g_slz += g_slz_st1

            # sm_after_evap 直通
            g_sm += g_sm_after

        return (g_sm, g_suz, g_slz, g_recharge, g_excess, g_perc, g_k0, g_k1, g_k2, g_uzl)


def run_block_grad_tests(device: Optional[str] = None) -> None:
    """运行三个拆分模块的 gradcheck，方便快速定位梯度问题。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.manual_seed(0)

    def _check(fn, inputs, name: str):
        test_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]
        ok = torch.autograd.gradcheck(lambda *args: fn.apply(*args), test_inputs, eps=1e-4, atol=1e-6, rtol=1e-6)
        print(f"[gradcheck] {name}: {ok}")

    # Snow block inputs
    p = torch.rand(8, device=device, dtype=dtype)
    t_val = torch.rand(8, device=device, dtype=dtype) * 4 + 1.0  # 远离阈值，避免符号切换
    snow = torch.rand(8, device=device, dtype=dtype) * 10
    melt = torch.rand(8, device=device, dtype=dtype) * 2
    tt = torch.zeros(8, device=device, dtype=dtype)
    cfmax = torch.ones(8, device=device, dtype=dtype) * 2.0
    cfr = torch.ones(8, device=device, dtype=dtype) * 0.05
    cwh = torch.ones(8, device=device, dtype=dtype) * 0.1
    _check(SnowBlock, [p, t_val, snow, melt, tt, cfmax, cfr, cwh], "snow")

    # Soil block inputs
    sm = torch.rand(8, device=device, dtype=dtype) * 60 + 20.0
    rain = torch.rand(8, device=device, dtype=dtype)
    tosoil = torch.rand(8, device=device, dtype=dtype)
    pet = torch.rand(8, device=device, dtype=dtype)
    fc = torch.ones(8, device=device, dtype=dtype) * 150.0
    beta = torch.ones(8, device=device, dtype=dtype) * 1.5
    lp = torch.ones(8, device=device, dtype=dtype) * 0.7
    betaet = torch.ones(8, device=device, dtype=dtype) * 1.2
    _check(SoilBlock, [sm, rain, tosoil, pet, fc, beta, lp, betaet], "soil")

    # Routing block inputs
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
    _check(RoutingBlock, [sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl], "routing")


if __name__=='__main__':
    run_block_grad_tests()