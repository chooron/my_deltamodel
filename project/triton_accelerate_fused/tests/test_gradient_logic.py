"""
纯 PyTorch 梯度逻辑验证测试（不需要 GPU/Triton）
验证手写梯度公式的正确性

在有 GPU 的服务器上运行完整的 Triton 测试:
    python project/triton_accelerate/tests/test_triton_gradients.py

在本地验证梯度逻辑:
    python project/triton_accelerate/tests/test_gradient_logic.py
"""

import torch


def snow_forward_pytorch(p, t_val, snow, melt, tt, cfmax, cfr, cwh):
    """Snow Block 前向计算 - 纯 PyTorch"""
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

    return snow_st3, melt_out, tosoil, rain


def soil_forward_pytorch(sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par):
    """Soil Block 前向计算 - 纯 PyTorch"""
    eps = 1e-6
    
    soil_ratio = sm / fc
    soil_wet = torch.clamp(torch.pow(torch.clamp(soil_ratio, min=eps), beta), 0.0, 1.0)
    recharge = (rain + tosoil) * soil_wet

    sm_st1 = sm + rain + tosoil - recharge
    excess = torch.clamp(sm_st1 - fc, min=0.0)
    sm_st2 = sm_st1 - excess

    ef1 = sm_st2 / (lp * fc)
    ef1 = torch.clamp(ef1, 0.0, 1.0)
    evapfactor = torch.clamp(torch.pow(torch.clamp(ef1, min=eps), betaet), 0.0, 1.0)

    etact = torch.minimum(pet * evapfactor, sm_st2)
    sm_after_evap = torch.clamp(sm_st2 - etact, min=eps)

    sm_ratio = torch.clamp(sm_after_evap / fc, max=1.0)
    capillary = torch.minimum(slz, c_par * slz * (1.0 - sm_ratio))
    sm_out = torch.clamp(sm_after_evap + capillary, min=eps)
    slz_out = torch.clamp(slz - capillary, min=eps)

    return sm_out, slz_out, recharge, excess, soil_wet, evapfactor, capillary


def routing_forward_pytorch(sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl):
    """Routing Block 前向计算 - 纯 PyTorch"""
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
    return sm_after, suz_out, slz_out, q_total


def test_snow_gradcheck():
    """测试 Snow Block 梯度 - 使用 PyTorch autograd"""
    print("\n[Snow Block] Running gradcheck...")
    dtype = torch.float64
    torch.manual_seed(42)
    
    p = torch.rand(8, dtype=dtype, requires_grad=True) * 5 + 0.1
    t_val = torch.rand(8, dtype=dtype, requires_grad=True) * 4 + 1.0
    snow = torch.rand(8, dtype=dtype, requires_grad=True) * 10 + 1.0
    melt = torch.rand(8, dtype=dtype, requires_grad=True) * 2 + 0.1
    tt = torch.zeros(8, dtype=dtype, requires_grad=True)
    cfmax = torch.ones(8, dtype=dtype, requires_grad=True) * 2.0
    cfr = torch.ones(8, dtype=dtype, requires_grad=True) * 0.05
    cwh = torch.ones(8, dtype=dtype, requires_grad=True) * 0.1
    
    inputs = (p, t_val, snow, melt, tt, cfmax, cfr, cwh)
    
    try:
        ok = torch.autograd.gradcheck(
            lambda *args: snow_forward_pytorch(*args),
            inputs, eps=1e-4, atol=1e-5, rtol=1e-4
        )
        print(f"[Snow Block] gradcheck: {ok}")
        return ok
    except Exception as e:
        print(f"[Snow Block] gradcheck FAILED: {e}")
        return False


def test_soil_gradcheck():
    """测试 Soil Block 梯度 - 使用 PyTorch autograd"""
    print("\n[Soil Block] Running gradcheck...")
    dtype = torch.float64
    torch.manual_seed(42)
    
    sm = torch.rand(8, dtype=dtype, requires_grad=True) * 60 + 20.0
    slz = torch.rand(8, dtype=dtype, requires_grad=True) * 30 + 10.0
    rain = torch.rand(8, dtype=dtype, requires_grad=True) + 0.1
    tosoil = torch.rand(8, dtype=dtype, requires_grad=True) + 0.1
    pet = torch.rand(8, dtype=dtype, requires_grad=True) + 0.1
    fc = torch.ones(8, dtype=dtype, requires_grad=True) * 150.0
    beta = torch.ones(8, dtype=dtype, requires_grad=True) * 1.5
    lp = torch.ones(8, dtype=dtype, requires_grad=True) * 0.7
    betaet = torch.ones(8, dtype=dtype, requires_grad=True) * 1.2
    c_par = torch.ones(8, dtype=dtype, requires_grad=True) * 0.05
    
    inputs = (sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par)
    
    try:
        ok = torch.autograd.gradcheck(
            lambda *args: soil_forward_pytorch(*args),
            inputs, eps=1e-4, atol=1e-5, rtol=1e-4
        )
        print(f"[Soil Block] gradcheck: {ok}")
        return ok
    except Exception as e:
        print(f"[Soil Block] gradcheck FAILED: {e}")
        return False


def test_routing_gradcheck():
    """测试 Routing Block 梯度 - 使用 PyTorch autograd"""
    print("\n[Routing Block] Running gradcheck...")
    dtype = torch.float64
    torch.manual_seed(42)
    
    sm_after = torch.rand(8, dtype=dtype, requires_grad=True) * 50 + 10.0
    suz = torch.rand(8, dtype=dtype, requires_grad=True) * 30 + 5.0
    slz = torch.rand(8, dtype=dtype, requires_grad=True) * 30 + 5.0
    recharge = torch.rand(8, dtype=dtype, requires_grad=True) + 0.1
    excess = torch.rand(8, dtype=dtype, requires_grad=True) + 0.1
    perc = torch.ones(8, dtype=dtype, requires_grad=True) * 3.0
    k0 = torch.ones(8, dtype=dtype, requires_grad=True) * 0.25
    k1 = torch.ones(8, dtype=dtype, requires_grad=True) * 0.05
    k2 = torch.ones(8, dtype=dtype, requires_grad=True) * 0.01
    uzl = torch.ones(8, dtype=dtype, requires_grad=True) * 5.0
    
    inputs = (sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl)
    
    try:
        ok = torch.autograd.gradcheck(
            lambda *args: routing_forward_pytorch(*args),
            inputs, eps=1e-4, atol=1e-5, rtol=1e-4
        )
        print(f"[Routing Block] gradcheck: {ok}")
        return ok
    except Exception as e:
        print(f"[Routing Block] gradcheck FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 梯度逻辑验证测试 (CPU)")
    print("=" * 60)
    
    snow_ok = test_snow_gradcheck()
    soil_ok = test_soil_gradcheck()
    routing_ok = test_routing_gradcheck()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Snow Block:    {'PASS' if snow_ok else 'FAIL'}")
    print(f"  Soil Block:    {'PASS' if soil_ok else 'FAIL'}")
    print(f"  Routing Block: {'PASS' if routing_ok else 'FAIL'}")
    print("=" * 60)
    
    if snow_ok and soil_ok and routing_ok:
        print("\n✓ 所有前向计算的梯度逻辑正确！")
        print("  请在有 GPU 的服务器上运行完整的 Triton 测试:")
        print("  python project/triton_accelerate/tests/test_triton_gradients.py")
    else:
        print("\n✗ 存在梯度问题，请检查前向计算逻辑")
