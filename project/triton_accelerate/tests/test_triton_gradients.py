"""
Property-based tests for Triton HBV backward kernel gradient correctness.
Compares Triton gradients against PyTorch autograd reference.

**Feature: triton-gradient-fix**
"""

import torch
import sys
import os
import importlib.util

# 直接加载模块文件，绕过 __init__.py 的依赖
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(script_dir), "models")
core_path = os.path.join(models_dir, "hbv_triton_core.py")

spec = importlib.util.spec_from_file_location("hbv_triton_core", core_path)
hbv_triton_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hbv_triton_core)

SnowBlockTriton = hbv_triton_core.SnowBlockTriton
SoilBlockTriton = hbv_triton_core.SoilBlockTriton
RoutingBlockTriton = hbv_triton_core.RoutingBlockTriton


def _create_reference_snow_forward(p, t_val, snow, melt, tt, cfmax, cfr, cwh):
    """Pure PyTorch forward for Snow Block (for autograd reference)."""
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

    return snow_st3, melt_out, tosoil, rain


def _create_reference_soil_forward(sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par):
    """Pure PyTorch forward for Soil Block (for autograd reference)."""
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


def _create_reference_routing_forward(sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl):
    """Pure PyTorch forward for Routing Block (for autograd reference)."""
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


def compare_gradients(triton_grads, ref_grads, names, rtol=1e-4, atol=1e-6):
    """Compare Triton gradients against reference, return max relative diff."""
    max_diff = 0.0
    failed = []
    for tg, rg, name in zip(triton_grads, ref_grads, names):
        if tg is None or rg is None:
            continue
        diff = torch.abs(tg - rg)
        rel_diff = diff / (torch.abs(rg) + 1e-12)
        max_rel = rel_diff.max().item()
        if max_rel > max_diff:
            max_diff = max_rel
        if max_rel > rtol:
            failed.append((name, max_rel))
    return max_diff, failed


# **Feature: triton-gradient-fix, Property 1: Snow Block Gradient Consistency**
# **Validates: Requirements 1.1, 1.4**
def test_snow_block_gradient_consistency(device=None, n_samples=100):
    """Property test: Snow Block Triton gradients match PyTorch autograd."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.manual_seed(42)
    
    passed = 0
    max_diff_overall = 0.0
    
    for i in range(n_samples):
        # Generate random inputs in physically reasonable ranges
        p = torch.rand(8, device=device, dtype=dtype) * 10  # precipitation
        t_val = torch.rand(8, device=device, dtype=dtype) * 10 - 2  # temperature
        snow = torch.rand(8, device=device, dtype=dtype) * 50  # snow state
        melt = torch.rand(8, device=device, dtype=dtype) * 10  # melt state
        tt = torch.rand(8, device=device, dtype=dtype) * 2 - 1  # threshold temp
        cfmax = torch.rand(8, device=device, dtype=dtype) * 5 + 0.5  # melt factor
        cfr = torch.rand(8, device=device, dtype=dtype) * 0.1  # refreeze coef
        cwh = torch.rand(8, device=device, dtype=dtype) * 0.2  # water holding
        
        # Triton forward/backward
        triton_inputs = [x.clone().detach().requires_grad_(True) for x in 
                        [p, t_val, snow, melt, tt, cfmax, cfr, cwh]]
        triton_out = SnowBlockTriton.apply(*triton_inputs)
        triton_loss = sum(o.sum() for o in triton_out)
        triton_loss.backward()
        triton_grads = [x.grad for x in triton_inputs]
        
        # Reference forward/backward
        ref_inputs = [x.clone().detach().requires_grad_(True) for x in 
                     [p, t_val, snow, melt, tt, cfmax, cfr, cwh]]
        ref_out = _create_reference_snow_forward(*ref_inputs)
        ref_loss = sum(o.sum() for o in ref_out)
        ref_loss.backward()
        ref_grads = [x.grad for x in ref_inputs]
        
        names = ['p', 't_val', 'snow', 'melt', 'tt', 'cfmax', 'cfr', 'cwh']
        max_diff, failed = compare_gradients(triton_grads, ref_grads, names)
        
        if max_diff > max_diff_overall:
            max_diff_overall = max_diff
        
        if not failed:
            passed += 1
    
    print(f"[Snow Block] Passed: {passed}/{n_samples}, Max diff: {max_diff_overall:.2e}")
    return passed == n_samples, max_diff_overall


# **Feature: triton-gradient-fix, Property 2: Soil Block Gradient Consistency**
# **Validates: Requirements 1.2, 1.4**
def test_soil_block_gradient_consistency(device=None, n_samples=100):
    """Property test: Soil Block Triton gradients match PyTorch autograd."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.manual_seed(42)
    
    passed = 0
    max_diff_overall = 0.0
    
    for i in range(n_samples):
        # Generate random inputs in physically reasonable ranges
        sm = torch.rand(8, device=device, dtype=dtype) * 100 + 10  # soil moisture
        slz = torch.rand(8, device=device, dtype=dtype) * 50 + 5  # lower zone
        rain = torch.rand(8, device=device, dtype=dtype) * 20  # rain
        tosoil = torch.rand(8, device=device, dtype=dtype) * 10  # melt to soil
        pet = torch.rand(8, device=device, dtype=dtype) * 5  # potential ET
        fc = torch.rand(8, device=device, dtype=dtype) * 200 + 50  # field capacity
        beta = torch.rand(8, device=device, dtype=dtype) * 3 + 1  # beta param
        lp = torch.rand(8, device=device, dtype=dtype) * 0.5 + 0.3  # LP param
        betaet = torch.rand(8, device=device, dtype=dtype) * 2 + 0.5  # betaET
        c_par = torch.rand(8, device=device, dtype=dtype) * 0.1  # capillary param
        
        # Triton forward/backward
        triton_inputs = [x.clone().detach().requires_grad_(True) for x in 
                        [sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par]]
        triton_out = SoilBlockTriton.apply(*triton_inputs)
        triton_loss = sum(o.sum() for o in triton_out)
        triton_loss.backward()
        triton_grads = [x.grad for x in triton_inputs]
        
        # Reference forward/backward
        ref_inputs = [x.clone().detach().requires_grad_(True) for x in 
                     [sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par]]
        ref_out = _create_reference_soil_forward(*ref_inputs)
        ref_loss = sum(o.sum() for o in ref_out)
        ref_loss.backward()
        ref_grads = [x.grad for x in ref_inputs]
        
        names = ['sm', 'slz', 'rain', 'tosoil', 'pet', 'fc', 'beta', 'lp', 'betaet', 'c_par']
        max_diff, failed = compare_gradients(triton_grads, ref_grads, names)
        
        if max_diff > max_diff_overall:
            max_diff_overall = max_diff
        
        if not failed:
            passed += 1
    
    print(f"[Soil Block] Passed: {passed}/{n_samples}, Max diff: {max_diff_overall:.2e}")
    return passed == n_samples, max_diff_overall


# **Feature: triton-gradient-fix, Property 3: Routing Block Gradient Consistency**
# **Validates: Requirements 1.3, 1.4**
def test_routing_block_gradient_consistency(device=None, n_samples=100):
    """Property test: Routing Block Triton gradients match PyTorch autograd."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.manual_seed(42)
    
    passed = 0
    max_diff_overall = 0.0
    
    for i in range(n_samples):
        # Generate random inputs in physically reasonable ranges
        sm_after = torch.rand(8, device=device, dtype=dtype) * 100 + 10
        suz = torch.rand(8, device=device, dtype=dtype) * 50 + 5
        slz = torch.rand(8, device=device, dtype=dtype) * 50 + 5
        recharge = torch.rand(8, device=device, dtype=dtype) * 10
        excess = torch.rand(8, device=device, dtype=dtype) * 5
        perc = torch.rand(8, device=device, dtype=dtype) * 5 + 0.5
        k0 = torch.rand(8, device=device, dtype=dtype) * 0.5 + 0.05
        k1 = torch.rand(8, device=device, dtype=dtype) * 0.3 + 0.01
        k2 = torch.rand(8, device=device, dtype=dtype) * 0.1 + 0.001
        uzl = torch.rand(8, device=device, dtype=dtype) * 30 + 5
        
        # Triton forward/backward
        triton_inputs = [x.clone().detach().requires_grad_(True) for x in 
                        [sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl]]
        triton_out = RoutingBlockTriton.apply(*triton_inputs)
        triton_loss = sum(o.sum() for o in triton_out)
        triton_loss.backward()
        triton_grads = [x.grad for x in triton_inputs]
        
        # Reference forward/backward
        ref_inputs = [x.clone().detach().requires_grad_(True) for x in 
                     [sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl]]
        ref_out = _create_reference_routing_forward(*ref_inputs)
        ref_loss = sum(o.sum() for o in ref_out)
        ref_loss.backward()
        ref_grads = [x.grad for x in ref_inputs]
        
        names = ['sm_after', 'suz', 'slz', 'recharge', 'excess', 'perc', 'k0', 'k1', 'k2', 'uzl']
        max_diff, failed = compare_gradients(triton_grads, ref_grads, names)
        
        if max_diff > max_diff_overall:
            max_diff_overall = max_diff
        
        if not failed:
            passed += 1
    
    print(f"[Routing Block] Passed: {passed}/{n_samples}, Max diff: {max_diff_overall:.2e}")
    return passed == n_samples, max_diff_overall


def run_gradcheck_tests(device=None):
    """Run torch.autograd.gradcheck on all blocks."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.manual_seed(0)
    
    results = {}
    
    # Snow block
    p = torch.rand(8, device=device, dtype=dtype) * 5 + 0.1
    t_val = torch.rand(8, device=device, dtype=dtype) * 4 + 1.0
    snow = torch.rand(8, device=device, dtype=dtype) * 10 + 1.0
    melt = torch.rand(8, device=device, dtype=dtype) * 2 + 0.1
    tt = torch.zeros(8, device=device, dtype=dtype)
    cfmax = torch.ones(8, device=device, dtype=dtype) * 2.0
    cfr = torch.ones(8, device=device, dtype=dtype) * 0.05
    cwh = torch.ones(8, device=device, dtype=dtype) * 0.1
    
    inputs = [x.clone().detach().requires_grad_(True) for x in [p, t_val, snow, melt, tt, cfmax, cfr, cwh]]
    try:
        ok = torch.autograd.gradcheck(lambda *args: SnowBlockTriton.apply(*args), inputs, eps=1e-4, atol=1e-5, rtol=1e-4)
        results['snow'] = ok
        print(f"[gradcheck] Snow Block: {ok}")
    except Exception as e:
        results['snow'] = False
        print(f"[gradcheck] Snow Block: FAILED - {e}")
    
    # Soil block
    sm = torch.rand(8, device=device, dtype=dtype) * 60 + 20.0
    slz = torch.rand(8, device=device, dtype=dtype) * 30 + 10.0
    rain = torch.rand(8, device=device, dtype=dtype) + 0.1
    tosoil = torch.rand(8, device=device, dtype=dtype) + 0.1
    pet = torch.rand(8, device=device, dtype=dtype) + 0.1
    fc = torch.ones(8, device=device, dtype=dtype) * 150.0
    beta = torch.ones(8, device=device, dtype=dtype) * 1.5
    lp = torch.ones(8, device=device, dtype=dtype) * 0.7
    betaet = torch.ones(8, device=device, dtype=dtype) * 1.2
    c_par = torch.ones(8, device=device, dtype=dtype) * 0.05
    
    inputs = [x.clone().detach().requires_grad_(True) for x in [sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par]]
    try:
        ok = torch.autograd.gradcheck(lambda *args: SoilBlockTriton.apply(*args), inputs, eps=1e-4, atol=1e-5, rtol=1e-4)
        results['soil'] = ok
        print(f"[gradcheck] Soil Block: {ok}")
    except Exception as e:
        results['soil'] = False
        print(f"[gradcheck] Soil Block: FAILED - {e}")
    
    # Routing block
    sm_after = torch.rand(8, device=device, dtype=dtype) * 50 + 10.0
    suz = torch.rand(8, device=device, dtype=dtype) * 30 + 5.0
    slz = torch.rand(8, device=device, dtype=dtype) * 30 + 5.0
    recharge = torch.rand(8, device=device, dtype=dtype) + 0.1
    excess = torch.rand(8, device=device, dtype=dtype) + 0.1
    perc = torch.ones(8, device=device, dtype=dtype) * 3.0
    k0 = torch.ones(8, device=device, dtype=dtype) * 0.25
    k1 = torch.ones(8, device=device, dtype=dtype) * 0.05
    k2 = torch.ones(8, device=device, dtype=dtype) * 0.01
    uzl = torch.ones(8, device=device, dtype=dtype) * 5.0
    
    inputs = [x.clone().detach().requires_grad_(True) for x in [sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl]]
    try:
        ok = torch.autograd.gradcheck(lambda *args: RoutingBlockTriton.apply(*args), inputs, eps=1e-4, atol=1e-5, rtol=1e-4)
        results['routing'] = ok
        print(f"[gradcheck] Routing Block: {ok}")
    except Exception as e:
        results['routing'] = False
        print(f"[gradcheck] Routing Block: FAILED - {e}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Running Triton Gradient Tests")
    print("=" * 60)
    
    # Run gradcheck tests
    print("\n--- Gradcheck Tests ---")
    gradcheck_results = run_gradcheck_tests()
    
    # Run property tests
    print("\n--- Property Tests (100 samples each) ---")
    snow_ok, snow_diff = test_snow_block_gradient_consistency(n_samples=100)
    soil_ok, soil_diff = test_soil_block_gradient_consistency(n_samples=100)
    routing_ok, routing_diff = test_routing_block_gradient_consistency(n_samples=100)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Snow Block:    {'PASS' if snow_ok else 'FAIL'} (max diff: {snow_diff:.2e})")
    print(f"  Soil Block:    {'PASS' if soil_ok else 'FAIL'} (max diff: {soil_diff:.2e})")
    print(f"  Routing Block: {'PASS' if routing_ok else 'FAIL'} (max diff: {routing_diff:.2e})")
    print("=" * 60)
