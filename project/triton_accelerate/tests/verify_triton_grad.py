"""
验证 Triton backward kernel 的梯度正确性
通过与 PyTorch autograd 对比来检测错误

这个脚本可以在 GPU 服务器上运行
"""

import torch
import sys
import os

# 添加项目路径，直接导入模块文件避免 __init__.py 的 hydra 依赖
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 使用 importlib 直接导入，绕过 __init__.py
import importlib.util

def load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hbv_triton_core = load_module_directly("hbv_triton_core", os.path.join(base_path, "models", "hbv_triton_core.py"))

SnowBlockTriton = hbv_triton_core.SnowBlockTriton
SoilBlockTriton = hbv_triton_core.SoilBlockTriton
RoutingBlockTriton = hbv_triton_core.RoutingBlockTriton


def test_snow_block_gradient():
    """测试 Snow Block 梯度"""
    print("\n=== Testing Snow Block Gradient ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # 用 float64 提高精度
    
    torch.manual_seed(42)
    
    # 创建输入 - Triton 测试
    p = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    t_val = (torch.rand(16, device=device, dtype=dtype) * 4 - 1).requires_grad_(True)
    snow = (torch.rand(16, device=device, dtype=dtype) * 10).requires_grad_(True)
    melt = (torch.rand(16, device=device, dtype=dtype) * 2).requires_grad_(True)
    tt = torch.zeros(16, device=device, dtype=dtype).requires_grad_(True)
    cfmax = (torch.ones(16, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    cfr = (torch.ones(16, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    cwh = (torch.ones(16, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    
    # Triton forward + backward
    snow_out, melt_out, tosoil, rain = SnowBlockTriton.apply(
        p, t_val, snow, melt, tt, cfmax, cfr, cwh
    )
    loss_triton = snow_out.sum() + melt_out.sum() + tosoil.sum() + rain.sum()
    loss_triton.backward()
    
    triton_grads = {
        'p': p.grad.clone(),
        't_val': t_val.grad.clone(),
        'snow': snow.grad.clone(),
        'melt': melt.grad.clone(),
        'tt': tt.grad.clone(),
        'cfmax': cfmax.grad.clone(),
        'cfr': cfr.grad.clone(),
        'cwh': cwh.grad.clone(),
    }
    
    # 重新创建输入 - PyTorch 参考测试
    torch.manual_seed(42)
    p2 = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    t_val2 = (torch.rand(16, device=device, dtype=dtype) * 4 - 1).requires_grad_(True)
    snow2 = (torch.rand(16, device=device, dtype=dtype) * 10).requires_grad_(True)
    melt2 = (torch.rand(16, device=device, dtype=dtype) * 2).requires_grad_(True)
    tt2 = torch.zeros(16, device=device, dtype=dtype).requires_grad_(True)
    cfmax2 = (torch.ones(16, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    cfr2 = (torch.ones(16, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    cwh2 = (torch.ones(16, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    
    # PyTorch autograd forward + backward (作为参考)
    temp_diff = t_val2 - tt2
    is_rain = temp_diff > 0.0
    rain_ref = torch.where(is_rain, p2, torch.zeros_like(p2))
    snow_input = torch.where(is_rain, torch.zeros_like(p2), p2)
    
    snow_st1 = snow2 + snow_input
    pot_melt = cfmax2 * torch.clamp(temp_diff, min=0.0)
    melt_amount = torch.minimum(pot_melt, snow_st1)
    snow_st2 = snow_st1 - melt_amount
    melt_st1 = melt2 + melt_amount
    
    pot_refreeze = cfr2 * cfmax2 * torch.clamp(-temp_diff, min=0.0)
    refreeze_amt = torch.minimum(pot_refreeze, melt_st1)
    snow_out_ref = snow_st2 + refreeze_amt
    melt_st2 = melt_st1 - refreeze_amt
    
    tosoil_ref = torch.clamp(melt_st2 - cwh2 * snow_out_ref, min=0.0)
    melt_out_ref = melt_st2 - tosoil_ref
    
    loss_ref = snow_out_ref.sum() + melt_out_ref.sum() + tosoil_ref.sum() + rain_ref.sum()
    loss_ref.backward()
    
    ref_grads = {
        'p': p2.grad.clone(),
        't_val': t_val2.grad.clone(),
        'snow': snow2.grad.clone(),
        'melt': melt2.grad.clone(),
        'tt': tt2.grad.clone(),
        'cfmax': cfmax2.grad.clone(),
        'cfr': cfr2.grad.clone(),
        'cwh': cwh2.grad.clone(),
    }
    
    # 比较梯度
    print("\nGradient comparison (Triton vs PyTorch autograd):")
    all_close = True
    for name in triton_grads:
        diff = (triton_grads[name] - ref_grads[name]).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (ref_grads[name].abs() + 1e-8)).max().item()
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {name:8s}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} {status}")
        if max_diff >= 1e-4:
            all_close = False
    
    return all_close


def test_soil_block_gradient():
    """测试 Soil Block 梯度"""
    print("\n=== Testing Soil Block Gradient ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    
    torch.manual_seed(42)
    
    # 创建输入 - Triton 测试
    sm = (torch.rand(16, device=device, dtype=dtype) * 60 + 20).requires_grad_(True)
    slz = (torch.rand(16, device=device, dtype=dtype) * 30 + 10).requires_grad_(True)
    rain = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    tosoil = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    pet = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    fc = (torch.ones(16, device=device, dtype=dtype) * 150.0).requires_grad_(True)
    beta = (torch.ones(16, device=device, dtype=dtype) * 1.5).requires_grad_(True)
    lp = (torch.ones(16, device=device, dtype=dtype) * 0.7).requires_grad_(True)
    betaet = (torch.ones(16, device=device, dtype=dtype) * 1.2).requires_grad_(True)
    c_par = (torch.ones(16, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    
    # Triton forward + backward
    sm_out, slz_out, recharge, excess, soil_wet, evapfactor, capillary = SoilBlockTriton.apply(
        sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par
    )
    loss_triton = sm_out.sum() + slz_out.sum() + recharge.sum() + excess.sum()
    loss_triton.backward()
    
    triton_grads = {
        'sm': sm.grad.clone(),
        'slz': slz.grad.clone(),
        'rain': rain.grad.clone(),
        'tosoil': tosoil.grad.clone(),
        'pet': pet.grad.clone(),
        'fc': fc.grad.clone(),
        'beta': beta.grad.clone(),
        'lp': lp.grad.clone(),
        'betaet': betaet.grad.clone(),
        'c_par': c_par.grad.clone(),
    }
    
    # 重新创建输入 - PyTorch 参考测试
    torch.manual_seed(42)
    sm2 = (torch.rand(16, device=device, dtype=dtype) * 60 + 20).requires_grad_(True)
    slz2 = (torch.rand(16, device=device, dtype=dtype) * 30 + 10).requires_grad_(True)
    rain2 = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    tosoil2 = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    pet2 = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    fc2 = (torch.ones(16, device=device, dtype=dtype) * 150.0).requires_grad_(True)
    beta2 = (torch.ones(16, device=device, dtype=dtype) * 1.5).requires_grad_(True)
    lp2 = (torch.ones(16, device=device, dtype=dtype) * 0.7).requires_grad_(True)
    betaet2 = (torch.ones(16, device=device, dtype=dtype) * 1.2).requires_grad_(True)
    c_par2 = (torch.ones(16, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    
    # PyTorch autograd 参考实现
    eps = 1e-6
    soil_ratio = sm2 / fc2
    soil_wet_ref = torch.clamp(torch.pow(torch.clamp(soil_ratio, min=eps), beta2), 0.0, 1.0)
    recharge_ref = (rain2 + tosoil2) * soil_wet_ref
    
    sm_st1 = sm2 + rain2 + tosoil2 - recharge_ref
    excess_ref = torch.clamp(sm_st1 - fc2, min=0.0)
    sm_st2 = sm_st1 - excess_ref
    
    ef1 = sm_st2 / (lp2 * fc2)
    ef1 = torch.clamp(ef1, 0.0, 1.0)
    evapfactor_ref = torch.clamp(torch.pow(torch.clamp(ef1, min=eps), betaet2), 0.0, 1.0)
    etact = torch.minimum(pet2 * evapfactor_ref, sm_st2)
    sm_after_evap = torch.clamp(sm_st2 - etact, min=eps)
    
    sm_ratio = torch.clamp(sm_after_evap / fc2, max=1.0)
    capillary_ref = torch.minimum(slz2, c_par2 * slz2 * (1.0 - sm_ratio))
    sm_out_ref = torch.clamp(sm_after_evap + capillary_ref, min=eps)
    slz_out_ref = torch.clamp(slz2 - capillary_ref, min=eps)
    
    loss_ref = sm_out_ref.sum() + slz_out_ref.sum() + recharge_ref.sum() + excess_ref.sum()
    loss_ref.backward()
    
    ref_grads = {
        'sm': sm2.grad.clone(),
        'slz': slz2.grad.clone(),
        'rain': rain2.grad.clone(),
        'tosoil': tosoil2.grad.clone(),
        'pet': pet2.grad.clone(),
        'fc': fc2.grad.clone(),
        'beta': beta2.grad.clone(),
        'lp': lp2.grad.clone(),
        'betaet': betaet2.grad.clone(),
        'c_par': c_par2.grad.clone(),
    }
    
    # 比较梯度
    print("\nGradient comparison (Triton vs PyTorch autograd):")
    all_close = True
    for name in triton_grads:
        diff = (triton_grads[name] - ref_grads[name]).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (ref_grads[name].abs() + 1e-8)).max().item()
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {name:8s}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} {status}")
        if max_diff >= 1e-4:
            all_close = False
            print(f"    Triton: {triton_grads[name][:4].tolist()}")
            print(f"    Ref:    {ref_grads[name][:4].tolist()}")
    
    return all_close


def test_routing_block_gradient():
    """测试 Routing Block 梯度"""
    print("\n=== Testing Routing Block Gradient ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    
    torch.manual_seed(42)
    
    # 创建输入 - Triton 测试
    sm_after = (torch.rand(16, device=device, dtype=dtype) * 50 + 1).requires_grad_(True)
    suz = (torch.rand(16, device=device, dtype=dtype) * 30 + 1).requires_grad_(True)
    slz = (torch.rand(16, device=device, dtype=dtype) * 30 + 1).requires_grad_(True)
    recharge = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    excess = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    perc = (torch.ones(16, device=device, dtype=dtype) * 3.0).requires_grad_(True)
    k0 = (torch.ones(16, device=device, dtype=dtype) * 0.25).requires_grad_(True)
    k1 = (torch.ones(16, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    k2 = (torch.ones(16, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    uzl = (torch.ones(16, device=device, dtype=dtype) * 5.0).requires_grad_(True)
    
    # Triton forward + backward
    sm_out, suz_out, slz_out, q_out = RoutingBlockTriton.apply(
        sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl
    )
    loss_triton = sm_out.sum() + suz_out.sum() + slz_out.sum() + q_out.sum()
    loss_triton.backward()
    
    triton_grads = {
        'sm_after': sm_after.grad.clone(),
        'suz': suz.grad.clone(),
        'slz': slz.grad.clone(),
        'recharge': recharge.grad.clone(),
        'excess': excess.grad.clone(),
        'perc': perc.grad.clone(),
        'k0': k0.grad.clone(),
        'k1': k1.grad.clone(),
        'k2': k2.grad.clone(),
        'uzl': uzl.grad.clone(),
    }
    
    # 重新创建输入 - PyTorch 参考测试
    torch.manual_seed(42)
    sm_after2 = (torch.rand(16, device=device, dtype=dtype) * 50 + 1).requires_grad_(True)
    suz2 = (torch.rand(16, device=device, dtype=dtype) * 30 + 1).requires_grad_(True)
    slz2 = (torch.rand(16, device=device, dtype=dtype) * 30 + 1).requires_grad_(True)
    recharge2 = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    excess2 = torch.rand(16, device=device, dtype=dtype).requires_grad_(True)
    perc2 = (torch.ones(16, device=device, dtype=dtype) * 3.0).requires_grad_(True)
    k02 = (torch.ones(16, device=device, dtype=dtype) * 0.25).requires_grad_(True)
    k12 = (torch.ones(16, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    k22 = (torch.ones(16, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    uzl2 = (torch.ones(16, device=device, dtype=dtype) * 5.0).requires_grad_(True)
    
    # PyTorch autograd 参考实现
    suz_st1 = suz2 + recharge2 + excess2
    perc_flux = torch.minimum(suz_st1, perc2)
    suz_st2 = suz_st1 - perc_flux
    slz_st1 = slz2 + perc_flux
    
    q0 = k02 * torch.clamp(suz_st2 - uzl2, min=0.0)
    suz_st3 = suz_st2 - q0
    q1 = k12 * suz_st3
    suz_out_ref = suz_st3 - q1
    
    q2 = k22 * slz_st1
    slz_out_ref = slz_st1 - q2
    
    q_total = q0 + q1 + q2
    
    loss_ref = sm_after2.sum() + suz_out_ref.sum() + slz_out_ref.sum() + q_total.sum()
    loss_ref.backward()
    
    ref_grads = {
        'sm_after': sm_after2.grad.clone(),
        'suz': suz2.grad.clone(),
        'slz': slz2.grad.clone(),
        'recharge': recharge2.grad.clone(),
        'excess': excess2.grad.clone(),
        'perc': perc2.grad.clone(),
        'k0': k02.grad.clone(),
        'k1': k12.grad.clone(),
        'k2': k22.grad.clone(),
        'uzl': uzl2.grad.clone(),
    }
    
    # 比较梯度
    print("\nGradient comparison (Triton vs PyTorch autograd):")
    all_close = True
    for name in triton_grads:
        diff = (triton_grads[name] - ref_grads[name]).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (ref_grads[name].abs() + 1e-8)).max().item()
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {name:8s}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} {status}")
        if max_diff >= 1e-4:
            all_close = False
            print(f"    Triton: {triton_grads[name][:4].tolist()}")
            print(f"    Ref:    {ref_grads[name][:4].tolist()}")
    
    return all_close


def main():
    print("=" * 60)
    print("Triton Backward Kernel Gradient Verification")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    snow_ok = test_snow_block_gradient()
    soil_ok = test_soil_block_gradient()
    routing_ok = test_routing_block_gradient()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Snow Block:    {'PASS' if snow_ok else 'FAIL'}")
    print(f"  Soil Block:    {'PASS' if soil_ok else 'FAIL'}")
    print(f"  Routing Block: {'PASS' if routing_ok else 'FAIL'}")
    print("=" * 60)
    
    if snow_ok and soil_ok and routing_ok:
        print("\n所有梯度测试通过！Triton backward kernel 正确。")
    else:
        print("\n存在梯度错误，需要修复 Triton backward kernel。")


if __name__ == "__main__":
    main()
