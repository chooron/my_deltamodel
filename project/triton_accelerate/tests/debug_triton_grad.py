"""
Debug script to isolate Triton gradient issues
"""

import torch
import triton
import triton.language as tl
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import importlib.util

def load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hbv_triton_core = load_module_directly("hbv_triton_core", os.path.join(base_path, "models", "hbv_triton_core.py"))

RoutingBlockTriton = hbv_triton_core.RoutingBlockTriton
SoilBlockTriton = hbv_triton_core.SoilBlockTriton


# Simple test kernel to verify Triton is working correctly
@triton.jit
def _simple_backward_kernel(
    x_ptr, g_out_ptr, g_x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple backward: y = x, so g_x = g_out"""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    g_out = tl.load(g_out_ptr + offs, mask=mask, other=0.0)
    g_x = g_out  # Simple pass-through
    
    tl.store(g_x_ptr + offs, g_x, mask=mask)


def test_simple_triton_kernel():
    """Test that basic Triton kernel works correctly"""
    print("\n=== Test Simple Triton Kernel ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping - need CUDA")
        return
    
    dtype = torch.float64
    n = 16
    
    x = torch.ones(n, device=device, dtype=dtype)
    g_out = torch.ones(n, device=device, dtype=dtype)
    g_x = torch.empty(n, device=device, dtype=dtype)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _simple_backward_kernel[grid](x, g_out, g_x, n, BLOCK_SIZE=256)
    
    print(f"g_out: {g_out.tolist()}")
    print(f"g_x:   {g_x.tolist()}")
    print(f"Expected: all 1.0")
    print(f"Match: {torch.allclose(g_x, g_out)}")


def debug_routing_block():
    """Debug Routing Block gradient with simple inputs"""
    print("\n=== Debug Routing Block Gradient ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    
    # Use very simple inputs - all same values
    n = 4
    sm_after = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    suz = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    slz = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    recharge = torch.ones(n, device=device, dtype=dtype) * 0.5
    recharge = recharge.requires_grad_(True)
    excess = torch.ones(n, device=device, dtype=dtype) * 0.5
    excess = excess.requires_grad_(True)
    perc = torch.ones(n, device=device, dtype=dtype) * 3.0
    perc = perc.requires_grad_(True)
    k0 = torch.ones(n, device=device, dtype=dtype) * 0.25
    k0 = k0.requires_grad_(True)
    k1 = torch.ones(n, device=device, dtype=dtype) * 0.05
    k1 = k1.requires_grad_(True)
    k2 = torch.ones(n, device=device, dtype=dtype) * 0.01
    k2 = k2.requires_grad_(True)
    uzl = torch.ones(n, device=device, dtype=dtype) * 5.0
    uzl = uzl.requires_grad_(True)
    
    print(f"Input shapes: sm_after={sm_after.shape}, suz={suz.shape}")
    print(f"Input values: sm_after={sm_after.tolist()}")
    
    # Triton forward + backward
    sm_out, suz_out, slz_out, q_out = RoutingBlockTriton.apply(
        sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl
    )
    
    print(f"\nForward outputs:")
    print(f"  sm_out: {sm_out.tolist()}")
    print(f"  suz_out: {suz_out.tolist()}")
    print(f"  slz_out: {slz_out.tolist()}")
    print(f"  q_out: {q_out.tolist()}")
    
    # Simple loss - just sum sm_out
    loss = sm_out.sum()
    print(f"\nLoss (sm_out.sum()): {loss.item()}")
    
    loss.backward()
    
    print(f"\nTriton gradients:")
    print(f"  g_sm_after: {sm_after.grad.tolist()}")
    print(f"  g_suz: {suz.grad.tolist()}")
    print(f"  g_slz: {slz.grad.tolist()}")
    
    # Expected: g_sm_after should be all 1.0 since sm_out = sm_after (pass-through)
    print(f"\nExpected g_sm_after: [1.0, 1.0, 1.0, 1.0]")
    
    # Now test with PyTorch reference
    print("\n--- PyTorch Reference ---")
    sm_after2 = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    suz2 = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    slz2 = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    recharge2 = (torch.ones(n, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    excess2 = (torch.ones(n, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    perc2 = (torch.ones(n, device=device, dtype=dtype) * 3.0).requires_grad_(True)
    k02 = (torch.ones(n, device=device, dtype=dtype) * 0.25).requires_grad_(True)
    k12 = (torch.ones(n, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    k22 = (torch.ones(n, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    uzl2 = (torch.ones(n, device=device, dtype=dtype) * 5.0).requires_grad_(True)
    
    # PyTorch forward
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
    
    sm_out_ref = sm_after2  # pass-through
    
    loss_ref = sm_out_ref.sum()
    loss_ref.backward()
    
    print(f"PyTorch gradients:")
    print(f"  g_sm_after: {sm_after2.grad.tolist()}")
    print(f"  g_suz: {suz2.grad.tolist()}")
    print(f"  g_slz: {slz2.grad.tolist()}")


def debug_soil_block():
    """Debug Soil Block gradient"""
    print("\n=== Debug Soil Block Gradient ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    
    n = 4
    # Use identical values for all elements to isolate the issue
    sm = torch.ones(n, device=device, dtype=dtype) * 50.0
    sm = sm.requires_grad_(True)
    slz = torch.ones(n, device=device, dtype=dtype) * 20.0
    slz = slz.requires_grad_(True)
    rain = torch.ones(n, device=device, dtype=dtype) * 0.5
    rain = rain.requires_grad_(True)
    tosoil = torch.ones(n, device=device, dtype=dtype) * 0.5
    tosoil = tosoil.requires_grad_(True)
    pet = torch.ones(n, device=device, dtype=dtype) * 0.5
    pet = pet.requires_grad_(True)
    fc = torch.ones(n, device=device, dtype=dtype) * 150.0
    fc = fc.requires_grad_(True)
    beta = torch.ones(n, device=device, dtype=dtype) * 1.5
    beta = beta.requires_grad_(True)
    lp = torch.ones(n, device=device, dtype=dtype) * 0.7
    lp = lp.requires_grad_(True)
    betaet = torch.ones(n, device=device, dtype=dtype) * 1.2
    betaet = betaet.requires_grad_(True)
    c_par = torch.ones(n, device=device, dtype=dtype) * 0.05
    c_par = c_par.requires_grad_(True)
    
    print(f"Input sm: {sm.tolist()}")
    
    # Triton forward + backward
    sm_out, slz_out, recharge, excess, soil_wet, evapfactor, capillary = SoilBlockTriton.apply(
        sm, slz, rain, tosoil, pet, fc, beta, lp, betaet, c_par
    )
    
    print(f"\nForward outputs:")
    print(f"  sm_out: {sm_out.tolist()}")
    print(f"  slz_out: {slz_out.tolist()}")
    
    # Simple loss
    loss = sm_out.sum()
    print(f"\nLoss (sm_out.sum()): {loss.item()}")
    
    loss.backward()
    
    print(f"\nTriton gradients:")
    print(f"  g_sm: {sm.grad.tolist()}")
    print(f"  g_slz: {slz.grad.tolist()}")
    
    # With identical inputs, all gradients should be identical
    print(f"\nExpected: all elements should have same gradient value")


if __name__ == "__main__":
    test_simple_triton_kernel()
    debug_routing_block()
    debug_soil_block()
