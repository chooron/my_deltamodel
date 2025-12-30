"""
直接测试 Triton backward kernel，绕过 autograd.Function
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

_routing_backward_kernel = hbv_triton_core._routing_backward_kernel


def test_routing_backward_direct():
    """直接调用 _routing_backward_kernel 测试"""
    print("\n=== 直接测试 Routing Backward Kernel ===")
    device = "cuda"
    dtype = torch.float64
    n = 4
    
    # 创建输入张量
    sm_after = torch.ones(n, device=device, dtype=dtype)
    suz = torch.ones(n, device=device, dtype=dtype)
    slz = torch.ones(n, device=device, dtype=dtype)
    recharge = torch.ones(n, device=device, dtype=dtype) * 0.5
    excess = torch.ones(n, device=device, dtype=dtype) * 0.5
    perc = torch.ones(n, device=device, dtype=dtype) * 3.0
    k0 = torch.ones(n, device=device, dtype=dtype) * 0.25
    k1 = torch.ones(n, device=device, dtype=dtype) * 0.05
    k2 = torch.ones(n, device=device, dtype=dtype) * 0.01
    uzl = torch.ones(n, device=device, dtype=dtype) * 5.0
    
    # 创建输出梯度张量 - 全部为 1.0
    g_sm_out = torch.ones(n, device=device, dtype=dtype)
    g_suz_out = torch.zeros(n, device=device, dtype=dtype)
    g_slz_out = torch.zeros(n, device=device, dtype=dtype)
    g_q_out = torch.zeros(n, device=device, dtype=dtype)
    
    print(f"输入梯度 g_sm_out: {g_sm_out.tolist()}")
    print(f"g_sm_out.is_contiguous(): {g_sm_out.is_contiguous()}")
    print(f"g_sm_out.data_ptr(): {g_sm_out.data_ptr()}")
    
    # 创建输出张量
    g_sm = torch.empty(n, device=device, dtype=dtype)
    g_suz = torch.empty(n, device=device, dtype=dtype)
    g_slz = torch.empty(n, device=device, dtype=dtype)
    g_recharge = torch.empty(n, device=device, dtype=dtype)
    g_excess = torch.empty(n, device=device, dtype=dtype)
    g_perc = torch.empty(n, device=device, dtype=dtype)
    g_k0 = torch.empty(n, device=device, dtype=dtype)
    g_k1 = torch.empty(n, device=device, dtype=dtype)
    g_k2 = torch.empty(n, device=device, dtype=dtype)
    g_uzl = torch.empty(n, device=device, dtype=dtype)
    
    # 直接调用 kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _routing_backward_kernel[grid](
        sm_after, suz, slz, recharge, excess,
        perc, k0, k1, k2, uzl,
        g_sm_out, g_suz_out, g_slz_out, g_q_out,
        g_sm, g_suz, g_slz, g_recharge, g_excess,
        g_perc, g_k0, g_k1, g_k2, g_uzl,
        n,
        BLOCK_SIZE=256,
    )
    
    # 同步 CUDA
    torch.cuda.synchronize()
    
    print(f"\n输出梯度:")
    print(f"  g_sm: {g_sm.tolist()}")
    print(f"  g_suz: {g_suz.tolist()}")
    print(f"  g_slz: {g_slz.tolist()}")
    
    print(f"\n期望 g_sm: [1.0, 1.0, 1.0, 1.0] (因为 sm_out = sm_after 是直接传递)")
    
    # 检查是否正确
    expected = torch.ones(n, device=device, dtype=dtype)
    if torch.allclose(g_sm, expected):
        print("\n✓ g_sm 正确!")
    else:
        print(f"\n✗ g_sm 错误! 差异: {(g_sm - expected).tolist()}")


def test_simple_passthrough_kernel():
    """测试一个简单的直通 kernel"""
    print("\n=== 测试简单直通 Kernel ===")
    
    @triton.jit
    def _passthrough_kernel(
        g_in_ptr, g_out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        
        g_in = tl.load(g_in_ptr + offs, mask=mask, other=0.0)
        tl.store(g_out_ptr + offs, g_in, mask=mask)
    
    device = "cuda"
    dtype = torch.float64
    n = 4
    
    g_in = torch.ones(n, device=device, dtype=dtype)
    g_out = torch.empty(n, device=device, dtype=dtype)
    
    print(f"输入: {g_in.tolist()}")
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _passthrough_kernel[grid](g_in, g_out, n, BLOCK_SIZE=256)
    
    torch.cuda.synchronize()
    
    print(f"输出: {g_out.tolist()}")
    
    if torch.allclose(g_in, g_out):
        print("✓ 直通 kernel 正确!")
    else:
        print("✗ 直通 kernel 错误!")


def test_kernel_with_debug():
    """带调试输出的 kernel 测试"""
    print("\n=== 带调试的 Kernel 测试 ===")
    
    @triton.jit
    def _debug_kernel(
        g_in_ptr, g_out_ptr, debug_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        
        # 加载输入
        g_in = tl.load(g_in_ptr + offs, mask=mask, other=-999.0)
        
        # 存储调试信息 (offs 值)
        tl.store(debug_ptr + offs, offs.to(tl.float64), mask=mask)
        
        # 存储输出
        tl.store(g_out_ptr + offs, g_in, mask=mask)
    
    device = "cuda"
    dtype = torch.float64
    n = 4
    
    g_in = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
    g_out = torch.empty(n, device=device, dtype=dtype)
    debug = torch.empty(n, device=device, dtype=dtype)
    
    print(f"输入: {g_in.tolist()}")
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _debug_kernel[grid](g_in, g_out, debug, n, BLOCK_SIZE=256)
    
    torch.cuda.synchronize()
    
    print(f"输出: {g_out.tolist()}")
    print(f"调试 (offs): {debug.tolist()}")
    
    if torch.allclose(g_in, g_out):
        print("✓ 正确!")
    else:
        print("✗ 错误!")


if __name__ == "__main__":
    test_simple_passthrough_kernel()
    test_kernel_with_debug()
    test_routing_backward_direct()
