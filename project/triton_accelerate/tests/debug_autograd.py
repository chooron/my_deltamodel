"""
调试 autograd.Function 的 backward 方法
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

_routing_forward_kernel = hbv_triton_core._routing_forward_kernel
_routing_backward_kernel = hbv_triton_core._routing_backward_kernel


class RoutingBlockDebug(torch.autograd.Function):
    """带调试输出的 RoutingBlock"""
    
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
        print("\n=== 进入 backward ===")
        print(f"g_sm_out 类型: {type(g_sm_out)}")
        print(f"g_sm_out shape: {g_sm_out.shape}")
        print(f"g_sm_out 值: {g_sm_out.tolist()}")
        print(f"g_sm_out.is_contiguous(): {g_sm_out.is_contiguous()}")
        print(f"g_sm_out.stride(): {g_sm_out.stride()}")
        
        sm_after_evap, suz, slz, recharge, excess, perc, k0, k1, k2, uzl = ctx.saved_tensors
        n = ctx.n
        
        # 确保输入梯度是连续的
        g_sm_out = g_sm_out.contiguous()
        g_suz_out = g_suz_out.contiguous()
        g_slz_out = g_slz_out.contiguous()
        g_q_out = g_q_out.contiguous()
        
        print(f"\n确保连续后:")
        print(f"g_sm_out 值: {g_sm_out.tolist()}")
        print(f"g_sm_out.is_contiguous(): {g_sm_out.is_contiguous()}")
        
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
        
        torch.cuda.synchronize()
        
        print(f"\nKernel 输出:")
        print(f"g_sm: {g_sm.tolist()}")
        
        return g_sm, g_suz, g_slz, g_recharge, g_excess, g_perc, g_k0, g_k1, g_k2, g_uzl


def test_with_debug():
    print("=== 测试带调试的 RoutingBlock ===")
    device = "cuda"
    dtype = torch.float64
    n = 4
    
    sm_after = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    suz = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    slz = torch.ones(n, device=device, dtype=dtype).requires_grad_(True)
    recharge = (torch.ones(n, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    excess = (torch.ones(n, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    perc = (torch.ones(n, device=device, dtype=dtype) * 3.0).requires_grad_(True)
    k0 = (torch.ones(n, device=device, dtype=dtype) * 0.25).requires_grad_(True)
    k1 = (torch.ones(n, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    k2 = (torch.ones(n, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    uzl = (torch.ones(n, device=device, dtype=dtype) * 5.0).requires_grad_(True)
    
    sm_out, suz_out, slz_out, q_out = RoutingBlockDebug.apply(
        sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl
    )
    
    print(f"\nForward 输出:")
    print(f"sm_out: {sm_out.tolist()}")
    
    loss = sm_out.sum()
    print(f"\nLoss: {loss.item()}")
    
    loss.backward()
    
    print(f"\n最终梯度:")
    print(f"sm_after.grad: {sm_after.grad.tolist()}")


if __name__ == "__main__":
    test_with_debug()
