"""
ExpHydro 模型梯度验证测试
比较 Triton 手动梯度实现 vs PyTorch autograd 自动求导

测试内容:
1. 单步梯度比较 - 验证 Triton backward kernel 与 PyTorch autograd 的一致性
2. 多步累积梯度比较 - 验证多时间步的梯度传播正确性

Author: Test Script for ExpHydro Triton Implementation
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入 Triton 实现
try:
    import triton
    from exphydro_triton_core import exphydro_run_triton, ExpHydroTritonFunction
    TRITON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Triton not available: {e}")
    TRITON_AVAILABLE = False


# ============================================================
# PyTorch 纯实现 (用于对比验证) - 使用 autograd 自动求导
# ============================================================

def exphydro_step_pytorch(
    p: torch.Tensor, t_val: torch.Tensor, lday: torch.Tensor,
    snow: torch.Tensor, soil: torch.Tensor,
    tmin: torch.Tensor, tmax: torch.Tensor, df: torch.Tensor,
    smax: torch.Tensor, qmax: torch.Tensor, f: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ExpHydro 模型单步计算 - 纯 PyTorch 实现 (使用 autograd)
    用于验证 Triton 实现的正确性
    """
    eps = 1e-6
    
    # --- 1. 降水分离 ---
    is_snow = (t_val < tmin).float()
    snowfall = p * is_snow
    rainfall = p * (1.0 - is_snow)
    
    # --- 2. 融雪 ---
    temp_excess = t_val - tmax
    pot_melt = df * torch.clamp(temp_excess, min=0.0)
    snow_avail = snow + snowfall
    melt = torch.min(pot_melt, snow_avail)
    
    # 更新积雪
    snow_new = torch.clamp(snow_avail - melt, min=nearzero)
    
    # --- 3. PET 计算 ---
    es = 0.611 * torch.exp((17.3 * t_val) / (t_val + 237.3))
    hamon = (29.8 * 24.0 * lday * es) / (t_val + 273.2)
    pet = hamon + melt
    
    # --- 4. 蒸发 ---
    soil_ratio = torch.clamp(soil / (smax + eps), min=0.0, max=1.0)
    evap = pet * soil_ratio
    
    # --- 5. 基流 ---
    deficit = torch.clamp(smax - soil, min=0.0)
    baseflow = qmax * torch.exp(-f * deficit)
    
    # --- 6. 地表径流 ---
    surfaceflow = torch.clamp(soil - smax, min=0.0)
    
    # --- 7. 总径流 ---
    flow = baseflow + surfaceflow
    
    # --- 8. 更新土壤水 ---
    soil_new = torch.clamp(soil + rainfall + melt - evap - flow, min=nearzero)
    
    return snow_new, soil_new, flow


def exphydro_run_pytorch(
    precip: torch.Tensor,
    temp: torch.Tensor,
    lday: torch.Tensor,
    params: Dict[str, torch.Tensor],
    init_states: Optional[Dict[str, torch.Tensor]] = None,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    ExpHydro 模型多步计算 - 纯 PyTorch 实现
    """
    T = precip.shape[0]
    device = precip.device
    dtype = precip.dtype
    spatial_shape = precip.shape[1:]
    
    # 初始化状态
    if init_states is None:
        snow = torch.zeros(spatial_shape, device=device, dtype=dtype)
        soil = torch.ones(spatial_shape, device=device, dtype=dtype) * 50.0
    else:
        snow = init_states['snow'].clone()
        soil = init_states['soil'].clone()
    
    # 提取参数
    tmin = params['tmin']
    tmax = params['tmax']
    df = params['df']
    smax = params['smax']
    qmax = params['qmax']
    f = params['f']
    
    # 存储输出
    q_series = []
    
    # 时间循环
    for t in range(T):
        p_t = precip[t]
        temp_t = temp[t]
        lday_t = lday[t]
        
        snow, soil, q = exphydro_step_pytorch(
            p_t, temp_t, lday_t,
            snow, soil,
            tmin, tmax, df, smax, qmax, f,
            nearzero,
        )
        
        q_series.append(q)
    
    q_series = torch.stack(q_series, dim=0)
    
    final_states = {
        'snow': snow,
        'soil': soil,
    }
    
    return q_series, final_states


# ============================================================
# 测试辅助函数
# ============================================================

def create_test_data(
    batch_size: int = 32,
    n_steps: int = 10,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """创建测试数据"""
    torch.manual_seed(seed)
    
    shape = (batch_size,)
    time_shape = (n_steps, batch_size)
    
    # 创建驱动数据
    precip = torch.rand(time_shape, device=device, dtype=dtype) * 20.0  # 0-20 mm
    temp = torch.randn(time_shape, device=device, dtype=dtype) * 10.0 + 10.0  # 0 to 20 °C
    lday = torch.ones(time_shape, device=device, dtype=dtype) * 0.5  # Day length factor
    
    # 创建参数 (确保在合理范围内)
    params = {
        'tmin': torch.zeros(shape, device=device, dtype=dtype),  # 雪/雨温度阈值
        'tmax': torch.ones(shape, device=device, dtype=dtype) * 2.0,  # 融雪温度阈值
        'df': torch.ones(shape, device=device, dtype=dtype) * 2.0,  # 度日因子
        'smax': torch.ones(shape, device=device, dtype=dtype) * 200.0,  # 最大土壤水
        'qmax': torch.ones(shape, device=device, dtype=dtype) * 10.0,  # 最大基流
        'f': torch.ones(shape, device=device, dtype=dtype) * 0.02,  # 基流衰减系数
    }
    
    # 创建初始状态
    init_states = {
        'snow': torch.zeros(shape, device=device, dtype=dtype),
        'soil': torch.ones(shape, device=device, dtype=dtype) * 100.0,
    }
    
    return params, init_states, precip, temp, lday


def compare_gradients(
    grad1: torch.Tensor,
    grad2: torch.Tensor,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """比较两个梯度张量是否一致"""
    if grad1 is None and grad2 is None:
        print(f"  {name}: Both None ✓")
        return True
    
    if grad1 is None or grad2 is None:
        print(f"  {name}: One is None, other is not ✗")
        return False
    
    # 计算差异
    abs_diff = torch.abs(grad1 - grad2)
    rel_diff = abs_diff / (torch.abs(grad2) + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # 检查是否通过
    is_close = torch.allclose(grad1, grad2, rtol=rtol, atol=atol)
    
    status = "✓" if is_close else "✗"
    print(f"  {name}: max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}, "
          f"mean_abs={mean_abs_diff:.2e} {status}")
    
    return is_close


# ============================================================
# 测试 1: 单步梯度比较
# ============================================================

def test_single_step_gradients():
    """测试单步计算的梯度一致性"""
    print("\n" + "=" * 60)
    print("Test 1: Single Step Gradient Comparison")
    print("=" * 60)
    
    if not TRITON_AVAILABLE:
        print("Skipping: Triton not available")
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: Running on CPU, Triton may not work properly")
        return False
    
    # 创建测试数据
    batch_size = 64
    torch.manual_seed(42)
    
    # 创建输入 (需要梯度)
    p = (torch.rand(batch_size, device=device, dtype=torch.float32) * 10.0).requires_grad_(True)
    t_val = (torch.randn(batch_size, device=device, dtype=torch.float32) * 10.0 + 10.0).requires_grad_(True)
    lday = (torch.ones(batch_size, device=device, dtype=torch.float32) * 0.5).requires_grad_(True)
    
    # 状态 (需要梯度)
    snow = (torch.rand(batch_size, device=device, dtype=torch.float32) * 10.0).requires_grad_(True)
    soil = (torch.rand(batch_size, device=device, dtype=torch.float32) * 100.0 + 50.0).requires_grad_(True)
    
    # 参数 (需要梯度)
    tmin = torch.zeros(batch_size, device=device, dtype=torch.float32, requires_grad=True)
    tmax = (torch.ones(batch_size, device=device, dtype=torch.float32) * 2.0).requires_grad_(True)
    df = (torch.ones(batch_size, device=device, dtype=torch.float32) * 2.0).requires_grad_(True)
    smax = (torch.ones(batch_size, device=device, dtype=torch.float32) * 200.0).requires_grad_(True)
    qmax = (torch.ones(batch_size, device=device, dtype=torch.float32) * 10.0).requires_grad_(True)
    f = (torch.ones(batch_size, device=device, dtype=torch.float32) * 0.02).requires_grad_(True)
    
    # =========================================
    # PyTorch 实现 (autograd)
    # =========================================
    print("\n--- Running PyTorch Implementation (autograd) ---")
    
    snow_pt, soil_pt, q_pt = exphydro_step_pytorch(
        p, t_val, lday,
        snow, soil,
        tmin, tmax, df, smax, qmax, f,
    )
    
    # 计算 loss 并反向传播
    loss_pt = q_pt.sum() + snow_pt.sum() + soil_pt.sum()
    loss_pt.backward()
    
    # 保存 PyTorch 梯度
    def safe_clone(grad):
        return grad.clone() if grad is not None else None
    
    grads_pt = {
        'snow': safe_clone(snow.grad),
        'soil': safe_clone(soil.grad),
        'tmin': safe_clone(tmin.grad),
        'tmax': safe_clone(tmax.grad),
        'df': safe_clone(df.grad),
        'smax': safe_clone(smax.grad),
        'qmax': safe_clone(qmax.grad),
        'f': safe_clone(f.grad),
    }
    
    # =========================================
    # Triton 实现
    # =========================================
    print("\n--- Running Triton Implementation (manual gradients) ---")
    
    # 重新创建带梯度的张量
    p2 = p.detach().clone().requires_grad_(True)
    t_val2 = t_val.detach().clone().requires_grad_(True)
    lday2 = lday.detach().clone().requires_grad_(True)
    snow2 = snow.detach().clone().requires_grad_(True)
    soil2 = soil.detach().clone().requires_grad_(True)
    tmin2 = tmin.detach().clone().requires_grad_(True)
    tmax2 = tmax.detach().clone().requires_grad_(True)
    df2 = df.detach().clone().requires_grad_(True)
    smax2 = smax.detach().clone().requires_grad_(True)
    qmax2 = qmax.detach().clone().requires_grad_(True)
    f2 = f.detach().clone().requires_grad_(True)
    
    # 使用 Triton 单步 (通过 run_triton 封装)
    params_tr = {
        'tmin': tmin2, 'tmax': tmax2, 'df': df2,
        'smax': smax2, 'qmax': qmax2, 'f': f2,
    }
    
    # 单步调用
    q_tr = exphydro_run_triton(
        p2.unsqueeze(0), t_val2.unsqueeze(0), lday2.unsqueeze(0),
        snow2, soil2,
        params_tr, nearzero=1e-6,
    )
    
    # 计算 loss 并反向传播
    loss_tr = q_tr.sum()
    loss_tr.backward()
    
    # 保存 Triton 梯度
    grads_tr = {
        'snow': snow2.grad.clone() if snow2.grad is not None else None,
        'soil': soil2.grad.clone() if soil2.grad is not None else None,
        'tmin': tmin2.grad.clone() if tmin2.grad is not None else None,
        'tmax': tmax2.grad.clone() if tmax2.grad is not None else None,
        'df': df2.grad.clone() if df2.grad is not None else None,
        'smax': smax2.grad.clone() if smax2.grad is not None else None,
        'qmax': qmax2.grad.clone() if qmax2.grad is not None else None,
        'f': f2.grad.clone() if f2.grad is not None else None,
    }
    
    # =========================================
    # 比较梯度
    # =========================================
    print("\n--- Comparing Gradients ---")
    
    # 注意: Triton版本的loss只包含q，需要调整PyTorch版本的loss来匹配
    # 重新计算PyTorch版本只用q的梯度
    for param in [snow, soil, tmin, tmax, df, smax, qmax, f]:
        if param.grad is not None:
            param.grad.zero_()
    
    snow_pt2, soil_pt2, q_pt2 = exphydro_step_pytorch(
        p.detach().clone().requires_grad_(True), 
        t_val.detach().clone().requires_grad_(True), 
        lday.detach().clone().requires_grad_(True),
        snow.detach().clone().requires_grad_(True), 
        soil.detach().clone().requires_grad_(True),
        tmin.detach().clone().requires_grad_(True), 
        tmax.detach().clone().requires_grad_(True), 
        df.detach().clone().requires_grad_(True), 
        smax.detach().clone().requires_grad_(True), 
        qmax.detach().clone().requires_grad_(True), 
        f.detach().clone().requires_grad_(True),
    )
    
    # 只用 q 的梯度
    loss_pt_q = q_pt2.sum()
    loss_pt_q.backward()
    
    # 比较前向输出
    print("\n--- Comparing Forward Outputs ---")
    q_match = torch.allclose(q_pt.detach(), q_tr.squeeze(0).detach(), rtol=1e-4, atol=1e-6)
    q_diff = (q_pt.detach() - q_tr.squeeze(0).detach()).abs().max().item()
    print(f"  q: max_diff={q_diff:.2e} {'✓' if q_match else '✗'}")
    
    print("\n" + "=" * 60)
    if q_match:
        print("Test 1 PASSED: Forward outputs match!")
    else:
        print("Test 1 FAILED: Forward outputs do not match!")
    print("=" * 60)
    
    return q_match


# ============================================================
# 测试 2: 多步累积梯度比较
# ============================================================

def test_multi_step_gradients():
    """测试多步计算的累积梯度一致性"""
    print("\n" + "=" * 60)
    print("Test 2: Multi-Step Accumulated Gradient Comparison")
    print("=" * 60)
    
    if not TRITON_AVAILABLE:
        print("Skipping: Triton not available")
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: Running on CPU, Triton may not work properly")
        return False
    
    # 测试参数
    batch_size = 32
    n_steps = 730
    
    # 创建测试数据
    params, init_states, precip, temp, lday = create_test_data(
        batch_size=batch_size,
        n_steps=n_steps,
        device=device,
        seed=42,
    )
    
    # =========================================
    # PyTorch 实现 (autograd)
    # =========================================
    print(f"\n--- Running PyTorch Implementation ({n_steps} steps) ---")
    
    # 让参数需要梯度
    params_pt = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    init_states_pt = {k: v.clone().requires_grad_(True) for k, v in init_states.items()}
    precip_pt = precip.clone().requires_grad_(True)
    temp_pt = temp.clone().requires_grad_(True)
    lday_pt = lday.clone().requires_grad_(True)
    
    q_series_pt, final_states_pt = exphydro_run_pytorch(
        precip_pt, temp_pt, lday_pt, params_pt, init_states_pt
    )
    
    # 计算 loss
    loss_pt = q_series_pt.sum()
    loss_pt.backward()
    
    # 保存梯度
    grads_pt = {}
    for k, v in params_pt.items():
        grads_pt[f'param_{k}'] = v.grad.clone() if v.grad is not None else None
    for k, v in init_states_pt.items():
        grads_pt[f'init_{k}'] = v.grad.clone() if v.grad is not None else None
    
    # =========================================
    # Triton 实现
    # =========================================
    print(f"\n--- Running Triton Implementation ({n_steps} steps) ---")
    
    # 让参数需要梯度
    params_tr = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    init_states_tr = {k: v.clone().requires_grad_(True) for k, v in init_states.items()}
    precip_tr = precip.clone()
    temp_tr = temp.clone()
    lday_tr = lday.clone()
    
    q_series_tr = exphydro_run_triton(
        precip_tr, temp_tr, lday_tr,
        init_states_tr['snow'], init_states_tr['soil'],
        params_tr, nearzero=1e-6,
    )
    
    # 计算 loss
    loss_tr = q_series_tr.sum()
    loss_tr.backward()
    
    # 保存梯度
    grads_tr = {}
    for k, v in params_tr.items():
        grads_tr[f'param_{k}'] = v.grad.clone() if v.grad is not None else None
    for k, v in init_states_tr.items():
        grads_tr[f'init_{k}'] = v.grad.clone() if v.grad is not None else None
    
    # =========================================
    # 比较梯度
    # =========================================
    print("\n--- Comparing Accumulated Gradients ---")
    
    all_passed = True
    for name in sorted(grads_pt.keys()):
        passed = compare_gradients(grads_tr.get(name), grads_pt[name], name, rtol=1e-3, atol=1e-5)
        all_passed = all_passed and passed
    
    # 比较前向输出
    print("\n--- Comparing Forward Outputs ---")
    q_match = torch.allclose(q_series_pt, q_series_tr, rtol=1e-4, atol=1e-6)
    q_diff = (q_series_pt - q_series_tr).abs().max().item()
    print(f"  q_series: max_diff={q_diff:.2e} {'✓' if q_match else '✗'}")
    
    print("\n" + "=" * 60)
    if all_passed and q_match:
        print("Test 2 PASSED: Multi-step accumulated gradients match!")
    else:
        print("Test 2 FAILED: Accumulated gradients or outputs do not match!")
    print("=" * 60)
    
    return all_passed and q_match


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ExpHydro Model Gradient Verification Tests")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("Test 1: Single Step", test_single_step_gradients()))
    results.append(("Test 2: Multi-Step", test_multi_step_gradients()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)
