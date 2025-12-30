"""
诊断训练不稳定问题的脚本

分析:
1. 梯度流 - 检查梯度是否爆炸/消失
2. 状态变量 - 检查 snow, sm, suz, slz 是否保持物理合理值
3. 对比 Triton 和 Autograd 后端的差异

Author: Kiro
"""

import os
import sys
import torch
import numpy as np
import importlib.util

# 直接导入模块，绕过 __init__.py 的 hydra 依赖
def load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), "models")

# 直接加载模块
hbv_triton_core = load_module_directly("hbv_triton_core", os.path.join(models_dir, "hbv_triton_core.py"))
hbv_triton_autograd = load_module_directly("hbv_triton_autograd", os.path.join(models_dir, "hbv_triton_autograd.py"))

# 导入需要的函数和类
SnowBlockTriton = hbv_triton_core.SnowBlockTriton
SoilBlockTriton = hbv_triton_core.SoilBlockTriton
RoutingBlockTriton = hbv_triton_core.RoutingBlockTriton
hbv_step_triton = hbv_triton_core.hbv_step_triton
hbv_step_pytorch = hbv_triton_autograd.hbv_step_pytorch


def analyze_single_step_gradients():
    """分析单步计算的梯度差异"""
    print("=" * 60)
    print("分析单步梯度差异 (Triton vs PyTorch Autograd)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    # 检查是否有 GPU
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        print("WARNING: 没有检测到 GPU，Triton 测试将被跳过")
        print("请在有 GPU 的服务器上运行此脚本以获得完整的对比分析")
    
    # 创建测试数据 - 使用物理合理的值
    B, nmul = 100, 16
    shape = (B, nmul)
    
    # 驱动数据
    p = torch.rand(shape, device=device, dtype=dtype) * 10  # 降水 0-10mm
    t_val = torch.rand(shape, device=device, dtype=dtype) * 20 - 5  # 温度 -5 to 15°C
    pet = torch.rand(shape, device=device, dtype=dtype) * 5  # PET 0-5mm
    
    # 状态变量 - 初始化为小的正值
    snow = torch.rand(shape, device=device, dtype=dtype) * 50 + 1
    melt = torch.rand(shape, device=device, dtype=dtype) * 10 + 0.1
    sm = torch.rand(shape, device=device, dtype=dtype) * 100 + 10
    suz = torch.rand(shape, device=device, dtype=dtype) * 20 + 1
    slz = torch.rand(shape, device=device, dtype=dtype) * 30 + 1
    
    # 参数
    tt = torch.zeros(shape, device=device, dtype=dtype)
    cfmax = torch.ones(shape, device=device, dtype=dtype) * 3.0
    cfr = torch.ones(shape, device=device, dtype=dtype) * 0.05
    cwh = torch.ones(shape, device=device, dtype=dtype) * 0.1
    fc = torch.ones(shape, device=device, dtype=dtype) * 200.0
    beta = torch.ones(shape, device=device, dtype=dtype) * 2.0
    lp = torch.ones(shape, device=device, dtype=dtype) * 0.7
    betaet = torch.ones(shape, device=device, dtype=dtype) * 1.5
    c_par = torch.ones(shape, device=device, dtype=dtype) * 0.05
    perc = torch.ones(shape, device=device, dtype=dtype) * 2.0
    k0 = torch.ones(shape, device=device, dtype=dtype) * 0.2
    k1 = torch.ones(shape, device=device, dtype=dtype) * 0.05
    k2 = torch.ones(shape, device=device, dtype=dtype) * 0.01
    uzl = torch.ones(shape, device=device, dtype=dtype) * 20.0
    
    grads_triton = None
    
    # ========== Triton 版本 (仅在有 GPU 时运行) ==========
    if has_gpu:
        inputs_triton = [p.clone(), t_val.clone(), pet.clone(), 
                         snow.clone(), melt.clone(), sm.clone(), suz.clone(), slz.clone(),
                         tt.clone(), cfmax.clone(), cfr.clone(), cwh.clone(),
                         fc.clone(), beta.clone(), lp.clone(), betaet.clone(), c_par.clone(),
                         perc.clone(), k0.clone(), k1.clone(), k2.clone(), uzl.clone()]
        for x in inputs_triton:
            x.requires_grad_(True)
        
        snow_out_t, melt_out_t, sm_out_t, suz_out_t, slz_out_t, q_out_t = hbv_step_triton(*inputs_triton)
        loss_t = q_out_t.sum()
        loss_t.backward()
        
        grads_triton = {
            'p': inputs_triton[0].grad.clone(),
            't_val': inputs_triton[1].grad.clone(),
            'snow': inputs_triton[3].grad.clone(),
            'sm': inputs_triton[5].grad.clone(),
            'fc': inputs_triton[12].grad.clone(),
            'beta': inputs_triton[13].grad.clone(),
        }
    
    # ========== PyTorch Autograd 版本 ==========
    inputs_pytorch = [p.clone(), t_val.clone(), pet.clone(),
                      snow.clone(), melt.clone(), sm.clone(), suz.clone(), slz.clone(),
                      tt.clone(), cfmax.clone(), cfr.clone(), cwh.clone(),
                      fc.clone(), beta.clone(), lp.clone(), betaet.clone(), c_par.clone(),
                      perc.clone(), k0.clone(), k1.clone(), k2.clone(), uzl.clone()]
    for x in inputs_pytorch:
        x.requires_grad_(True)
    
    snow_out_p, melt_out_p, sm_out_p, suz_out_p, slz_out_p, q_out_p = hbv_step_pytorch(*inputs_pytorch)
    loss_p = q_out_p.sum()
    loss_p.backward()
    
    grads_pytorch = {
        'p': inputs_pytorch[0].grad.clone(),
        't_val': inputs_pytorch[1].grad.clone(),
        'snow': inputs_pytorch[3].grad.clone(),
        'sm': inputs_pytorch[5].grad.clone(),
        'fc': inputs_pytorch[12].grad.clone(),
        'beta': inputs_pytorch[13].grad.clone(),
    }
    
    # ========== 输出结果 ==========
    print("\nPyTorch Autograd 梯度统计 (关键参数):")
    print("-" * 60)
    for name, g_p in grads_pytorch.items():
        print(f"\n{name}:")
        print(f"  PyTorch: mean={g_p.mean().item():.6f}, std={g_p.std().item():.6f}, "
              f"min={g_p.min().item():.6f}, max={g_p.max().item():.6f}")
        
        if grads_triton is not None:
            g_t = grads_triton[name]
            abs_diff = (g_t - g_p).abs()
            rel_diff = abs_diff / (g_p.abs() + 1e-8)
            print(f"  Triton:  mean={g_t.mean().item():.6f}, std={g_t.std().item():.6f}, "
                  f"min={g_t.min().item():.6f}, max={g_t.max().item():.6f}")
            print(f"  Diff:    max_abs={abs_diff.max().item():.6f}, max_rel={rel_diff.max().item():.6f}")
            
            if g_t.isnan().any():
                print(f"  WARNING: Triton gradient has NaN!")
        
        if g_p.isnan().any():
            print(f"  WARNING: PyTorch gradient has NaN!")


def analyze_multi_step_gradient_accumulation():
    """分析多步计算中的梯度累积"""
    print("\n" + "=" * 60)
    print("分析多步梯度累积 (模拟训练过程)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    has_gpu = torch.cuda.is_available()
    
    if not has_gpu:
        print("WARNING: 没有检测到 GPU，仅运行 PyTorch Autograd 测试")
    
    B, nmul = 50, 16
    T = 365  # 模拟一年的时间步
    shape = (B, nmul)
    
    # 创建时间序列驱动数据
    torch.manual_seed(42)
    P = torch.rand(T, B, nmul, device=device, dtype=dtype) * 10
    Temp = torch.rand(T, B, nmul, device=device, dtype=dtype) * 20 - 5
    PET = torch.rand(T, B, nmul, device=device, dtype=dtype) * 5
    
    # 参数 (需要梯度)
    params = {
        'tt': torch.zeros(shape, device=device, dtype=dtype, requires_grad=True),
        'cfmax': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 3.0,
        'cfr': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.05,
        'cwh': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.1,
        'fc': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 200.0,
        'beta': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 2.0,
        'lp': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.7,
        'betaet': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 1.5,
        'c_par': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.05,
        'perc': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 2.0,
        'k0': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.2,
        'k1': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.05,
        'k2': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 0.01,
        'uzl': torch.ones(shape, device=device, dtype=dtype, requires_grad=True) * 20.0,
    }
    
    # 创建叶子张量版本的参数（避免 non-leaf tensor 警告）
    params_leaf = {}
    for name, val in params.items():
        params_leaf[name] = val.detach().clone().requires_grad_(True)
    
    def run_sequence(step_fn, name):
        """运行时间序列并收集统计信息"""
        # 初始化状态
        snow = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        melt = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        sm = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        suz = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        slz = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        
        q_list = []
        state_stats = {'snow': [], 'sm': [], 'suz': [], 'slz': []}
        
        for t in range(T):
            snow, melt, sm, suz, slz, q = step_fn(
                P[t], Temp[t], PET[t], snow, melt, sm, suz, slz,
                params_leaf['tt'], params_leaf['cfmax'], params_leaf['cfr'], params_leaf['cwh'],
                params_leaf['fc'], params_leaf['beta'], params_leaf['lp'], params_leaf['betaet'], params_leaf['c_par'],
                params_leaf['perc'], params_leaf['k0'], params_leaf['k1'], params_leaf['k2'], params_leaf['uzl'],
            )
            q_list.append(q)
            
            # 记录状态统计
            state_stats['snow'].append(snow.detach().mean().item())
            state_stats['sm'].append(sm.detach().mean().item())
            state_stats['suz'].append(suz.detach().mean().item())
            state_stats['slz'].append(slz.detach().mean().item())
        
        Q = torch.stack(q_list, dim=0)
        loss = Q.sum()
        loss.backward()
        
        # 收集梯度统计
        grad_stats = {}
        for pname, pval in params_leaf.items():
            if pval.grad is not None:
                g = pval.grad
                grad_stats[pname] = {
                    'mean': g.mean().item(),
                    'std': g.std().item(),
                    'min': g.min().item(),
                    'max': g.max().item(),
                    'has_nan': g.isnan().any().item(),
                    'has_inf': g.isinf().any().item(),
                }
                pval.grad.zero_()
        
        return grad_stats, state_stats
    
    # 运行 Triton 版本 (仅在有 GPU 时)
    stats_triton = None
    if has_gpu:
        print("\n运行 Triton 版本...")
        stats_triton, state_stats_t = run_sequence(hbv_step_triton, "Triton")
    
    # 重置参数梯度
    for p in params_leaf.values():
        if p.grad is not None:
            p.grad.zero_()
    
    # 运行 PyTorch 版本
    print("运行 PyTorch Autograd 版本...")
    stats_pytorch, state_stats_p = run_sequence(hbv_step_pytorch, "PyTorch")
    
    # 打印结果
    print("\n" + "-" * 60)
    print("PyTorch Autograd 梯度统计:")
    print("-" * 60)
    
    for pname in stats_pytorch.keys():
        gp = stats_pytorch[pname]
        print(f"\n{pname}:")
        print(f"  PyTorch: mean={gp['mean']:.4e}, std={gp['std']:.4e}, "
              f"range=[{gp['min']:.4e}, {gp['max']:.4e}]")
        
        if stats_triton is not None:
            gt = stats_triton[pname]
            print(f"  Triton:  mean={gt['mean']:.4e}, std={gt['std']:.4e}, "
                  f"range=[{gt['min']:.4e}, {gt['max']:.4e}]")
            if gt['has_nan'] or gt['has_inf']:
                print(f"  WARNING: Triton has NaN={gt['has_nan']}, Inf={gt['has_inf']}")
        
        if gp['has_nan'] or gp['has_inf']:
            print(f"  WARNING: PyTorch has NaN={gp['has_nan']}, Inf={gp['has_inf']}")
    
    # 状态变量分析
    print("\n" + "-" * 60)
    print("状态变量演化 (最后100步的统计):")
    print("-" * 60)
    
    for state_name in ['snow', 'sm', 'suz', 'slz']:
        vals_p = state_stats_p[state_name][-100:]
        print(f"\n{state_name}:")
        print(f"  PyTorch: mean={np.mean(vals_p):.4f}, std={np.std(vals_p):.4f}, "
              f"range=[{np.min(vals_p):.4f}, {np.max(vals_p):.4f}]")
        
        if stats_triton is not None:
            vals_t = state_stats_t[state_name][-100:]
            print(f"  Triton:  mean={np.mean(vals_t):.4f}, std={np.std(vals_t):.4f}, "
                  f"range=[{np.min(vals_t):.4f}, {np.max(vals_t):.4f}]")


def check_gradient_clipping_effect():
    """检查梯度裁剪的效果"""
    print("\n" + "=" * 60)
    print("检查梯度裁剪效果")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    has_gpu = torch.cuda.is_available()
    
    if not has_gpu:
        print("WARNING: 没有检测到 GPU，Triton 测试将被跳过")
        print("仅运行 PyTorch Autograd 测试")
    
    # 创建可能导致极端梯度的输入
    B, nmul = 10, 16
    shape = (B, nmul)
    
    # 极端情况: 非常小的 fc 值可能导致除法问题
    test_cases = [
        ("正常情况", {"fc": 200.0, "sm": 50.0, "beta": 2.0}),
        ("小 fc", {"fc": 51.0, "sm": 50.0, "beta": 2.0}),  # fc 接近下界
        ("大 beta", {"fc": 200.0, "sm": 50.0, "beta": 5.9}),  # beta 接近上界
        ("sm 接近 fc", {"fc": 100.0, "sm": 99.0, "beta": 2.0}),  # 土壤接近饱和
    ]
    
    for case_name, case_params in test_cases:
        print(f"\n测试: {case_name}")
        
        # 创建输入
        torch.manual_seed(42)  # 确保可重复性
        p = torch.rand(shape, device=device, dtype=dtype) * 10
        t_val = torch.rand(shape, device=device, dtype=dtype) * 20 - 5
        pet = torch.rand(shape, device=device, dtype=dtype) * 5
        snow = torch.rand(shape, device=device, dtype=dtype) * 50 + 1
        melt = torch.rand(shape, device=device, dtype=dtype) * 10 + 0.1
        sm = torch.ones(shape, device=device, dtype=dtype) * case_params["sm"]
        suz = torch.rand(shape, device=device, dtype=dtype) * 20 + 1
        slz = torch.rand(shape, device=device, dtype=dtype) * 30 + 1
        
        tt = torch.zeros(shape, device=device, dtype=dtype)
        cfmax = torch.ones(shape, device=device, dtype=dtype) * 3.0
        cfr = torch.ones(shape, device=device, dtype=dtype) * 0.05
        cwh = torch.ones(shape, device=device, dtype=dtype) * 0.1
        fc = torch.ones(shape, device=device, dtype=dtype) * case_params["fc"]
        beta = torch.ones(shape, device=device, dtype=dtype) * case_params["beta"]
        lp = torch.ones(shape, device=device, dtype=dtype) * 0.7
        betaet = torch.ones(shape, device=device, dtype=dtype) * 1.5
        c_par = torch.ones(shape, device=device, dtype=dtype) * 0.05
        perc = torch.ones(shape, device=device, dtype=dtype) * 2.0
        k0 = torch.ones(shape, device=device, dtype=dtype) * 0.2
        k1 = torch.ones(shape, device=device, dtype=dtype) * 0.05
        k2 = torch.ones(shape, device=device, dtype=dtype) * 0.01
        uzl = torch.ones(shape, device=device, dtype=dtype) * 20.0
        
        inputs = [p, t_val, pet, snow, melt, sm, suz, slz,
                  tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                  perc, k0, k1, k2, uzl]
        for x in inputs:
            x.requires_grad_(True)
        
        # Triton (仅在有 GPU 时运行)
        if has_gpu:
            try:
                outputs_t = hbv_step_triton(*inputs)
                loss_t = outputs_t[-1].sum()
                loss_t.backward()
                
                # 检查关键梯度
                g_fc = inputs[12].grad
                g_beta = inputs[13].grad
                
                print(f"  Triton - fc grad: mean={g_fc.mean().item():.4e}, max={g_fc.abs().max().item():.4e}")
                print(f"  Triton - beta grad: mean={g_beta.mean().item():.4e}, max={g_beta.abs().max().item():.4e}")
                
                if g_fc.isnan().any() or g_beta.isnan().any():
                    print("  WARNING: NaN detected in Triton gradients!")
                if g_fc.abs().max() > 1e5 or g_beta.abs().max() > 1e5:
                    print("  WARNING: Large gradients detected in Triton!")
                    
            except Exception as e:
                print(f"  Triton ERROR: {e}")
            
            # 清理梯度
            for x in inputs:
                if x.grad is not None:
                    x.grad.zero_()
        
        # PyTorch Autograd
        try:
            outputs_p = hbv_step_pytorch(*inputs)
            loss_p = outputs_p[-1].sum()
            loss_p.backward()
            
            # 检查关键梯度
            g_fc = inputs[12].grad
            g_beta = inputs[13].grad
            
            print(f"  PyTorch - fc grad: mean={g_fc.mean().item():.4e}, max={g_fc.abs().max().item():.4e}")
            print(f"  PyTorch - beta grad: mean={g_beta.mean().item():.4e}, max={g_beta.abs().max().item():.4e}")
            
            if g_fc.isnan().any() or g_beta.isnan().any():
                print("  WARNING: NaN detected in PyTorch gradients!")
            if g_fc.abs().max() > 1e5 or g_beta.abs().max() > 1e5:
                print("  WARNING: Large gradients detected in PyTorch!")
                
        except Exception as e:
            print(f"  PyTorch ERROR: {e}")
        
        # 清理梯度
        for x in inputs:
            if x.grad is not None:
                x.grad.zero_()


if __name__ == "__main__":
    print("HBV Triton 训练不稳定诊断")
    print("=" * 60)
    
    # 1. 单步梯度分析
    analyze_single_step_gradients()
    
    # 2. 多步梯度累积分析
    analyze_multi_step_gradient_accumulation()
    
    # 3. 梯度裁剪效果检查
    check_gradient_clipping_effect()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
