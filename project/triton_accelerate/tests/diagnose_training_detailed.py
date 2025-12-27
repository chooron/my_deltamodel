"""
详细诊断训练不稳定问题

重点检查:
1. 梯度在多个 epoch 中的变化
2. 状态变量的演化
3. 参数更新的稳定性

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

hbv_step_triton = hbv_triton_core.hbv_step_triton
hbv_step_pytorch = hbv_triton_autograd.hbv_step_pytorch


def simulate_training_epoch(step_fn, params, forcing_data, n_epochs=5, lr=0.1):
    """
    模拟训练过程，监控梯度和状态变量
    
    Returns:
        dict: 包含每个 epoch 的统计信息
    """
    device = forcing_data['P'].device
    dtype = forcing_data['P'].dtype
    T, B, nmul = forcing_data['P'].shape
    shape = (B, nmul)
    
    # 复制参数以便修改
    params_train = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    
    epoch_stats = []
    
    for epoch in range(n_epochs):
        # 初始化状态
        snow = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        melt = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        sm = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        suz = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        slz = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
        
        q_list = []
        state_history = {'snow': [], 'sm': [], 'suz': [], 'slz': [], 'q': []}
        
        # 前向传播
        for t in range(T):
            snow, melt, sm, suz, slz, q = step_fn(
                forcing_data['P'][t], forcing_data['T'][t], forcing_data['PET'][t],
                snow, melt, sm, suz, slz,
                params_train['tt'], params_train['cfmax'], params_train['cfr'], params_train['cwh'],
                params_train['fc'], params_train['beta'], params_train['lp'], params_train['betaet'],
                params_train['c_par'],
                params_train['perc'], params_train['k0'], params_train['k1'], params_train['k2'],
                params_train['uzl'],
            )
            q_list.append(q)
            
            # 每隔一段时间记录状态
            if t % 50 == 0:
                state_history['snow'].append(snow.detach().mean().item())
                state_history['sm'].append(sm.detach().mean().item())
                state_history['suz'].append(suz.detach().mean().item())
                state_history['slz'].append(slz.detach().mean().item())
                state_history['q'].append(q.detach().mean().item())
        
        Q = torch.stack(q_list, dim=0)
        
        # 计算损失 (简化的 NSE loss)
        Q_mean = Q.mean()
        loss = ((Q - Q_mean) ** 2).sum() / (Q.numel() + 1e-8)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度统计
        grad_stats = {}
        for pname, pval in params_train.items():
            if pval.grad is not None:
                g = pval.grad
                grad_stats[pname] = {
                    'mean': g.mean().item(),
                    'std': g.std().item(),
                    'min': g.min().item(),
                    'max': g.max().item(),
                    'abs_max': g.abs().max().item(),
                    'has_nan': g.isnan().any().item(),
                    'has_inf': g.isinf().any().item(),
                }
        
        # 更新参数 (简单的 SGD)
        with torch.no_grad():
            for pname, pval in params_train.items():
                if pval.grad is not None:
                    # 梯度裁剪
                    grad_norm = pval.grad.norm()
                    if grad_norm > 1.0:
                        pval.grad.mul_(1.0 / grad_norm)
                    pval.sub_(lr * pval.grad)
                    pval.grad.zero_()
        
        epoch_stats.append({
            'epoch': epoch,
            'loss': loss.item(),
            'grad_stats': grad_stats,
            'state_history': state_history,
            'final_q_mean': Q[-100:].mean().item(),
            'final_q_std': Q[-100:].std().item(),
        })
        
        print(f"Epoch {epoch}: loss={loss.item():.6f}, Q_mean={Q[-100:].mean().item():.4f}")
    
    return epoch_stats


def compare_backends():
    """对比 Triton 和 PyTorch 后端的训练行为"""
    print("=" * 70)
    print("对比 Triton 和 PyTorch Autograd 后端的训练行为")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    # 创建测试数据
    T, B, nmul = 365, 20, 16
    shape = (B, nmul)
    
    torch.manual_seed(42)
    forcing_data = {
        'P': torch.rand(T, B, nmul, device=device, dtype=dtype) * 10,
        'T': torch.rand(T, B, nmul, device=device, dtype=dtype) * 20 - 5,
        'PET': torch.rand(T, B, nmul, device=device, dtype=dtype) * 5,
    }
    
    # 初始参数
    params_init = {
        'tt': torch.zeros(shape, device=device, dtype=dtype),
        'cfmax': torch.ones(shape, device=device, dtype=dtype) * 3.0,
        'cfr': torch.ones(shape, device=device, dtype=dtype) * 0.05,
        'cwh': torch.ones(shape, device=device, dtype=dtype) * 0.1,
        'fc': torch.ones(shape, device=device, dtype=dtype) * 200.0,
        'beta': torch.ones(shape, device=device, dtype=dtype) * 2.0,
        'lp': torch.ones(shape, device=device, dtype=dtype) * 0.7,
        'betaet': torch.ones(shape, device=device, dtype=dtype) * 1.5,
        'c_par': torch.ones(shape, device=device, dtype=dtype) * 0.05,
        'perc': torch.ones(shape, device=device, dtype=dtype) * 2.0,
        'k0': torch.ones(shape, device=device, dtype=dtype) * 0.2,
        'k1': torch.ones(shape, device=device, dtype=dtype) * 0.05,
        'k2': torch.ones(shape, device=device, dtype=dtype) * 0.01,
        'uzl': torch.ones(shape, device=device, dtype=dtype) * 20.0,
    }
    
    print("\n--- Triton 后端 ---")
    stats_triton = simulate_training_epoch(
        hbv_step_triton, 
        {k: v.clone() for k, v in params_init.items()},
        forcing_data,
        n_epochs=5,
        lr=0.01
    )
    
    print("\n--- PyTorch Autograd 后端 ---")
    stats_pytorch = simulate_training_epoch(
        hbv_step_pytorch,
        {k: v.clone() for k, v in params_init.items()},
        forcing_data,
        n_epochs=5,
        lr=0.01
    )
    
    # 对比分析
    print("\n" + "=" * 70)
    print("梯度对比分析")
    print("=" * 70)
    
    for epoch in range(5):
        print(f"\n--- Epoch {epoch} ---")
        for pname in ['fc', 'beta', 'k0', 'k1', 'k2']:
            gt = stats_triton[epoch]['grad_stats'].get(pname, {})
            gp = stats_pytorch[epoch]['grad_stats'].get(pname, {})
            
            if gt and gp:
                print(f"  {pname}:")
                print(f"    Triton:  abs_max={gt['abs_max']:.4e}, mean={gt['mean']:.4e}")
                print(f"    PyTorch: abs_max={gp['abs_max']:.4e}, mean={gp['mean']:.4e}")
                
                # 检查差异
                if abs(gt['abs_max'] - gp['abs_max']) / (gp['abs_max'] + 1e-8) > 0.1:
                    print(f"    WARNING: 梯度差异超过 10%!")


def check_gradient_accumulation_over_time():
    """检查梯度在长时间序列中的累积行为"""
    print("\n" + "=" * 70)
    print("检查梯度在长时间序列中的累积")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    B, nmul = 10, 16
    shape = (B, nmul)
    
    # 测试不同长度的时间序列
    for T in [100, 365, 730]:
        print(f"\n时间序列长度: T={T}")
        
        torch.manual_seed(42)
        P = torch.rand(T, B, nmul, device=device, dtype=dtype) * 10
        Temp = torch.rand(T, B, nmul, device=device, dtype=dtype) * 20 - 5
        PET = torch.rand(T, B, nmul, device=device, dtype=dtype) * 5
        
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
        
        for backend_name, step_fn in [("Triton", hbv_step_triton), ("PyTorch", hbv_step_pytorch)]:
            # 初始化状态
            snow = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
            melt = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
            sm = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
            suz = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
            slz = torch.zeros(shape, device=device, dtype=dtype) + 1e-5
            
            q_list = []
            for t in range(T):
                snow, melt, sm, suz, slz, q = step_fn(
                    P[t], Temp[t], PET[t], snow, melt, sm, suz, slz,
                    params['tt'], params['cfmax'], params['cfr'], params['cwh'],
                    params['fc'], params['beta'], params['lp'], params['betaet'], params['c_par'],
                    params['perc'], params['k0'], params['k1'], params['k2'], params['uzl'],
                )
                q_list.append(q)
            
            Q = torch.stack(q_list, dim=0)
            loss = Q.sum()
            loss.backward()
            
            # 检查关键参数的梯度
            g_fc = params['fc'].grad
            g_beta = params['beta'].grad
            g_k0 = params['k0'].grad
            
            print(f"  {backend_name}:")
            print(f"    fc:   abs_max={g_fc.abs().max().item():.4e}, has_nan={g_fc.isnan().any().item()}")
            print(f"    beta: abs_max={g_beta.abs().max().item():.4e}, has_nan={g_beta.isnan().any().item()}")
            print(f"    k0:   abs_max={g_k0.abs().max().item():.4e}, has_nan={g_k0.isnan().any().item()}")
            
            # 清理梯度
            for p in params.values():
                if p.grad is not None:
                    p.grad.zero_()


if __name__ == "__main__":
    print("HBV Triton 训练不稳定详细诊断")
    print("=" * 70)
    
    # 1. 对比两个后端
    compare_backends()
    
    # 2. 检查梯度累积
    check_gradient_accumulation_over_time()
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
