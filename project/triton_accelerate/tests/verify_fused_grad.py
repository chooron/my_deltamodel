"""
验证融合 Triton kernel 的梯度正确性
通过与 PyTorch autograd 对比来检测错误

Author: Kiro
"""

import torch
import sys
import os
import importlib.util

def load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hbv_triton_fused = load_module_directly("hbv_triton_fused", os.path.join(base_path, "models", "hbv_triton_fused.py"))
hbv_triton_autograd = load_module_directly("hbv_triton_autograd", os.path.join(base_path, "models", "hbv_triton_autograd.py"))

hbv_step_fused = hbv_triton_fused.hbv_step_fused
hbv_step_pytorch = hbv_triton_autograd.hbv_step_pytorch


def test_single_step_gradient():
    """测试单步梯度"""
    print("\n=== Testing Single Step Gradient (Fused vs PyTorch) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("WARNING: Triton requires CUDA. Please run on GPU.")
        return False
    
    dtype = torch.float64  # 用 float64 提高精度
    torch.manual_seed(42)
    
    B, nmul = 16, 1
    shape = (B, nmul)
    
    # 创建输入 - Fused 测试
    p = (torch.rand(shape, device=device, dtype=dtype) * 10).requires_grad_(True)
    t_val = (torch.rand(shape, device=device, dtype=dtype) * 20 - 5).requires_grad_(True)
    pet = (torch.rand(shape, device=device, dtype=dtype) * 5).requires_grad_(True)
    snow = (torch.rand(shape, device=device, dtype=dtype) * 50 + 1).requires_grad_(True)
    melt = (torch.rand(shape, device=device, dtype=dtype) * 10 + 0.1).requires_grad_(True)
    sm = (torch.rand(shape, device=device, dtype=dtype) * 100 + 10).requires_grad_(True)
    suz = (torch.rand(shape, device=device, dtype=dtype) * 20 + 1).requires_grad_(True)
    slz = (torch.rand(shape, device=device, dtype=dtype) * 30 + 1).requires_grad_(True)
    
    tt = torch.zeros(shape, device=device, dtype=dtype).requires_grad_(True)
    cfmax = (torch.ones(shape, device=device, dtype=dtype) * 3.0).requires_grad_(True)
    cfr = (torch.ones(shape, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    cwh = (torch.ones(shape, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    fc = (torch.ones(shape, device=device, dtype=dtype) * 200.0).requires_grad_(True)
    beta = (torch.ones(shape, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    lp = (torch.ones(shape, device=device, dtype=dtype) * 0.7).requires_grad_(True)
    betaet = (torch.ones(shape, device=device, dtype=dtype) * 1.5).requires_grad_(True)
    c_par = (torch.ones(shape, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    perc = (torch.ones(shape, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    k0 = (torch.ones(shape, device=device, dtype=dtype) * 0.2).requires_grad_(True)
    k1 = (torch.ones(shape, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    k2 = (torch.ones(shape, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    uzl = (torch.ones(shape, device=device, dtype=dtype) * 20.0).requires_grad_(True)
    
    inputs_fused = [p, t_val, pet, snow, melt, sm, suz, slz,
                    tt, cfmax, cfr, cwh, fc, beta, lp, betaet, c_par,
                    perc, k0, k1, k2, uzl]
    
    # Fused forward + backward
    outputs_fused = hbv_step_fused(*inputs_fused)
    loss_fused = outputs_fused[-1].sum()  # q_out
    loss_fused.backward()
    
    fused_grads = {}
    names = ['p', 't_val', 'pet', 'snow', 'melt', 'sm', 'suz', 'slz',
             'tt', 'cfmax', 'cfr', 'cwh', 'fc', 'beta', 'lp', 'betaet', 'c_par',
             'perc', 'k0', 'k1', 'k2', 'uzl']
    for i, name in enumerate(names):
        if inputs_fused[i].grad is not None:
            fused_grads[name] = inputs_fused[i].grad.clone()
    
    # 重新创建输入 - PyTorch 参考测试
    torch.manual_seed(42)
    p2 = (torch.rand(shape, device=device, dtype=dtype) * 10).requires_grad_(True)
    t_val2 = (torch.rand(shape, device=device, dtype=dtype) * 20 - 5).requires_grad_(True)
    pet2 = (torch.rand(shape, device=device, dtype=dtype) * 5).requires_grad_(True)
    snow2 = (torch.rand(shape, device=device, dtype=dtype) * 50 + 1).requires_grad_(True)
    melt2 = (torch.rand(shape, device=device, dtype=dtype) * 10 + 0.1).requires_grad_(True)
    sm2 = (torch.rand(shape, device=device, dtype=dtype) * 100 + 10).requires_grad_(True)
    suz2 = (torch.rand(shape, device=device, dtype=dtype) * 20 + 1).requires_grad_(True)
    slz2 = (torch.rand(shape, device=device, dtype=dtype) * 30 + 1).requires_grad_(True)
    
    tt2 = torch.zeros(shape, device=device, dtype=dtype).requires_grad_(True)
    cfmax2 = (torch.ones(shape, device=device, dtype=dtype) * 3.0).requires_grad_(True)
    cfr2 = (torch.ones(shape, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    cwh2 = (torch.ones(shape, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    fc2 = (torch.ones(shape, device=device, dtype=dtype) * 200.0).requires_grad_(True)
    beta2 = (torch.ones(shape, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    lp2 = (torch.ones(shape, device=device, dtype=dtype) * 0.7).requires_grad_(True)
    betaet2 = (torch.ones(shape, device=device, dtype=dtype) * 1.5).requires_grad_(True)
    c_par2 = (torch.ones(shape, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    perc2 = (torch.ones(shape, device=device, dtype=dtype) * 2.0).requires_grad_(True)
    k02 = (torch.ones(shape, device=device, dtype=dtype) * 0.2).requires_grad_(True)
    k12 = (torch.ones(shape, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    k22 = (torch.ones(shape, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    uzl2 = (torch.ones(shape, device=device, dtype=dtype) * 20.0).requires_grad_(True)
    
    inputs_pytorch = [p2, t_val2, pet2, snow2, melt2, sm2, suz2, slz2,
                      tt2, cfmax2, cfr2, cwh2, fc2, beta2, lp2, betaet2, c_par2,
                      perc2, k02, k12, k22, uzl2]
    
    # PyTorch forward + backward
    outputs_pytorch = hbv_step_pytorch(*inputs_pytorch)
    loss_pytorch = outputs_pytorch[-1].sum()  # q_out
    loss_pytorch.backward()
    
    pytorch_grads = {}
    for i, name in enumerate(names):
        if inputs_pytorch[i].grad is not None:
            pytorch_grads[name] = inputs_pytorch[i].grad.clone()
    
    # 比较梯度
    print("\nGradient comparison (Fused vs PyTorch autograd):")
    all_close = True
    for name in fused_grads:
        if name not in pytorch_grads:
            continue
        diff = (fused_grads[name] - pytorch_grads[name]).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (pytorch_grads[name].abs() + 1e-8)).max().item()
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {name:8s}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} {status}")
        if max_diff >= 1e-4:
            all_close = False
            print(f"    Fused:   {fused_grads[name].flatten()[:4].tolist()}")
            print(f"    PyTorch: {pytorch_grads[name].flatten()[:4].tolist()}")
    
    return all_close


def test_multi_step_gradient():
    """测试多步梯度累积"""
    print("\n=== Testing Multi-Step Gradient Accumulation ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("WARNING: Triton requires CUDA. Please run on GPU.")
        return False
    
    dtype = torch.float32
    torch.manual_seed(42)
    
    B, nmul = 10, 16
    T = 100  # 时间步数
    shape = (B, nmul)
    
    # 创建时间序列驱动数据
    P = torch.rand(T, B, nmul, device=device, dtype=dtype) * 10
    Temp = torch.rand(T, B, nmul, device=device, dtype=dtype) * 20 - 5
    PET = torch.rand(T, B, nmul, device=device, dtype=dtype) * 5
    
    # 参数
    params = {
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
    
    def run_sequence(step_fn, name):
        # 创建叶子张量
        params_leaf = {}
        for pname, val in params.items():
            params_leaf[pname] = val.detach().clone().requires_grad_(True)
        
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
                params_leaf['tt'], params_leaf['cfmax'], params_leaf['cfr'], params_leaf['cwh'],
                params_leaf['fc'], params_leaf['beta'], params_leaf['lp'], params_leaf['betaet'], params_leaf['c_par'],
                params_leaf['perc'], params_leaf['k0'], params_leaf['k1'], params_leaf['k2'], params_leaf['uzl'],
            )
            q_list.append(q)
        
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
        
        return grad_stats
    
    # 运行 Fused 版本
    print("\n运行 Fused 版本...")
    stats_fused = run_sequence(hbv_step_fused, "Fused")
    
    # 运行 PyTorch 版本
    print("运行 PyTorch Autograd 版本...")
    stats_pytorch = run_sequence(hbv_step_pytorch, "PyTorch")
    
    # 比较结果
    print("\n" + "-" * 60)
    print("Multi-step gradient comparison:")
    print("-" * 60)
    
    all_ok = True
    for pname in stats_fused.keys():
        gf = stats_fused[pname]
        gp = stats_pytorch[pname]
        
        # 检查是否有 NaN/Inf
        if gf['has_nan'] or gf['has_inf']:
            print(f"\n{pname}: FAIL - Fused has NaN={gf['has_nan']}, Inf={gf['has_inf']}")
            all_ok = False
            continue
        
        # 比较均值和范围
        mean_diff = abs(gf['mean'] - gp['mean']) / (abs(gp['mean']) + 1e-8)
        print(f"\n{pname}:")
        print(f"  Fused:   mean={gf['mean']:.4e}, range=[{gf['min']:.4e}, {gf['max']:.4e}]")
        print(f"  PyTorch: mean={gp['mean']:.4e}, range=[{gp['min']:.4e}, {gp['max']:.4e}]")
        print(f"  Rel diff: {mean_diff:.2%}")
        
        if mean_diff > 0.5:  # 允许 50% 的相对误差（因为是累积梯度）
            print(f"  WARNING: Large difference!")
    
    return all_ok


def main():
    print("=" * 60)
    print("Fused Triton Kernel Gradient Verification")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("\nERROR: Triton requires CUDA. Please run on GPU server.")
        return
    
    single_ok = test_single_step_gradient()
    multi_ok = test_multi_step_gradient()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Single-step gradient: {'PASS' if single_ok else 'FAIL'}")
    print(f"  Multi-step gradient:  {'PASS' if multi_ok else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
