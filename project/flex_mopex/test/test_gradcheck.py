"""
梯度检查测试：验证 MOPEX 模型的梯度计算正确性

使用 torch.autograd.gradcheck 对比数值梯度和自动微分梯度
"""

import sys
import torch
import numpy as np
sys.path.append("/workspace/my_deltamodel")
from project.flex_mopex.models.mopex_core import mopex_step, MOPEX_PARAMS_BOUNDS


class MopexStepWrapper(torch.nn.Module):
    """
    包装 mopex_step 函数以便进行梯度检查
    
    将所有可学习参数打包为一个输入张量，便于 gradcheck 测试
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        inputs,  # [P, T, PET, doy]
        weights,  # [w_phen, w_int, w_snow, w_sub]
        params,  # [Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax]
        states,  # [S1, S2, Sc1, Sc2, Sn]
    ):
        """
        Args:
            inputs: [4] - 气象驱动
            weights: [4] - 结构权重
            params: [12] - 模型参数
            states: [5] - 状态变量
            
        Returns:
            output: [2] - [Q_total, ET_total] (只返回关键输出用于梯度检查)
        """
        # 解包输入
        P, T, PET, doy = inputs[0], inputs[1], inputs[2], inputs[3]
        w_phen, w_int, w_snow, w_sub = weights[0], weights[1], weights[2], weights[3]
        
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax = (
            params[0], params[1], params[2], params[3], params[4], params[5],
            params[6], params[7], params[8], params[9], params[10], params[11]
        )
        
        S1, S2, Sc1, Sc2, Sn = states[0], states[1], states[2], states[3], states[4]
        
        # 调用 mopex_step
        Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new = mopex_step(
            P=P, T=T, PET=PET, doy=doy,
            w_phen=w_phen, w_int=w_int, w_snow=w_snow, w_sub=w_sub,
            Sb1=Sb1, tw=tw, tu=tu, Se=Se, tc=tc, ddf=ddf, tcrit=tcrit,
            Sb2=Sb2, alpha=alpha, is_time=is_time, tmin=tmin, tmax=tmax,
            S1=S1, S2=S2, Sc1=Sc1, Sc2=Sc2, Sn=Sn,
        )
        
        # 返回关键输出（Q 和 ET）
        return torch.stack([Q_total, ET_total])


def generate_test_inputs(device="cpu", dtype=torch.float64):
    """
    生成测试输入（使用 float64 以提高数值精度）
    
    Returns:
        inputs, weights, params, states (所有张量 requires_grad=True)
    """
    # 气象输入 [P, T, PET, doy]
    inputs = torch.tensor([
        25.0,   # P: 降雨 25mm
        15.0,   # T: 温度 15°C
        5.0,    # PET: 5mm
        180.0,  # doy: 第180天
    ], device=device, dtype=dtype, requires_grad=True)
    
    # 结构权重 [w_phen, w_int, w_snow, w_sub]
    weights = torch.tensor([
        0.5, 0.3, 0.7, 0.4
    ], device=device, dtype=dtype, requires_grad=True)
    
    # 模型参数（使用中等值）
    params = torch.tensor([
        25.0,   # Sb1
        2.5,    # tw
        1000.0, # tu
        500.0,  # Se
        15.0,   # tc
        10.0,   # ddf
        0.0,    # tcrit
        750.0,  # Sb2
        0.5,    # alpha
        180.0,  # is_time
        0.0,    # tmin
        20.0,   # tmax
    ], device=device, dtype=dtype, requires_grad=True)
    
    # 状态变量 [S1, S2, Sc1, Sc2, Sn]
    states = torch.tensor([
        10.0,  # S1
        50.0,  # S2
        5.0,   # Sc1
        25.0,  # Sc2
        20.0,  # Sn
    ], device=device, dtype=dtype, requires_grad=True)
    
    return inputs, weights, params, states


def test_gradcheck_single(eps=1e-6, atol=1e-5, rtol=1e-3):
    """
    单次梯度检查测试
    
    Args:
        eps: 数值梯度的扰动大小
        atol: 绝对容差
        rtol: 相对容差
    """
    print("\n" + "=" * 80)
    print("单次梯度检查测试")
    print("=" * 80)
    
    # 创建包装器
    model = MopexStepWrapper()
    
    # 生成测试输入
    inputs, weights, params, states = generate_test_inputs(dtype=torch.float64)
    
    print(f"输入形状: {inputs.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"参数形状: {params.shape}")
    print(f"状态形状: {states.shape}")
    print(f"\n数值梯度参数:")
    print(f"  eps: {eps}")
    print(f"  atol: {atol}")
    print(f"  rtol: {rtol}")
    
    # 运行梯度检查
    print("\n执行梯度检查...")
    test_passed = torch.autograd.gradcheck(
        model,
        (inputs, weights, params, states),
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=False,
    )
    
    if test_passed:
        print("✓ 梯度检查通过！")
    else:
        print("✗ 梯度检查失败！")
        print("\n尝试更详细的检查...")
        
        # 单独检查每个输入的梯度
        print("\n分别测试各输入的梯度:")
        
        # 测试气象输入
        print("  - 气象输入 (P, T, PET, doy)...", end=" ")
        inputs_copy = inputs.detach().clone().requires_grad_(True)
        test1 = torch.autograd.gradcheck(
            lambda x: model(x, weights.detach(), params.detach(), states.detach()),
            inputs_copy,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False,
        )
        print("✓" if test1 else "✗")
        
        # 测试权重
        print("  - 结构权重 (w_phen, w_int, w_snow, w_sub)...", end=" ")
        weights_copy = weights.detach().clone().requires_grad_(True)
        test2 = torch.autograd.gradcheck(
            lambda x: model(inputs.detach(), x, params.detach(), states.detach()),
            weights_copy,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False,
        )
        print("✓" if test2 else "✗")
        
        # 测试参数
        print("  - 模型参数 (Sb1, tw, ...)...", end=" ")
        params_copy = params.detach().clone().requires_grad_(True)
        test3 = torch.autograd.gradcheck(
            lambda x: model(inputs.detach(), weights.detach(), x, states.detach()),
            params_copy,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False,
        )
        print("✓" if test3 else "✗")
        
        # 测试状态
        print("  - 状态变量 (S1, S2, ...)...", end=" ")
        states_copy = states.detach().clone().requires_grad_(True)
        test4 = torch.autograd.gradcheck(
            lambda x: model(inputs.detach(), weights.detach(), params.detach(), x),
            states_copy,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False,
        )
        print("✓" if test4 else "✗")
    
    return test_passed


def test_gradcheck_random_samples(num_samples=10, eps=1e-6, atol=1e-5, rtol=1e-3):
    """
    多组随机样本的梯度检查
    
    Args:
        num_samples: 测试样本数量
        eps: 数值梯度的扰动大小
        atol: 绝对容差
        rtol: 相对容差
    """
    print("\n" + "=" * 80)
    print(f"随机样本梯度检查测试 (N={num_samples})")
    print("=" * 80)
    
    model = MopexStepWrapper()
    passed_count = 0
    
    for i in range(num_samples):
        print(f"\n测试样本 {i+1}/{num_samples}...", end=" ")
        
        # 生成随机输入
        inputs = torch.tensor([
            torch.rand(1).item() * 50.0,  # P: 0-50mm
            torch.rand(1).item() * 40.0 - 10.0,  # T: -10-30°C
            torch.rand(1).item() * 10.0,  # PET: 0-10mm
            torch.rand(1).item() * 365.0,  # doy: 0-365
        ], dtype=torch.float64, requires_grad=True)
        
        weights = torch.rand(4, dtype=torch.float64, requires_grad=True)
        
        # 随机参数（在合理范围内）
        params = torch.tensor([
            torch.rand(1).item() * 49.99 + 0.01,  # Sb1: [0.01, 50]
            torch.rand(1).item() * 4.99 + 0.01,   # tw: [0.01, 5]
            torch.rand(1).item() * 1999.0 + 1.0,  # tu: [1, 2000]
            torch.rand(1).item() * 999.0 + 1.0,   # Se: [1, 1000]
            torch.rand(1).item() * 29.9 + 0.1,    # tc: [0.1, 30]
            torch.rand(1).item() * 20.0,          # ddf: [0, 20]
            torch.rand(1).item() * 6.0 - 3.0,     # tcrit: [-3, 3]
            torch.rand(1).item() * 1499.0 + 1.0,  # Sb2: [1, 1500]
            torch.rand(1).item(),                 # alpha: [0, 1]
            torch.rand(1).item() * 365.0,         # is_time: [0, 365]
            torch.rand(1).item() * 15.0 - 10.0,   # tmin: [-10, 5]
            torch.rand(1).item() * 25.0 + 5.0,    # tmax: [5, 30]
        ], dtype=torch.float64, requires_grad=True)
        
        states = torch.tensor([
            torch.rand(1).item() * 20.0,   # S1
            torch.rand(1).item() * 100.0,  # S2
            torch.rand(1).item() * 10.0,   # Sc1
            torch.rand(1).item() * 50.0,   # Sc2
            torch.rand(1).item() * 100.0,  # Sn
        ], dtype=torch.float64, requires_grad=True)
        
        # 梯度检查
        test_passed = torch.autograd.gradcheck(
            model,
            (inputs, weights, params, states),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False,
        )
        
        if test_passed:
            print("✓")
            passed_count += 1
        else:
            print("✗")
    
    # 统计结果
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总样本数: {num_samples}")
    print(f"通过数: {passed_count}")
    print(f"失败数: {num_samples - passed_count}")
    print(f"通过率: {passed_count / num_samples * 100:.2f}%")
    
    if passed_count == num_samples:
        print("\n✓ 所有梯度检查通过！")
    else:
        print(f"\n✗ {num_samples - passed_count} 个样本梯度检查失败")
    
    return passed_count == num_samples


def test_backward_pass():
    """
    测试反向传播是否正常工作
    """
    print("\n" + "=" * 80)
    print("反向传播测试")
    print("=" * 80)
    
    model = MopexStepWrapper()
    inputs, weights, params, states = generate_test_inputs(dtype=torch.float32)
    
    # 前向传播
    output = model(inputs, weights, params, states)
    print(f"输出: Q={output[0].item():.4f}, ET={output[1].item():.4f}")
    
    # 计算损失（简单求和）
    loss = output.sum()
    print(f"损失: {loss.item():.4f}")
    
    # 反向传播
    print("\n执行反向传播...")
    loss.backward()
    
    # 检查梯度是否存在
    print("\n梯度检查:")
    print(f"  inputs.grad: {'✓' if inputs.grad is not None else '✗'} (shape: {inputs.grad.shape if inputs.grad is not None else 'None'})")
    print(f"  weights.grad: {'✓' if weights.grad is not None else '✗'} (shape: {weights.grad.shape if weights.grad is not None else 'None'})")
    print(f"  params.grad: {'✓' if params.grad is not None else '✗'} (shape: {params.grad.shape if params.grad is not None else 'None'})")
    print(f"  states.grad: {'✓' if states.grad is not None else '✗'} (shape: {states.grad.shape if states.grad is not None else 'None'})")
    
    # 打印部分梯度值
    if inputs.grad is not None:
        print(f"\n输入梯度样例:")
        print(f"  dL/dP: {inputs.grad[0].item():.6f}")
        print(f"  dL/dT: {inputs.grad[1].item():.6f}")
        print(f"  dL/dPET: {inputs.grad[2].item():.6f}")
    
    if weights.grad is not None:
        print(f"\n权重梯度样例:")
        print(f"  dL/dw_phen: {weights.grad[0].item():.6f}")
        print(f"  dL/dw_int: {weights.grad[1].item():.6f}")
        print(f"  dL/dw_snow: {weights.grad[2].item():.6f}")
        print(f"  dL/dw_sub: {weights.grad[3].item():.6f}")
    
    all_grads_exist = all([
        inputs.grad is not None,
        weights.grad is not None,
        params.grad is not None,
        states.grad is not None,
    ])
    
    if all_grads_exist:
        print("\n✓ 反向传播正常工作！")
    else:
        print("\n✗ 部分梯度未计算！")
    
    return all_grads_exist


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("MOPEX 模型梯度检查测试")
    print("使用 torch.autograd.gradcheck 验证梯度正确性")
    
    # 测试1: 反向传播基础测试
    test1_passed = test_backward_pass()
    
    # 测试2: 单次梯度检查
    test2_passed = test_gradcheck_single(eps=1e-6, atol=1e-5, rtol=1e-3)
    
    # 测试3: 多组随机样本
    test3_passed = test_gradcheck_random_samples(
        num_samples=20,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3
    )
    
    # 总结
    print("\n" + "=" * 80)
    print("最终总结")
    print("=" * 80)
    print(f"反向传播测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"单次梯度检查: {'✓ 通过' if test2_passed else '✗ 失败'}")
    print(f"随机样本测试: {'✓ 通过' if test3_passed else '✗ 失败'}")
    print("=" * 80)
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 所有测试通过！MOPEX 模型梯度计算正确！")
    else:
        print("\n⚠️  部分测试失败，请检查模型实现")
