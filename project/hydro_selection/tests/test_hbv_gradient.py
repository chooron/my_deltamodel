"""
HBV 模型参数梯度测试代码

测试内容：
1. 验证每个参数的自动微分梯度
2. 使用有限差分法验证梯度正确性
3. 检查梯度的数值稳定性
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到路径
sys.path.append(os.getenv("PROJ_PATH", "."))

# 导入 HBV 核心函数 (同目录导入)
from project.hydro_selection.models.hbv_core import hbv_timestep_loop  # noqa


def create_test_data(
    n_steps: int = 100,
    n_grid: int = 10,
    nmul: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    创建测试数据

    Parameters
    ----------
    n_steps : int
        时间步数
    n_grid : int
        流域数量
    nmul : int
        专家数
    device : str
        设备类型
    seed : int
        随机种子

    Returns
    -------
    Dict[str, torch.Tensor]
        包含所有输入数据和参数的字典
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 驱动数据 (T, B, E)
    P = (
        torch.rand((n_steps, n_grid, nmul), device=device) * 20.0
    )  # 降水 0-20 mm
    T = (
        torch.randn((n_steps, n_grid, nmul), device=device) * 10.0
    )  # 温度 -10 到 10 度
    PET = (
        torch.rand((n_steps, n_grid, nmul), device=device) * 5.0
    )  # 潜在蒸发 0-5 mm

    # 初始状态 (B, E)
    SNOWPACK = torch.rand((n_grid, nmul), device=device) * 50.0
    MELTWATER = torch.rand((n_grid, nmul), device=device) * 10.0
    SM = torch.rand((n_grid, nmul), device=device) * 100.0 + 50.0
    SUZ = torch.rand((n_grid, nmul), device=device) * 20.0
    SLZ = torch.rand((n_grid, nmul), device=device) * 50.0

    # 静态参数 (B, E) - 设置合理的物理范围
    params = {
        "parTT": torch.zeros((n_grid, nmul), device=device),  # 温度阈值 ~0°C
        "parCFMAX": torch.rand((n_grid, nmul), device=device) * 3.0
        + 1.0,  # 1-4 mm/°C/day
        "parCFR": torch.rand((n_grid, nmul), device=device) * 0.1
        + 0.01,  # 0.01-0.11
        "parCWH": torch.rand((n_grid, nmul), device=device) * 0.1
        + 0.05,  # 0.05-0.15
        "parFC": torch.rand((n_grid, nmul), device=device) * 200.0
        + 100.0,  # 100-300 mm
        "parBETA": torch.rand((n_grid, nmul), device=device) * 3.0 + 1.0,  # 1-4
        "parLP": torch.rand((n_grid, nmul), device=device) * 0.5
        + 0.3,  # 0.3-0.8
        "parBETAET": torch.rand((n_grid, nmul), device=device) * 1.0
        + 0.5,  # 0.5-1.5
        "parC": torch.rand((n_grid, nmul), device=device) * 0.05
        + 0.01,  # 0.01-0.06
        "parPERC": torch.rand((n_grid, nmul), device=device) * 3.0
        + 0.5,  # 0.5-3.5 mm/day
        "parK0": torch.rand((n_grid, nmul), device=device) * 0.3
        + 0.1,  # 0.1-0.4
        "parK1": torch.rand((n_grid, nmul), device=device) * 0.1
        + 0.01,  # 0.01-0.11
        "parK2": torch.rand((n_grid, nmul), device=device) * 0.05
        + 0.001,  # 0.001-0.051
        "parUZL": torch.rand((n_grid, nmul), device=device) * 30.0
        + 10.0,  # 10-40 mm
    }

    return {
        "P": P,
        "T": T,
        "PET": PET,
        "SNOWPACK": SNOWPACK,
        "MELTWATER": MELTWATER,
        "SM": SM,
        "SUZ": SUZ,
        "SLZ": SLZ,
        **params,
    }


def compute_loss(outputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    计算损失函数（用于反向传播）

    使用总产流的均值作为简单的损失函数
    """
    Qsim_out = outputs[0]  # 总产流
    return Qsim_out.mean()


def test_single_param_gradient(
    data: Dict[str, torch.Tensor],
    param_name: str,
    eps: float = 1e-4,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    测试单个参数的梯度

    Parameters
    ----------
    data : Dict[str, torch.Tensor]
        输入数据
    param_name : str
        参数名称
    eps : float
        有限差分步长
    rtol : float
        相对容差
    atol : float
        绝对容差

    Returns
    -------
    Dict[str, any]
        包含测试结果的字典
    """
    device = data["P"].device
    nearzero = 1e-6

    # 准备参数 - 需要梯度的参数
    param = data[param_name].clone().detach().requires_grad_(True)

    # 构建参数字典
    params_dict = {}
    for key in data:
        if key.startswith("par"):
            if key == param_name:
                params_dict[key] = param
            else:
                params_dict[key] = data[key].clone().detach()
        elif key != "nearzero":
            params_dict[key] = data[key].clone().detach()

    # ==================== 自动微分梯度 ====================
    outputs = hbv_timestep_loop(
        P=params_dict.get("P", data["P"]),
        T=params_dict.get("T", data["T"]),
        PET=params_dict.get("PET", data["PET"]),
        SNOWPACK=params_dict.get("SNOWPACK", data["SNOWPACK"]).clone(),
        MELTWATER=params_dict.get("MELTWATER", data["MELTWATER"]).clone(),
        SM=params_dict.get("SM", data["SM"]).clone(),
        SUZ=params_dict.get("SUZ", data["SUZ"]).clone(),
        SLZ=params_dict.get("SLZ", data["SLZ"]).clone(),
        parTT=params_dict.get("parTT", data["parTT"]),
        parCFMAX=params_dict.get("parCFMAX", data["parCFMAX"]),
        parCFR=params_dict.get("parCFR", data["parCFR"]),
        parCWH=params_dict.get("parCWH", data["parCWH"]),
        parFC=params_dict.get("parFC", data["parFC"]),
        parBETA=params_dict.get("parBETA", data["parBETA"]),
        parLP=params_dict.get("parLP", data["parLP"]),
        parBETAET=params_dict.get("parBETAET", data["parBETAET"]),
        parC=params_dict.get("parC", data["parC"]),
        parPERC=params_dict.get("parPERC", data["parPERC"]),
        parK0=params_dict.get("parK0", data["parK0"]),
        parK1=params_dict.get("parK1", data["parK1"]),
        parK2=params_dict.get("parK2", data["parK2"]),
        parUZL=params_dict.get("parUZL", data["parUZL"]),
        nearzero=nearzero,
    )

    loss = compute_loss(outputs)
    loss.backward()

    autograd_grad = param.grad.clone()

    # ==================== 有限差分梯度 ====================
    finite_diff_grad = torch.zeros_like(param)
    param_flat = param.detach().flatten()

    for i in range(min(param_flat.numel(), 20)):  # 只测试前20个元素以节省时间
        # 正向扰动
        param_plus = param.detach().clone()
        param_plus.view(-1)[i] += eps

        outputs_plus = hbv_timestep_loop(
            P=data["P"],
            T=data["T"],
            PET=data["PET"],
            SNOWPACK=data["SNOWPACK"].clone(),
            MELTWATER=data["MELTWATER"].clone(),
            SM=data["SM"].clone(),
            SUZ=data["SUZ"].clone(),
            SLZ=data["SLZ"].clone(),
            parTT=param_plus if param_name == "parTT" else data["parTT"],
            parCFMAX=param_plus
            if param_name == "parCFMAX"
            else data["parCFMAX"],
            parCFR=param_plus if param_name == "parCFR" else data["parCFR"],
            parCWH=param_plus if param_name == "parCWH" else data["parCWH"],
            parFC=param_plus if param_name == "parFC" else data["parFC"],
            parBETA=param_plus if param_name == "parBETA" else data["parBETA"],
            parLP=param_plus if param_name == "parLP" else data["parLP"],
            parBETAET=param_plus
            if param_name == "parBETAET"
            else data["parBETAET"],
            parC=param_plus if param_name == "parC" else data["parC"],
            parPERC=param_plus if param_name == "parPERC" else data["parPERC"],
            parK0=param_plus if param_name == "parK0" else data["parK0"],
            parK1=param_plus if param_name == "parK1" else data["parK1"],
            parK2=param_plus if param_name == "parK2" else data["parK2"],
            parUZL=param_plus if param_name == "parUZL" else data["parUZL"],
            nearzero=nearzero,
        )
        loss_plus = compute_loss(outputs_plus)

        # 负向扰动
        param_minus = param.detach().clone()
        param_minus.view(-1)[i] -= eps

        outputs_minus = hbv_timestep_loop(
            P=data["P"],
            T=data["T"],
            PET=data["PET"],
            SNOWPACK=data["SNOWPACK"].clone(),
            MELTWATER=data["MELTWATER"].clone(),
            SM=data["SM"].clone(),
            SUZ=data["SUZ"].clone(),
            SLZ=data["SLZ"].clone(),
            parTT=param_minus if param_name == "parTT" else data["parTT"],
            parCFMAX=param_minus
            if param_name == "parCFMAX"
            else data["parCFMAX"],
            parCFR=param_minus if param_name == "parCFR" else data["parCFR"],
            parCWH=param_minus if param_name == "parCWH" else data["parCWH"],
            parFC=param_minus if param_name == "parFC" else data["parFC"],
            parBETA=param_minus if param_name == "parBETA" else data["parBETA"],
            parLP=param_minus if param_name == "parLP" else data["parLP"],
            parBETAET=param_minus
            if param_name == "parBETAET"
            else data["parBETAET"],
            parC=param_minus if param_name == "parC" else data["parC"],
            parPERC=param_minus if param_name == "parPERC" else data["parPERC"],
            parK0=param_minus if param_name == "parK0" else data["parK0"],
            parK1=param_minus if param_name == "parK1" else data["parK1"],
            parK2=param_minus if param_name == "parK2" else data["parK2"],
            parUZL=param_minus if param_name == "parUZL" else data["parUZL"],
            nearzero=nearzero,
        )
        loss_minus = compute_loss(outputs_minus)

        # 中心差分
        finite_diff_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)

    # ==================== 比较结果 ====================
    # 只比较有限差分计算过的元素
    n_compare = min(param_flat.numel(), 20)
    autograd_flat = autograd_grad.flatten()[:n_compare]
    finite_diff_flat = finite_diff_grad.flatten()[:n_compare]

    # 计算相对误差
    abs_diff = torch.abs(autograd_flat - finite_diff_flat)
    rel_diff = abs_diff / (torch.abs(finite_diff_flat) + 1e-8)

    # 判断是否通过
    passed = torch.allclose(
        autograd_flat, finite_diff_flat, rtol=rtol, atol=atol
    )

    return {
        "param_name": param_name,
        "passed": passed,
        "autograd_grad": autograd_grad,
        "finite_diff_grad": finite_diff_grad,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "grad_norm": autograd_grad.norm().item(),
    }


def test_all_param_gradients(
    n_steps: int = 50,
    n_grid: int = 5,
    nmul: int = 1,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    测试所有参数的梯度

    Parameters
    ----------
    n_steps : int
        时间步数
    n_grid : int
        流域数量
    nmul : int
        专家数
    verbose : bool
        是否打印详细信息

    Returns
    -------
    Dict[str, Dict[str, any]]
        所有参数的测试结果
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"测试配置: n_steps={n_steps}, n_grid={n_grid}, nmul={nmul}")
    print("=" * 80)

    # 创建测试数据
    data = create_test_data(n_steps, n_grid, nmul, device)

    # 所有需要测试的参数
    param_names = [
        "parTT",
        "parCFMAX",
        "parCFR",
        "parCWH",
        "parFC",
        "parBETA",
        "parLP",
        "parBETAET",
        "parC",
        "parPERC",
        "parK0",
        "parK1",
        "parK2",
        "parUZL",
    ]

    results = {}

    for param_name in param_names:
        if verbose:
            print(f"\n测试参数: {param_name}")

        try:
            result = test_single_param_gradient(data, param_name)
            results[param_name] = result

            if verbose:
                status = "✅ 通过" if result["passed"] else "❌ 失败"
                print(f"  状态: {status}")
                print(f"  梯度范数: {result['grad_norm']:.6e}")
                print(f"  最大绝对误差: {result['max_abs_diff']:.6e}")
                print(f"  最大相对误差: {result['max_rel_diff']:.6e}")

        except Exception as e:
            results[param_name] = {"passed": False, "error": str(e)}
            if verbose:
                print(f"  ❌ 错误: {e}")

    # 汇总结果
    print("\n" + "=" * 80)
    print("测试汇总:")
    print("-" * 80)

    n_passed = sum(1 for r in results.values() if r.get("passed", False))
    n_total = len(results)

    print(f"通过: {n_passed}/{n_total}")

    if n_passed < n_total:
        print("\n失败的参数:")
        for name, result in results.items():
            if not result.get("passed", False):
                if "error" in result:
                    print(f"  - {name}: {result['error']}")
                else:
                    print(
                        f"  - {name}: 最大相对误差 = {result['max_rel_diff']:.6e}"
                    )

    return results


def gradient_check_detailed(
    param_name: str = "parK0",
    n_steps: int = 10,
    n_grid: int = 2,
    nmul: int = 1,
):
    """
    对单个参数进行详细的梯度检查

    打印每个元素的梯度值对比
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = create_test_data(n_steps, n_grid, nmul, device)

    print(f"参数 {param_name} 的详细梯度检查")
    print(f"参数形状: {data[param_name].shape}")
    print("=" * 60)

    result = test_single_param_gradient(data, param_name, eps=1e-4)

    autograd = result["autograd_grad"].flatten()[:20]
    finite_diff = result["finite_diff_grad"].flatten()[:20]

    print(
        f"{'Index':<8} {'Autograd':<15} {'FiniteDiff':<15} {'AbsDiff':<12} {'RelDiff':<12}"
    )
    print("-" * 60)

    for i in range(len(autograd)):
        ag = autograd[i].item()
        fd = finite_diff[i].item()
        abs_diff = abs(ag - fd)
        rel_diff = abs_diff / (abs(fd) + 1e-8)
        print(
            f"{i:<8} {ag:<15.6e} {fd:<15.6e} {abs_diff:<12.6e} {rel_diff:<12.6e}"
        )

    print("-" * 60)
    print(f"总体状态: {'✅ 通过' if result['passed'] else '❌ 失败'}")


def test_gradient_flow():
    """
    测试梯度流是否正确传播到所有参数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = create_test_data(n_steps=20, n_grid=3, nmul=1, device=device)

    print("梯度流测试")
    print("=" * 60)

    # 将所有参数设为需要梯度
    params = {}
    for key in data:
        if key.startswith("par"):
            params[key] = data[key].clone().detach().requires_grad_(True)

    # 运行模型
    outputs = hbv_timestep_loop(
        P=data["P"],
        T=data["T"],
        PET=data["PET"],
        SNOWPACK=data["SNOWPACK"].clone(),
        MELTWATER=data["MELTWATER"].clone(),
        SM=data["SM"].clone(),
        SUZ=data["SUZ"].clone(),
        SLZ=data["SLZ"].clone(),
        **params,
    )

    loss = compute_loss(outputs)
    loss.backward()

    print(f"{'参数':<15} {'梯度范数':<15} {'梯度均值':<15} {'非零比例':<15}")
    print("-" * 60)

    for name, param in params.items():
        if param.grad is not None:
            grad = param.grad
            norm = grad.norm().item()
            mean = grad.mean().item()
            nonzero_ratio = (grad != 0).float().mean().item()
            print(
                f"{name:<15} {norm:<15.6e} {mean:<15.6e} {nonzero_ratio:<15.2%}"
            )
        else:
            print(f"{name:<15} {'None':<15} {'N/A':<15} {'N/A':<15}")


def test_numerical_stability():
    """
    测试梯度的数值稳定性
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("数值稳定性测试")
    print("=" * 60)

    # 测试不同的时间步长
    for n_steps in [10, 50, 100, 200]:
        data = create_test_data(
            n_steps=n_steps, n_grid=5, nmul=1, device=device
        )

        # 测试 parK0 的梯度
        param = data["parK0"].clone().detach().requires_grad_(True)

        outputs = hbv_timestep_loop(
            P=data["P"],
            T=data["T"],
            PET=data["PET"],
            SNOWPACK=data["SNOWPACK"].clone(),
            MELTWATER=data["MELTWATER"].clone(),
            SM=data["SM"].clone(),
            SUZ=data["SUZ"].clone(),
            SLZ=data["SLZ"].clone(),
            parTT=data["parTT"],
            parCFMAX=data["parCFMAX"],
            parCFR=data["parCFR"],
            parCWH=data["parCWH"],
            parFC=data["parFC"],
            parBETA=data["parBETA"],
            parLP=data["parLP"],
            parBETAET=data["parBETAET"],
            parC=data["parC"],
            parPERC=data["parPERC"],
            parK0=param,
            parK1=data["parK1"],
            parK2=data["parK2"],
            parUZL=data["parUZL"],
        )

        loss = compute_loss(outputs)
        loss.backward()

        grad_norm = param.grad.norm().item()
        grad_max = param.grad.abs().max().item()
        has_nan = torch.isnan(param.grad).any().item()
        has_inf = torch.isinf(param.grad).any().item()

        status = "✅" if not (has_nan or has_inf) else "❌"
        print(
            f"n_steps={n_steps:>4}: 范数={grad_norm:.6e}, 最大值={grad_max:.6e}, "
            f"NaN={has_nan}, Inf={has_inf} {status}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HBV 模型参数梯度测试")
    print("=" * 80 + "\n")

    # 1. 测试梯度流
    print("\n[1] 梯度流测试")
    print("-" * 80)
    test_gradient_flow()

    # 2. 测试数值稳定性
    print("\n\n[2] 数值稳定性测试")
    print("-" * 80)
    test_numerical_stability()

    # 3. 测试所有参数的梯度正确性
    print("\n\n[3] 所有参数梯度正确性测试")
    print("-" * 80)
    results = test_all_param_gradients(
        n_steps=30, n_grid=3, nmul=1, verbose=True
    )

    # 4. 对一个参数进行详细检查
    print("\n\n[4] 单参数详细梯度检查")
    print("-" * 80)
    gradient_check_detailed(param_name="parK0", n_steps=10, n_grid=2)
