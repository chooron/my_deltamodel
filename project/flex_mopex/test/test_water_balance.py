"""
水量平衡测试：验证 MOPEX 模型的质量守恒

测试方程：P = Q + ET + ΔS

其中：
- P: 降雨输入
- Q: 总径流输出 (Q_total)
- ET: 总蒸散发 (ET_total)
- ΔS: 所有状态变量的变化量 (S1, S2, Sc1, Sc2, Sn)
"""
import sys
import torch
import numpy as np
sys.path.append("/workspace/my_deltamodel")
from project.flex_mopex.models.mopex_core import mopex_step, MOPEX_PARAMS_BOUNDS


def generate_random_params(bounds: dict, batch_size: int, device: str = "cpu") -> dict:
    """生成随机参数在给定范围内"""
    params = {}
    for name, (min_val, max_val) in bounds.items():
        params[name] = torch.rand(batch_size, device=device) * (max_val - min_val) + min_val
    return params


def generate_random_weights(batch_size: int, device: str = "cpu") -> dict:
    """生成随机结构权重 [0, 1]"""
    return {
        "w_phen": torch.rand(batch_size, device=device),
        "w_int": torch.rand(batch_size, device=device),
        "w_snow": torch.rand(batch_size, device=device),
        "w_sub": torch.rand(batch_size, device=device),
    }


def generate_random_inputs(batch_size: int, device: str = "cpu") -> dict:
    """生成随机气象输入"""
    return {
        "P": torch.rand(batch_size, device=device) * 50.0,  # 0-50 mm/day
        "T": torch.rand(batch_size, device=device) * 40.0 - 10.0,  # -10 to 30°C
        "PET": torch.rand(batch_size, device=device) * 10.0,  # 0-10 mm/day
        "doy": torch.randint(1, 366, (batch_size,), device=device, dtype=torch.float32),
    }


def initialize_states(batch_size: int, device: str = "cpu") -> dict:
    """初始化状态变量（随机初始值）"""
    return {
        "S1": torch.rand(batch_size, device=device) * 20.0,  # 0-20 mm
        "S2": torch.rand(batch_size, device=device) * 100.0,  # 0-100 mm
        "Sc1": torch.rand(batch_size, device=device) * 10.0,  # 0-10 mm
        "Sc2": torch.rand(batch_size, device=device) * 50.0,  # 0-50 mm
        "Sn": torch.rand(batch_size, device=device) * 100.0,  # 0-100 mm (雪)
    }


def test_water_balance_single(
    params: dict,
    weights: dict,
    inputs: dict,
    states: dict,
    device: str = "cpu",
    tolerance: float = 1e-4
) -> dict:
    """
    单次水量平衡测试
    
    Returns:
        dict: 包含测试结果的字典
    """
    # 运行模型
    Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new = mopex_step(
        P=inputs["P"],
        T=inputs["T"],
        PET=inputs["PET"],
        doy=inputs["doy"],
        w_phen=weights["w_phen"],
        w_int=weights["w_int"],
        w_snow=weights["w_snow"],
        w_sub=weights["w_sub"],
        Sb1=params["Sb1"],
        tw=params["tw"],
        tu=params["tu"],
        Se=params["Se"],
        tc=params["tc"],
        ddf=params["ddf"],
        tcrit=params["tcrit"],
        Sb2=params["Sb2"],
        alpha=params["alpha"],
        is_time=params["is_time"],
        tmin=params["tmin"],
        tmax=params["tmax"],
        S1=states["S1"],
        S2=states["S2"],
        Sc1=states["Sc1"],
        Sc2=states["Sc2"],
        Sn=states["Sn"],
    )
    
    # 计算状态变化量
    delta_S1 = S1_new - states["S1"]
    delta_S2 = S2_new - states["S2"]
    delta_Sc1 = Sc1_new - states["Sc1"]
    delta_Sc2 = Sc2_new - states["Sc2"]
    delta_Sn = Sn_new - states["Sn"]
    
    # 总状态变化
    delta_S_total = delta_S1 + delta_S2 + delta_Sc1 + delta_Sc2 + delta_Sn
    
    # 水量平衡：P = Q + ET + ΔS
    P = inputs["P"]
    water_balance_left = P
    water_balance_right = Q_total + ET_total + delta_S_total
    
    # 计算误差
    error = torch.abs(water_balance_left - water_balance_right)
    relative_error = error / (P + 1e-6)
    
    # 判断是否满足平衡
    is_balanced = error < tolerance
    
    return {
        "P": P,
        "Q": Q_total,
        "ET": ET_total,
        "delta_S": delta_S_total,
        "balance_left": water_balance_left,
        "balance_right": water_balance_right,
        "error": error,
        "relative_error": relative_error,
        "is_balanced": is_balanced,
        "states": {
            "S1": (states["S1"], S1_new, delta_S1),
            "S2": (states["S2"], S2_new, delta_S2),
            "Sc1": (states["Sc1"], Sc1_new, delta_Sc1),
            "Sc2": (states["Sc2"], Sc2_new, delta_Sc2),
            "Sn": (states["Sn"], Sn_new, delta_Sn),
        }
    }


def run_comprehensive_tests(
    num_tests: int = 100,
    batch_size: int = 10,
    device: str = "cpu",
    tolerance: float = 1e-4
):
    """
    运行综合水量平衡测试
    
    Args:
        num_tests: 测试组数
        batch_size: 每组测试的批次大小
        device: 计算设备
        tolerance: 容许误差阈值
    """
    print("=" * 80)
    print("MOPEX 模型水量平衡测试")
    print("=" * 80)
    print(f"测试配置：")
    print(f"  - 测试组数: {num_tests}")
    print(f"  - 每组批次大小: {batch_size}")
    print(f"  - 总测试样本数: {num_tests * batch_size}")
    print(f"  - 容许误差: {tolerance} mm")
    print(f"  - 计算设备: {device}")
    print("=" * 80)
    
    all_errors = []
    all_relative_errors = []
    failed_count = 0
    total_samples = 0
    
    for test_idx in range(num_tests):
        # 生成随机参数、权重和输入
        params = generate_random_params(MOPEX_PARAMS_BOUNDS, batch_size, device)
        weights = generate_random_weights(batch_size, device)
        inputs = generate_random_inputs(batch_size, device)
        states = initialize_states(batch_size, device)
        
        # 运行测试
        results = test_water_balance_single(params, weights, inputs, states, device, tolerance)
        
        # 收集统计信息
        errors = results["error"].cpu().numpy()
        rel_errors = results["relative_error"].cpu().numpy()
        balanced = results["is_balanced"].cpu().numpy()
        
        all_errors.extend(errors)
        all_relative_errors.extend(rel_errors)
        failed_count += (~balanced).sum()
        total_samples += batch_size
        
        # 打印详细信息（每10组打印一次）
        if (test_idx + 1) % 10 == 0:
            print(f"\n测试组 {test_idx + 1}/{num_tests}:")
            print(f"  平均绝对误差: {np.mean(errors):.6e} mm")
            print(f"  最大绝对误差: {np.max(errors):.6e} mm")
            print(f"  平均相对误差: {np.mean(rel_errors) * 100:.4f}%")
            print(f"  通过率: {balanced.sum()}/{batch_size}")
            
            # 打印一个详细样本
            sample_idx = 0
            print(f"\n  样本详情 (index={sample_idx}):")
            print(f"    输入 P: {results['P'][sample_idx].item():.4f} mm")
            print(f"    输出 Q: {results['Q'][sample_idx].item():.4f} mm")
            print(f"    蒸发 ET: {results['ET'][sample_idx].item():.4f} mm")
            print(f"    ΔS: {results['delta_S'][sample_idx].item():.4f} mm")
            print(f"    左边: {results['balance_left'][sample_idx].item():.4f} mm")
            print(f"    右边: {results['balance_right'][sample_idx].item():.4f} mm")
            print(f"    误差: {results['error'][sample_idx].item():.6e} mm")
            print(f"    相对误差: {results['relative_error'][sample_idx].item() * 100:.4f}%")
            print(f"    权重: w_phen={weights['w_phen'][sample_idx].item():.3f}, "
                  f"w_int={weights['w_int'][sample_idx].item():.3f}, "
                  f"w_snow={weights['w_snow'][sample_idx].item():.3f}, "
                  f"w_sub={weights['w_sub'][sample_idx].item():.3f}")
    
    # 总结统计
    all_errors = np.array(all_errors)
    all_relative_errors = np.array(all_relative_errors)
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"通过样本数: {total_samples - failed_count}")
    print(f"失败样本数: {failed_count}")
    print(f"通过率: {(total_samples - failed_count) / total_samples * 100:.2f}%")
    print(f"\n绝对误差统计 (mm):")
    print(f"  平均值: {np.mean(all_errors):.6e}")
    print(f"  标准差: {np.std(all_errors):.6e}")
    print(f"  最小值: {np.min(all_errors):.6e}")
    print(f"  中位数: {np.median(all_errors):.6e}")
    print(f"  最大值: {np.max(all_errors):.6e}")
    print(f"  99分位: {np.percentile(all_errors, 99):.6e}")
    print(f"\n相对误差统计 (%):")
    print(f"  平均值: {np.mean(all_relative_errors) * 100:.4f}")
    print(f"  标准差: {np.std(all_relative_errors) * 100:.4f}")
    print(f"  最小值: {np.min(all_relative_errors) * 100:.4f}")
    print(f"  中位数: {np.median(all_relative_errors) * 100:.4f}")
    print(f"  最大值: {np.max(all_relative_errors) * 100:.4f}")
    print(f"  99分位: {np.percentile(all_relative_errors, 99) * 100:.4f}")
    print("=" * 80)
    
    # 判断测试是否全部通过
    if failed_count == 0:
        print("\n✓ 所有测试通过！水量平衡完美！")
    else:
        print(f"\n✗ {failed_count}/{total_samples} 个样本未通过测试")
        print(f"  (容许误差阈值: {tolerance} mm)")
    
    return {
        "total_samples": total_samples,
        "passed": total_samples - failed_count,
        "failed": failed_count,
        "pass_rate": (total_samples - failed_count) / total_samples,
        "errors": all_errors,
        "relative_errors": all_relative_errors,
    }


if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    # 运行测试
    results = run_comprehensive_tests(
        num_tests=50,  # 50组测试
        batch_size=20,  # 每组20个样本
        device=device,
        tolerance=1e-4  # 容许误差 0.0001 mm
    )
