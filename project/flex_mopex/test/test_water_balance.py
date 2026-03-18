"""
水量平衡测试：验证 MOPEX 模型的质量守恒

测试方程：P = Q + ET + ΔS

注意：新版本截留蒸发受 min(P, PET_effective) 约束，
      水量平衡仍然成立：P = Q + ET + ΔS
"""
import sys
import torch
import numpy as np
sys.path.append("/workspace/my_deltamodel")
from project.flex_mopex.models.mopex_core import (
    mopex_step,
    mopex_step_static,
    MOPEX_PARAMS_BOUNDS,
)


def generate_random_params(bounds: dict, batch_size: int, device: str = "cpu") -> dict:
    params = {}
    for name, (min_val, max_val) in bounds.items():
        params[name] = torch.rand(batch_size, device=device) * (max_val - min_val) + min_val
    # 确保 tmax > tmin（tmax 在接口中作为 tmin+trange 使用）
    params["tmax"] = params["tmin"] + torch.rand(batch_size, device=device) * 25.0 + 1.0
    return params


def generate_random_weights(batch_size: int, device: str = "cpu") -> dict:
    return {
        "w_phen": torch.rand(batch_size, device=device),
        "w_int":  torch.rand(batch_size, device=device),
        "w_snow": torch.rand(batch_size, device=device),
        "w_sub":  torch.rand(batch_size, device=device),
    }


def generate_random_inputs(batch_size: int, device: str = "cpu") -> dict:
    return {
        "P":   torch.rand(batch_size, device=device) * 50.0,
        "T":   torch.rand(batch_size, device=device) * 40.0 - 10.0,
        "PET": torch.rand(batch_size, device=device) * 10.0,
        "doy": torch.randint(1, 366, (batch_size,), device=device, dtype=torch.float32),
    }


def initialize_states(batch_size: int, device: str = "cpu") -> dict:
    return {
        "S1":  torch.rand(batch_size, device=device) * 20.0,
        "S2":  torch.rand(batch_size, device=device) * 100.0,
        "Sc1": torch.rand(batch_size, device=device) * 10.0,
        "Sc2": torch.rand(batch_size, device=device) * 50.0,
        "Sn":  torch.rand(batch_size, device=device) * 100.0,
    }


def _check_balance(P, Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new,
                   states, tolerance):
    delta_S = (
        (S1_new  - states["S1"])  +
        (S2_new  - states["S2"])  +
        (Sc1_new - states["Sc1"]) +
        (Sc2_new - states["Sc2"]) +
        (Sn_new  - states["Sn"])
    )
    rhs = Q_total + ET_total + delta_S
    error = torch.abs(P - rhs)
    return {
        "P": P, "Q": Q_total, "ET": ET_total, "delta_S": delta_S,
        "balance_left": P, "balance_right": rhs,
        "error": error,
        "relative_error": error / (P + 1e-6),
        "is_balanced": error < tolerance,
    }


def test_water_balance_step(params, weights, inputs, states, tolerance=1e-4):
    Q, ET, S1n, S2n, Sc1n, Sc2n, Snn = mopex_step(
        P=inputs["P"], T=inputs["T"], PET=inputs["PET"], doy=inputs["doy"],
        w_phen=weights["w_phen"], w_int=weights["w_int"],
        w_snow=weights["w_snow"], w_sub=weights["w_sub"],
        Sb1=params["Sb1"], tw=params["tw"], tu=params["tu"], Se=params["Se"],
        tc=params["tc"], ddf=params["ddf"], tcrit=params["tcrit"],
        Sb2=params["Sb2"], alpha=params["alpha"], is_time=params["is_time"],
        tmin=params["tmin"], tmax=params["tmax"],
        S1=states["S1"], S2=states["S2"], Sc1=states["Sc1"],
        Sc2=states["Sc2"], Sn=states["Sn"],
    )
    return _check_balance(inputs["P"], Q, ET, S1n, S2n, Sc1n, Sc2n, Snn, states, tolerance)


def test_water_balance_step_static(params, inputs, states, tolerance=1e-4):
    Q, ET, S1n, S2n, Sc1n, Sc2n, Snn = mopex_step_static(
        P=inputs["P"], T=inputs["T"], PET=inputs["PET"], doy=inputs["doy"],
        Sb1=params["Sb1"], tw=params["tw"], tu=params["tu"], Se=params["Se"],
        tc=params["tc"], ddf=params["ddf"], tcrit=params["tcrit"],
        Sb2=params["Sb2"], alpha=params["alpha"], is_time=params["is_time"],
        tmin=params["tmin"], tmax=params["tmax"],
        S1=states["S1"], S2=states["S2"], Sc1=states["Sc1"],
        Sc2=states["Sc2"], Sn=states["Sn"],
    )
    return _check_balance(inputs["P"], Q, ET, S1n, S2n, Sc1n, Sc2n, Snn, states, tolerance)


def run_comprehensive_tests(num_tests=100, batch_size=10, device="cpu", tolerance=1e-4):
    print("=" * 80)
    print("MOPEX 模型水量平衡测试 (mopex_step + mopex_step_static)")
    print("=" * 80)
    print(f"  测试组数: {num_tests}  |  批次大小: {batch_size}  |  容许误差: {tolerance} mm")
    print("=" * 80)

    results_by_fn = {
        "mopex_step":        {"errors": [], "failed": 0},
        "mopex_step_static": {"errors": [], "failed": 0},
    }
    total_samples = 0

    for _ in range(num_tests):
        params  = generate_random_params(MOPEX_PARAMS_BOUNDS, batch_size, device)
        weights = generate_random_weights(batch_size, device)
        inputs  = generate_random_inputs(batch_size, device)
        states  = initialize_states(batch_size, device)

        for fn_name, res in [
            ("mopex_step",        test_water_balance_step(params, weights, inputs, states, tolerance)),
            ("mopex_step_static", test_water_balance_step_static(params, inputs, states, tolerance)),
        ]:
            errs = res["error"].cpu().numpy()
            results_by_fn[fn_name]["errors"].extend(errs)
            results_by_fn[fn_name]["failed"] += (~res["is_balanced"].cpu().numpy()).sum()

        total_samples += batch_size

    print(f"\n{'函数':<22} {'平均误差':>12} {'最大误差':>12} {'失败数':>8} {'通过率':>8}")
    print("-" * 70)
    all_passed = True
    for fn_name, data in results_by_fn.items():
        errs = np.array(data["errors"])
        failed = data["failed"]
        passed = total_samples - failed
        pass_rate = passed / total_samples * 100
        print(f"{fn_name:<22} {np.mean(errs):>12.3e} {np.max(errs):>12.3e} "
              f"{failed:>8} {pass_rate:>7.2f}%")
        if failed > 0:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("✓ 所有测试通过！水量平衡完美！")
    else:
        print("✗ 部分样本未通过水量平衡测试")

    return all_passed


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    run_comprehensive_tests(num_tests=50, batch_size=20, device=device, tolerance=1e-4)
