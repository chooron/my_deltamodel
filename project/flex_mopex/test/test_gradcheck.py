"""
梯度检查测试：验证 mopex_step 和 mopex_step_static 的梯度计算正确性

使用 torch.autograd.gradcheck 对比数值梯度和自动微分梯度
"""

import sys
import torch
import numpy as np
sys.path.append("/workspace/my_deltamodel")
from project.flex_mopex.models.mopex_core import mopex_step, mopex_step_static


# ============================================================
# Wrappers
# ============================================================

class MopexStepWrapper(torch.nn.Module):
    """mopex_step (with structural weights)"""

    def forward(self, inputs, weights, params, states):
        """
        inputs:  [P, T, PET, doy]
        weights: [w_phen, w_int, w_snow, w_sub]
        params:  [Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax]
        states:  [S1, S2, Sc1, Sc2, Sn]
        """
        P, T, PET, doy = inputs[0], inputs[1], inputs[2], inputs[3]
        w_phen, w_int, w_snow, w_sub = weights[0], weights[1], weights[2], weights[3]
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax = (
            params[0], params[1], params[2], params[3], params[4], params[5],
            params[6], params[7], params[8], params[9], params[10], params[11],
        )
        S1, S2, Sc1, Sc2, Sn = states[0], states[1], states[2], states[3], states[4]

        Q, ET, S1n, S2n, Sc1n, Sc2n, Snn = mopex_step(
            P=P, T=T, PET=PET, doy=doy,
            w_phen=w_phen, w_int=w_int, w_snow=w_snow, w_sub=w_sub,
            Sb1=Sb1, tw=tw, tu=tu, Se=Se, tc=tc, ddf=ddf, tcrit=tcrit,
            Sb2=Sb2, alpha=alpha, is_time=is_time, tmin=tmin, tmax=tmax,
            S1=S1, S2=S2, Sc1=Sc1, Sc2=Sc2, Sn=Sn,
        )
        return torch.stack([Q, ET, S1n, S2n, Sc1n, Sc2n, Snn])


class MopexStepStaticWrapper(torch.nn.Module):
    """mopex_step_static (no structural weights)"""

    def forward(self, inputs, params, states):
        """
        inputs:  [P, T, PET, doy]
        params:  [Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax]
        states:  [S1, S2, Sc1, Sc2, Sn]
        """
        P, T, PET, doy = inputs[0], inputs[1], inputs[2], inputs[3]
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax = (
            params[0], params[1], params[2], params[3], params[4], params[5],
            params[6], params[7], params[8], params[9], params[10], params[11],
        )
        S1, S2, Sc1, Sc2, Sn = states[0], states[1], states[2], states[3], states[4]

        Q, ET, S1n, S2n, Sc1n, Sc2n, Snn = mopex_step_static(
            P=P, T=T, PET=PET, doy=doy,
            Sb1=Sb1, tw=tw, tu=tu, Se=Se, tc=tc, ddf=ddf, tcrit=tcrit,
            Sb2=Sb2, alpha=alpha, is_time=is_time, tmin=tmin, tmax=tmax,
            S1=S1, S2=S2, Sc1=Sc1, Sc2=Sc2, Sn=Sn,
        )
        return torch.stack([Q, ET, S1n, S2n, Sc1n, Sc2n, Snn])


# ============================================================
# Input generators
# ============================================================

def make_inputs(device="cpu", dtype=torch.float64):
    """固定测试点（夏季降雨场景，T > tcrit 确保无积雪歧义）"""
    inputs = torch.tensor([25.0, 15.0, 5.0, 180.0],
                          device=device, dtype=dtype, requires_grad=True)
    weights = torch.tensor([0.5, 0.3, 0.7, 0.4],
                           device=device, dtype=dtype, requires_grad=True)
    # tmin=2, tmax=22 → trange=20，T=15 在线性区，GSI 梯度非零
    params = torch.tensor([
        25.0,   # Sb1
        2.5,    # tw
        500.0,  # tu
        300.0,  # Se
        10.0,   # tc
        5.0,    # ddf
        0.0,    # tcrit
        500.0,  # Sb2
        0.3,    # alpha
        180.0,  # is_time
        2.0,    # tmin
        22.0,   # tmax  (trange = 20)
    ], device=device, dtype=dtype, requires_grad=True)
    states = torch.tensor([10.0, 50.0, 5.0, 25.0, 5.0],
                          device=device, dtype=dtype, requires_grad=True)
    return inputs, weights, params, states


def make_random_inputs(device="cpu", dtype=torch.float64):
    """随机测试点，保证 tmax > tmin + 1"""
    P   = torch.rand(1).item() * 40.0 + 5.0      # 5–45 mm
    T   = torch.rand(1).item() * 20.0 + 2.0       # 2–22 °C (避免 tcrit 附近硬边界)
    PET = torch.rand(1).item() * 8.0 + 1.0        # 1–9 mm
    doy = torch.rand(1).item() * 300.0 + 30.0     # 30–330

    tmin_v  = torch.rand(1).item() * 5.0           # 0–5
    trange_v = torch.rand(1).item() * 15.0 + 5.0  # 5–20
    tmax_v  = tmin_v + trange_v

    inputs = torch.tensor([P, T, PET, doy], dtype=dtype, requires_grad=True)
    weights = torch.rand(4, dtype=dtype).clamp(0.05, 0.95).requires_grad_(True)
    params = torch.tensor([
        torch.rand(1).item() * 40.0 + 5.0,    # Sb1: 5–45
        torch.rand(1).item() * 3.0 + 0.5,     # tw:  0.5–3.5
        torch.rand(1).item() * 400.0 + 50.0,  # tu:  50–450
        torch.rand(1).item() * 200.0 + 50.0,  # Se:  50–250
        torch.rand(1).item() * 15.0 + 1.0,    # tc:  1–16
        torch.rand(1).item() * 8.0 + 1.0,     # ddf: 1–9
        torch.rand(1).item() * 2.0 - 2.0,     # tcrit: -2–0 (T=2~22 远离 tcrit)
        torch.rand(1).item() * 400.0 + 50.0,  # Sb2: 50–450
        torch.rand(1).item() * 0.6 + 0.1,     # alpha: 0.1–0.7
        torch.rand(1).item() * 300.0 + 30.0,  # is_time: 30–330
        tmin_v,
        tmax_v,
    ], dtype=dtype, requires_grad=True)
    states = torch.tensor([
        torch.rand(1).item() * 15.0 + 1.0,   # S1
        torch.rand(1).item() * 80.0 + 10.0,  # S2
        torch.rand(1).item() * 8.0 + 1.0,    # Sc1
        torch.rand(1).item() * 30.0 + 5.0,   # Sc2
        torch.rand(1).item() * 5.0,           # Sn (小雪包，避免 min(melt, Sn) 边界)
    ], dtype=dtype, requires_grad=True)
    return inputs, weights, params, states


# ============================================================
# Test functions
# ============================================================

def test_backward_pass():
    print("\n" + "=" * 70)
    print("反向传播测试 (float32)")
    print("=" * 70)

    all_passed = True
    for fn_name, wrapper_cls, use_weights in [
        ("mopex_step",        MopexStepWrapper,       True),
        ("mopex_step_static", MopexStepStaticWrapper, False),
    ]:
        model = wrapper_cls()
        inputs, weights, params, states = make_inputs(dtype=torch.float32)

        if use_weights:
            out = model(inputs, weights, params, states)
        else:
            out = model(inputs, params, states)

        loss = out.sum()
        loss.backward()

        grads_ok = all(
            t.grad is not None and not torch.isnan(t.grad).any()
            for t in ([inputs, weights, params, states] if use_weights
                      else [inputs, params, states])
        )
        status = "✓" if grads_ok else "✗"
        print(f"  {fn_name:<24} {status}  Q={out[0].item():.4f}  ET={out[1].item():.4f}")
        if not grads_ok:
            all_passed = False

    return all_passed


def _gradcheck(model, args, eps=1e-6, atol=1e-4, rtol=1e-3):
    return torch.autograd.gradcheck(
        model, args, eps=eps, atol=atol, rtol=rtol, raise_exception=False
    )


def test_gradcheck_fixed(eps=1e-6, atol=1e-4, rtol=1e-3):
    print("\n" + "=" * 70)
    print("固定测试点梯度检查 (float64)")
    print("=" * 70)

    inputs, weights, params, states = make_inputs(dtype=torch.float64)

    results = {}

    # mopex_step
    model_dyn = MopexStepWrapper()
    ok = _gradcheck(model_dyn, (inputs, weights, params, states), eps, atol, rtol)
    results["mopex_step"] = ok
    print(f"  mopex_step        {'✓' if ok else '✗'}")

    if not ok:
        # 逐组诊断
        for label, fn in [
            ("inputs",  lambda x: model_dyn(x, weights.detach(), params.detach(), states.detach())),
            ("weights", lambda x: model_dyn(inputs.detach(), x, params.detach(), states.detach())),
            ("params",  lambda x: model_dyn(inputs.detach(), weights.detach(), x, states.detach())),
            ("states",  lambda x: model_dyn(inputs.detach(), weights.detach(), params.detach(), x)),
        ]:
            t = {"inputs": inputs, "weights": weights, "params": params, "states": states}[label]
            ok_sub = _gradcheck(fn, t.detach().clone().requires_grad_(True), eps, atol, rtol)
            print(f"    {label:<10} {'✓' if ok_sub else '✗'}")

    # mopex_step_static
    model_sta = MopexStepStaticWrapper()
    ok_s = _gradcheck(model_sta, (inputs, params, states), eps, atol, rtol)
    results["mopex_step_static"] = ok_s
    print(f"  mopex_step_static {'✓' if ok_s else '✗'}")

    if not ok_s:
        for label, fn in [
            ("inputs",  lambda x: model_sta(x, params.detach(), states.detach())),
            ("params",  lambda x: model_sta(inputs.detach(), x, states.detach())),
            ("states",  lambda x: model_sta(inputs.detach(), params.detach(), x)),
        ]:
            t = {"inputs": inputs, "params": params, "states": states}[label]
            ok_sub = _gradcheck(fn, t.detach().clone().requires_grad_(True), eps, atol, rtol)
            print(f"    {label:<10} {'✓' if ok_sub else '✗'}")

    return all(results.values())


def test_gradcheck_random(num_samples=10, eps=1e-6, atol=1e-4, rtol=1e-3):
    print("\n" + "=" * 70)
    print(f"随机样本梯度检查 (N={num_samples}, float64)")
    print("=" * 70)

    model_dyn = MopexStepWrapper()
    model_sta = MopexStepStaticWrapper()

    counts = {"mopex_step": 0, "mopex_step_static": 0}

    for i in range(num_samples):
        inputs, weights, params, states = make_random_inputs(dtype=torch.float64)

        ok_d = _gradcheck(model_dyn, (inputs, weights, params, states), eps, atol, rtol)
        ok_s = _gradcheck(model_sta, (inputs, params, states), eps, atol, rtol)

        counts["mopex_step"]        += int(ok_d)
        counts["mopex_step_static"] += int(ok_s)

        sym_d = "✓" if ok_d else "✗"
        sym_s = "✓" if ok_s else "✗"
        print(f"  [{i+1:2d}/{num_samples}]  mopex_step {sym_d}  mopex_step_static {sym_s}")

    print(f"\n  mopex_step        通过率: {counts['mopex_step']}/{num_samples}")
    print(f"  mopex_step_static 通过率: {counts['mopex_step_static']}/{num_samples}")

    return counts["mopex_step"] == num_samples and counts["mopex_step_static"] == num_samples


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("MOPEX 梯度检查测试")
    print("验证 mopex_step 和 mopex_step_static 的梯度正确性")

    r1 = test_backward_pass()
    r2 = test_gradcheck_fixed(eps=1e-6, atol=1e-4, rtol=1e-3)
    r3 = test_gradcheck_random(num_samples=15, eps=1e-6, atol=1e-4, rtol=1e-3)

    print("\n" + "=" * 70)
    print("最终总结")
    print("=" * 70)
    print(f"  反向传播测试:   {'✓ 通过' if r1 else '✗ 失败'}")
    print(f"  固定点梯度检查: {'✓ 通过' if r2 else '✗ 失败'}")
    print(f"  随机样本检查:   {'✓ 通过' if r3 else '✗ 失败'}")
    print("=" * 70)
    if all([r1, r2, r3]):
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查模型实现")
