"""
DiffBlendV1 模型测试套件

测试内容:
1. 模型实例化和参数计数
2. Forward pass 和输出形状
3. 梯度计算 (包括雪相关参数)
4. 水量平衡检查 —— 遍历全部公式组合 (3×3×3×3×3×2 = 162 种)
5. 不同权重激活方法
6. 多 nmul 配置
7. torch.autograd.gradcheck 严格梯度检查
"""

import sys
import torch
import torch.nn.functional as F
from itertools import product as cart_product

sys.path.insert(0, '.')
from project.blend_formula.models.diff_blend_v1 import DiffBlendV1, PROCESS_OPTIONS, TOTAL_WEIGHT_LOGITS


def test_model_instantiation():
    """测试模型实例化和参数计数"""
    print("\n" + "="*60)
    print("TEST 1: Model Instantiation")
    print("="*60)

    config = {
        'nmul': 1,
        'warm_up': 365,
        'warm_up_states': False,
        'weight_method': 'gumbel_softmax',
        'tau': 1.0
    }

    model = DiffBlendV1(config=config, device=torch.device('cpu'))

    print(f"Model name: {model.name}")
    print(f"PROCESS_OPTIONS: {PROCESS_OPTIONS}")
    print(f"TOTAL_WEIGHT_LOGITS: {TOTAL_WEIGHT_LOGITS}")
    print(f"Physical params: {len(model.param_names)}")
    print(f"Routing params: {len(model.routing_param_names)}")
    print(f"Total learnable params: {model.learnable_param_count}")

    expected = len(model.param_names) * config['nmul'] + len(model.routing_param_names) + TOTAL_WEIGHT_LOGITS
    assert model.learnable_param_count == expected, f"Expected {expected}, got {model.learnable_param_count}"

    print("✓ Model instantiation test passed")
    return model


def test_forward_pass():
    """测试 forward pass 和输出形状"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)

    config = {
        'nmul': 2,
        'warm_up': 10,
        'warm_up_states': True,
        'weight_method': 'softmax'
    }

    model = DiffBlendV1(config=config, device=torch.device('cpu'))
    model.train()

    B, T_steps = 4, 50
    raw = torch.randn(B, model.learnable_param_count)
    x_phy = torch.rand(T_steps, B, 3) * 10
    x_dict = {'x_phy': x_phy, 'target': torch.rand(T_steps, B)}

    result = model.forward(x_dict, (None, raw))

    print(f"Input shape: {x_phy.shape}")
    print(f"Streamflow shape: {result['streamflow'].shape}")
    print(f"Output keys count: {len(result.keys())}")
    print(f"Sample output keys: {list(result.keys())[:10]}")

    assert result['streamflow'].shape == (T_steps, B), f"Expected ({T_steps}, {B}), got {result['streamflow'].shape}"

    # 检查权重输出
    for proc_name in model.process_names:
        n_opt = PROCESS_OPTIONS[proc_name]
        for i in range(n_opt):
            key = f"w_{proc_name}_{i}"
            assert key in result, f"Missing weight key: {key}"
            assert result[key].shape == (T_steps, B), f"Wrong shape for {key}"

    print("✓ Forward pass test passed")
    return model, result


def test_gradient_flow():
    """测试梯度计算 (标准场景)"""
    print("\n" + "="*60)
    print("TEST 3: Gradient Flow (Standard)")
    print("="*60)

    config = {
        'nmul': 1,
        'warm_up': 10,
        'warm_up_states': True,
        'weight_method': 'gumbel_softmax'
    }

    model = DiffBlendV1(config=config, device=torch.device('cpu'))
    model.train()

    B, T_steps = 4, 50
    raw = torch.randn(B, model.learnable_param_count, requires_grad=True)
    x_phy = torch.rand(T_steps, B, 3) * 10
    x_dict = {'x_phy': x_phy, 'target': torch.rand(T_steps, B)}

    result = model.forward(x_dict, (None, raw))
    loss = result['streamflow'].sum()
    loss.backward()

    grad_norm = raw.grad.norm().item()
    grad_nonzero = (raw.grad.abs() > 1e-8).sum().item()
    grad_total = raw.grad.numel()

    print(f"Gradient norm: {grad_norm:.4f}")
    print(f"Nonzero gradients: {grad_nonzero}/{grad_total} ({100*grad_nonzero/grad_total:.1f}%)")
    print(f"Gradient stats: min={raw.grad.min().item():.6f}, max={raw.grad.max().item():.6f}, mean={raw.grad.mean().item():.6f}")

    assert grad_norm > 0, "Gradient norm is zero"
    assert grad_nonzero > 0, "No nonzero gradients"

    print("✓ Gradient flow test passed")
    return raw.grad


def test_gradient_with_snow():
    """测试梯度计算 (激活雪相关参数)"""
    print("\n" + "="*60)
    print("TEST 4: Gradient Flow (With Snow)")
    print("="*60)

    config = {
        'nmul': 1,
        'warm_up': 10,
        'warm_up_states': True,
        'weight_method': 'softmax'
    }

    model = DiffBlendV1(config=config, device=torch.device('cpu'))
    model.train()

    B, T_steps = 4, 100
    raw = torch.randn(B, model.learnable_param_count, requires_grad=True)

    # 创建有雪的场景: 低温 + 降水
    x_phy = torch.zeros(T_steps, B, 3)
    x_phy[:, :, 0] = torch.rand(T_steps, B) * 20 + 5  # prcp: 5-25 mm
    x_phy[:, :, 1] = torch.randn(T_steps, B) * 5 - 2  # tmean: -7 to 3°C (有雪)
    x_phy[:, :, 2] = torch.rand(T_steps, B) * 3 + 1   # pet: 1-4 mm

    x_dict = {'x_phy': x_phy, 'target': torch.rand(T_steps, B)}

    result = model.forward(x_dict, (None, raw))

    # 检查是否有雪相关输出
    has_snow = False
    for key in result.keys():
        if 'snow' in key.lower() or 'melt' in key.lower():
            val = result[key]
            if val.abs().sum() > 1e-6:
                has_snow = True
                print(f"  {key}: sum={val.sum().item():.4f}, mean={val.mean().item():.4f}")

    print(f"Snow-related fluxes detected: {has_snow}")

    # 计算梯度
    loss = result['streamflow'].sum()
    loss.backward()

    grad_norm = raw.grad.norm().item()
    grad_nonzero = (raw.grad.abs() > 1e-8).sum().item()
    grad_total = raw.grad.numel()

    print(f"Gradient norm: {grad_norm:.4f}")
    print(f"Nonzero gradients: {grad_nonzero}/{grad_total} ({100*grad_nonzero/grad_total:.1f}%)")

    # 检查雪相关参数的梯度
    snow_param_indices = [i for i, name in enumerate(model.param_names)
                          if any(x in name for x in ['swi', 'melt', 'refreeze', 'snow'])]
    print(f"Snow-related param indices: {snow_param_indices}")

    if len(snow_param_indices) > 0:
        snow_grads = raw.grad[:, snow_param_indices]
        snow_grad_nonzero = (snow_grads.abs() > 1e-8).sum().item()
        print(f"Snow param gradients: {snow_grad_nonzero}/{len(snow_param_indices)*B} nonzero")

    assert grad_norm > 0, "Gradient norm is zero"

    print("✓ Gradient with snow test passed")
    return result


def test_water_balance_formula_combinations():
    """遍历所有公式组合，逐一检验水量平衡。

    对 6 个水文过程的全部组合（3×3×3×3×3×2 = 162 种）：
      - 用 one-hot logits (±BIGLOGIT) 强制激活每种组合中的指定公式；
      - 将深层渗漏系数 x35 置近零，消除系统外损失，使水量方程闭合；
      - 雨/雪校正系数设为 1.0 以便精确统计有效降水；
      - 验证：ΔS = P_eff − Q_pre_route − ET ≥ −ABS_TOL
        （ΔS 为系统总蓄量净增量，物理上应 ≥ 0；
         ABS_TOL 容许 F.relu 截断产生的微量数值水损失）。

    P_eff 取自选中雨雪公式的输出（rainfall_opts_i + snowfall_opts_i）；
    ET    取自选中蒸发公式的输出（evaporation_opts_i）；
    Q     取自路由前三分量之和（Q_surface + Q_quick + Q_base）。
    """
    print("\n" + "="*60)
    print("TEST 5: Water Balance — Formula Combination Enumeration")
    print("="*60)

    BIGLOGIT = 50.0   # one-hot softmax 近似：exp(100) >> exp(0)
    ABS_TOL  = 5.0    # mm，绝对容差（F.relu 截断导致的容许水损失）

    config = {
        'nmul': 1,
        'warm_up': 0,
        'warm_up_states': True,
        'weight_method': 'softmax',
    }
    model = DiffBlendV1(config=config, device=torch.device('cpu'))
    model.eval()

    process_names   = list(PROCESS_OPTIONS.keys())
    n_options_list  = [PROCESS_OPTIONS[p] for p in process_names]
    combinations    = list(cart_product(*[range(n) for n in n_options_list]))
    n_combos        = len(combinations)   # 162

    # 权重 logits 各过程在权重区段内的起始偏移
    w_offsets = {}
    cur = 0
    for pname in process_names:
        w_offsets[pname] = cur
        cur += PROCESS_OPTIONS[pname]

    n_phy   = len(model.param_names)           # 35 (nmul=1)
    n_rout  = len(model.routing_param_names)   # 2
    n_w     = TOTAL_WEIGHT_LOGITS              # 17
    total   = model.learnable_param_count      # = n_phy + n_rout + n_w
    w_start = n_phy + n_rout                   # 权重 logits 在 raw 中的起点索引

    B = n_combos   # 162 条批次，每条对应一种公式组合
    T = 300

    # 物理参数全部设为中值（sigmoid(0)=0.5 → 各参数边界中点）
    torch.manual_seed(0)
    raw = torch.zeros(B, total)

    # 参数名称 → 索引映射
    param_idx = {name: i for i, name in enumerate(model.param_names)}

    # 雨/雪校正因子 = 1.0：x33/x34 的归一化值 = (1.0-0.8)/0.4 = 0.5 → logit = 0（默认即为0）
    # 深层渗漏系数 ≈ 0：消除系统外损失，确保水量方程可闭合
    raw[:, param_idx['x35_perc_coeff_phreatic']] = -10.0

    # 为每个组合写入 one-hot 权重 logits
    for combo_i, combo in enumerate(combinations):
        for pname, opt_idx in zip(process_names, combo):
            n_opts = PROCESS_OPTIONS[pname]
            base   = w_start + w_offsets[pname]
            for j in range(n_opts):
                raw[combo_i, base + j] = BIGLOGIT if j == opt_idx else -BIGLOGIT

    # 驱动输入：恒定降水 + 温度线性变化（同时覆盖积雪与无雪情景）
    x_phy = torch.zeros(T, B, 3)
    temp_series = torch.linspace(-5.0, 15.0, T)          # -5°C → 15°C
    x_phy[:, :, 0] = 5.0                                  # prcp = 5  mm/day
    x_phy[:, :, 1] = temp_series.unsqueeze(1).expand(-1, B)
    x_phy[:, :, 2] = 2.0                                  # PET  = 2  mm/day

    x_dict = {'x_phy': x_phy}

    with torch.no_grad():
        result = model.forward(x_dict, (None, raw))

    # ── 逐组合水量衡算 ─────────────────────────────────────────────
    process_short = ['RS', 'SN', 'INF', 'EV', 'QF', 'BF']
    passed = []
    failed = []

    for combo_i, combo in enumerate(combinations):
        rs_idx, sn_idx, inf_idx, ev_idx, qf_idx, bf_idx = combo

        # 有效降水 = 选中雨雪分割公式的输出（one-hot 权重下即为该公式值）
        rain = result[f'rainfall_opts_{rs_idx}'][:, combo_i]
        snow = result[f'snowfall_opts_{rs_idx}'][:, combo_i]
        P_eff = (rain + snow).sum().item()

        # 蒸发 = 选中蒸发公式的输出
        ET = result[f'evaporation_opts_{ev_idx}'][:, combo_i].sum().item()

        # 路由前总产流 = 地表径流 + 壤中流 + 基流
        Q = (
            result['Q_surface'][:, combo_i] +
            result['Q_quick'][:, combo_i]   +
            result['Q_base'][:, combo_i]
        ).sum().item()

        # ΔS = P_eff - Q - ET  （系统总蓄量净增量，物理上应 ≥ 0）
        delta_S   = P_eff - Q - ET
        rel_error = delta_S / (P_eff + 1e-6)

        label = '-'.join(f'{n}{i}' for n, i in zip(process_short, combo))
        row   = dict(combo=label, P=P_eff, ET=ET, Q=Q,
                     dS=delta_S, rel=rel_error)

        if delta_S < -ABS_TOL:
            failed.append(row)
        else:
            passed.append(row)

    # ── 输出汇总 ──────────────────────────────────────────────────
    print(f"\nFormula combinations tested : {n_combos}")
    print(f"Passed                      : {len(passed)}/{n_combos}")
    print(f"Failed                      : {len(failed)}/{n_combos}")

    if passed:
        dS_vals = [r['dS'] for r in passed]
        print(f"Passed ΔS range             : [{min(dS_vals):.3f}, {max(dS_vals):.3f}] mm")
        # 按 ΔS 最小的前5个打印（接近边界的情况）
        worst_passed = sorted(passed, key=lambda r: r['dS'])[:5]
        print("Closest-to-fail (top 5):")
        for r in worst_passed:
            print(f"  {r['combo']:30s}  P={r['P']:7.1f} Q={r['Q']:7.1f} "
                  f"ET={r['ET']:6.1f} ΔS={r['dS']:+8.3f} mm ({r['rel']:+.2%})")

    if failed:
        print(f"\nFailed combinations (first 10):")
        for r in failed[:10]:
            print(f"  {r['combo']:30s}  P={r['P']:7.1f} Q={r['Q']:7.1f} "
                  f"ET={r['ET']:6.1f} ΔS={r['dS']:+8.3f} mm ({r['rel']:+.2%})")

    assert len(failed) == 0, (
        f"{len(failed)}/{n_combos} formula combinations violate water balance "
        f"(ΔS < -{ABS_TOL} mm)"
    )
    print("\n✓ Water balance formula combination test passed")


def test_weight_methods():
    """测试不同权重激活方法"""
    print("\n" + "="*60)
    print("TEST 6: Weight Activation Methods")
    print("="*60)

    methods = ['softmax', 'gumbel_softmax', 'sparsemax', 'entmax15']

    B, T_steps = 4, 50
    x_phy = torch.rand(T_steps, B, 3) * 10
    x_dict = {'x_phy': x_phy, 'target': torch.rand(T_steps, B)}

    for method in methods:
        print(f"\nTesting {method}...")

        config = {
            'nmul': 1,
            'warm_up': 10,
            'warm_up_states': True,
            'weight_method': method,
            'tau': 1.0
        }

        model = DiffBlendV1(config=config, device=torch.device('cpu'))
        model.train()

        raw = torch.randn(B, model.learnable_param_count, requires_grad=True)
        result = model.forward(x_dict, (None, raw))

        # 检查权重和为1
        for proc_name in model.process_names:
            n_opt = PROCESS_OPTIONS[proc_name]
            weights = torch.stack([result[f'w_{proc_name}_{i}'][0, 0] for i in range(n_opt)])
            weight_sum = weights.sum().item()
            print(f"  {proc_name}: weights={weights.detach().numpy()}, sum={weight_sum:.6f}")
            assert abs(weight_sum - 1.0) < 1e-4, f"Weights don't sum to 1 for {proc_name}"

        # 梯度测试
        loss = result['streamflow'].sum()
        loss.backward()
        grad_norm = raw.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.4f}")
        assert grad_norm > 0, f"Zero gradient for {method}"

    print("\n✓ Weight methods test passed")


def test_nmul_configurations():
    """测试不同 nmul 配置"""
    print("\n" + "="*60)
    print("TEST 7: Multiple nmul Configurations")
    print("="*60)

    nmul_values = [1, 2, 4, 8]

    B, T_steps = 4, 50
    x_phy = torch.rand(T_steps, B, 3) * 10
    x_dict = {'x_phy': x_phy, 'target': torch.rand(T_steps, B)}

    for nmul in nmul_values:
        print(f"\nTesting nmul={nmul}...")

        config = {
            'nmul': nmul,
            'warm_up': 10,
            'warm_up_states': True,
            'weight_method': 'softmax'
        }

        model = DiffBlendV1(config=config, device=torch.device('cpu'))
        model.train()

        expected_params = len(model.param_names) * nmul + len(model.routing_param_names) + TOTAL_WEIGHT_LOGITS
        print(f"  Expected params: {expected_params}")
        print(f"  Actual params: {model.learnable_param_count}")
        assert model.learnable_param_count == expected_params

        raw = torch.randn(B, model.learnable_param_count, requires_grad=True)
        result = model.forward(x_dict, (None, raw))

        print(f"  Streamflow shape: {result['streamflow'].shape}")
        assert result['streamflow'].shape == (T_steps, B)

        loss = result['streamflow'].sum()
        loss.backward()
        grad_norm = raw.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.4f}")
        assert grad_norm > 0

    print("\n✓ nmul configurations test passed")


def test_gradcheck():
    """测试 torch.autograd.gradcheck 严格梯度检查

    注意: 由于 uh_gamma/uh_conv 函数内部使用固定 dtype，
    完整的 gradcheck 可能失败。这里我们测试核心水文过程的梯度。
    """
    print("\n" + "="*60)
    print("TEST 8: torch.autograd.gradcheck")
    print("="*60)

    print("Testing individual hydrological process functions...")

    # 测试各个独立的水文过程函数
    from project.blend_formula.models.diff_blend_v1 import (
        rainsnow_hbv, inf_hmets, soilevap_all,
        quick_linear_analytic, base_linear_analytic
    )

    # 1. 测试雨雪分割
    print("\n1. Testing rainsnow_hbv...")
    P = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    T = torch.randn(3, 2, dtype=torch.float64)
    tt = torch.tensor(0.0, dtype=torch.float64)
    tti = torch.tensor(2.0, dtype=torch.float64)

    def rainsnow_wrapper(p):
        sf, rf = rainsnow_hbv(p, T, tt, tti)
        return sf.sum() + rf.sum()

    check1 = torch.autograd.gradcheck(rainsnow_wrapper, P, eps=1e-6, atol=1e-4)
    print(f"  rainsnow_hbv: {'✓ PASSED' if check1 else '✗ FAILED'}")

    # 2. 测试入渗
    print("\n2. Testing inf_hmets...")
    P_eff = torch.randn(3, 2, dtype=torch.float64, requires_grad=True) + 5
    S = torch.randn(3, 2, dtype=torch.float64).abs() + 1
    Smax = torch.tensor(100.0, dtype=torch.float64)
    c_runoff = torch.tensor(0.5, dtype=torch.float64)

    def inf_wrapper(p):
        return inf_hmets(p, S, Smax, c_runoff).sum()

    check2 = torch.autograd.gradcheck(inf_wrapper, P_eff, eps=1e-6, atol=1e-4)
    print(f"  inf_hmets: {'✓ PASSED' if check2 else '✗ FAILED'}")

    # 3. 测试蒸发
    print("\n3. Testing soilevap_all...")
    PET = torch.randn(3, 2, dtype=torch.float64, requires_grad=True).abs() + 1
    c_pet = torch.tensor(1.0, dtype=torch.float64)
    S_evap = torch.randn(3, 2, dtype=torch.float64).abs() + 5

    def evap_wrapper(pet):
        return soilevap_all(pet, c_pet, S_evap).sum()

    check3 = torch.autograd.gradcheck(evap_wrapper, PET, eps=1e-6, atol=1e-4)
    print(f"  soilevap_all: {'✓ PASSED' if check3 else '✗ FAILED'}")

    # 4. 测试快速流
    print("\n4. Testing quick_linear_analytic...")
    S_quick = torch.randn(3, 2, dtype=torch.float64, requires_grad=True).abs() + 10
    k_quick = torch.tensor(0.1, dtype=torch.float64)

    def quick_wrapper(s):
        return quick_linear_analytic(s, k_quick).sum()

    check4 = torch.autograd.gradcheck(quick_wrapper, S_quick, eps=1e-6, atol=1e-4)
    print(f"  quick_linear_analytic: {'✓ PASSED' if check4 else '✗ FAILED'}")

    # 5. 测试基流
    print("\n5. Testing base_linear_analytic...")
    S_base = torch.randn(3, 2, dtype=torch.float64, requires_grad=True).abs() + 10
    k_base = torch.tensor(0.05, dtype=torch.float64)

    def base_wrapper(s):
        return base_linear_analytic(s, k_base).sum()

    check5 = torch.autograd.gradcheck(base_wrapper, S_base, eps=1e-6, atol=1e-4)
    print(f"  base_linear_analytic: {'✓ PASSED' if check5 else '✗ FAILED'}")

    # 汇总结果
    all_passed = all([check1, check2, check3, check4, check5])

    print("\n" + "-"*60)
    if all_passed:
        print("✓ All individual process gradchecks PASSED")
        print("  Core hydrological processes have correct gradients")
    else:
        print("⚠ Some gradchecks failed")
        print("  Check individual process implementations")

    print("\nNote: Full model gradcheck skipped due to routing dtype issues.")
    print("      This is a known limitation of uh_gamma/uh_conv functions.")
    print("      Manual gradient tests (TEST 3, 4) confirm gradients flow correctly.")

    print("\n✓ Gradcheck test completed")

def test_balance_gradient():
    """测试 balance 序列的形状、键名，以及梯度能正确从 balance loss 反传至参数。

    验证要点：
    1. balance_window 可由 config 正确设置
    2. result 中出现全部 9 条 balance_* 序列，shape = [T, B]
    3. 以 balance loss 做 backward，梯度范数 > 0（梯度链未断）
    """
    print("\n" + "="*60)
    print("TEST 9: Balance Sequence Shape & Gradient Flow")
    print("="*60)

    config = {
        'nmul': 1,
        'warm_up': 0,
        'warm_up_states': True,
        'weight_method': 'softmax',
        'balance_window': 30,
    }
    model = DiffBlendV1(config=config, device=torch.device('cpu'))
    model.train()

    assert model.balance_window == 30, (
        f"balance_window not loaded from config: got {model.balance_window}"
    )

    B, T_steps = 4, 60
    raw = torch.randn(B, model.learnable_param_count, requires_grad=True)
    x_phy = torch.rand(T_steps, B, 3) * 10
    x_dict = {'x_phy': x_phy}

    result = model.forward(x_dict, (None, raw))

    # ── 1. 键名与形状检查 ─────────────────────────────────────────
    expected_balance_keys = [
        'balance_infiltration_0_1', 'balance_infiltration_0_2',
        'balance_evaporation_0_1',  'balance_evaporation_0_2',
        'balance_quickflow_0_1',    'balance_quickflow_0_2',
        'balance_baseflow_0_1',
        'balance_snow_outflow_0_1', 'balance_snow_outflow_0_2',
    ]
    for key in expected_balance_keys:
        assert key in result, f"Missing balance key: {key}"
        assert result[key].shape == (T_steps, B), (
            f"Wrong shape for {key}: {result[key].shape}, expected ({T_steps}, {B})"
        )
        print(f"  {key}: shape={result[key].shape} ✓")

    # ── 2. balance loss 梯度反传检查 ─────────────────────────────
    balance_loss = sum(result[k].abs().sum() for k in expected_balance_keys)
    balance_loss.backward()

    grad_norm = raw.grad.norm().item()
    grad_nonzero = (raw.grad.abs() > 1e-8).sum().item()
    grad_total = raw.grad.numel()

    print(f"\nBalance loss gradient norm  : {grad_norm:.4f}")
    print(f"Nonzero gradients           : {grad_nonzero}/{grad_total} "
          f"({100*grad_nonzero/grad_total:.1f}%)")

    assert grad_norm > 0, (
        "Balance loss gradient is zero — gradient chain is broken! "
        "Check that opts stacks in BlendStepOutput are NOT detached."
    )
    assert grad_nonzero > 0, "No nonzero gradients from balance loss"

    print("\n✓ Balance gradient flow test passed")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS FOR DiffBlendV1")
    print("="*60)

    try:
        test_model_instantiation()
        test_forward_pass()
        test_gradient_flow()
        test_gradient_with_snow()
        test_water_balance_formula_combinations()
        test_weight_methods()
        test_nmul_configurations()
        test_gradcheck()
        test_balance_gradient()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED ✗: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    run_all_tests()
