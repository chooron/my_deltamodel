import torch
import sys
sys.path.append("/workspace/my_deltamodel")
from project.triton_accelerate.models.hbv_manual_core import (
    SnowBlock,
    SoilBlock,
    RoutingBlock,
    _snow_block_formula,
    _soil_block_formula,
    _routing_block_formula,
)


def _compare_block(name, apply_fn, ref_fn, inputs, atol=1e-6, rtol=1e-6):
    # clone inputs for manual branch
    manual_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]
    ref_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]

    # manual forward/backward
    manual_out = apply_fn(*manual_inputs)
    if isinstance(manual_out, torch.Tensor):
        manual_loss = manual_out.sum()
    else:
        manual_loss = torch.stack([o if o.ndim == 0 else o.sum() for o in manual_out]).sum()
    manual_loss.backward()

    # reference forward/backward (pure autograd)
    with torch.enable_grad():
        ref_out = ref_fn(*ref_inputs)
        if isinstance(ref_out, torch.Tensor):
            ref_loss = ref_out.sum()
        else:
            ref_loss = torch.stack([o if o.ndim == 0 else o.sum() for o in ref_out]).sum()
    ref_loss.backward()

    # compare gradients
    diffs = []
    for mi, ri in zip(manual_inputs, ref_inputs):
        mg = mi.grad
        rg = ri.grad
        if mg is None or rg is None:
            diffs.append(float('inf'))
            continue
        diff = torch.max(torch.abs(mg - rg) / (torch.abs(rg) + 1e-12)).item()
        diffs.append(diff)
    max_diff = max(diffs)
    ok = max_diff <= max(atol, rtol)
    print(f"[{name}] max relative diff={max_diff:.3e} -> {'OK' if ok else 'FAIL'}")
    return ok, diffs


def test_snow_block(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    # avoid threshold: temp 2.0 >0, snow positive
    p = torch.full((5,), 5.0, device=device, dtype=dtype)
    t = torch.full((5,), 2.0, device=device, dtype=dtype)
    snow = torch.full((5,), 3.0, device=device, dtype=dtype)
    melt = torch.full((5,), 1.0, device=device, dtype=dtype)
    tt = torch.full((5,), 0.5, device=device, dtype=dtype)
    cfmax = torch.full((5,), 2.0, device=device, dtype=dtype)
    cfr = torch.full((5,), 0.05, device=device, dtype=dtype)
    cwh = torch.full((5,), 0.1, device=device, dtype=dtype)

    return _compare_block(
        "snow",
        lambda *args: SnowBlock.apply(*args),
        lambda *args: _snow_block_formula(*args),
        [p, t, snow, melt, tt, cfmax, cfr, cwh],
    )


def test_soil_block(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    sm = torch.full((5,), 60.0, device=device, dtype=dtype)
    rain = torch.full((5,), 2.0, device=device, dtype=dtype)
    tosoil = torch.full((5,), 1.0, device=device, dtype=dtype)
    pet = torch.full((5,), 1.5, device=device, dtype=dtype)
    fc = torch.full((5,), 150.0, device=device, dtype=dtype)
    beta = torch.full((5,), 1.5, device=device, dtype=dtype)
    lp = torch.full((5,), 0.7, device=device, dtype=dtype)
    betaet = torch.full((5,), 1.2, device=device, dtype=dtype)

    return _compare_block(
        "soil",
        lambda *args: SoilBlock.apply(*args),
        lambda *args: _soil_block_formula(*args),
        [sm, rain, tosoil, pet, fc, beta, lp, betaet],
    )


def test_routing_block(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    sm_after = torch.full((5,), 40.0, device=device, dtype=dtype)
    suz = torch.full((5,), 20.0, device=device, dtype=dtype)
    slz = torch.full((5,), 15.0, device=device, dtype=dtype)
    recharge = torch.full((5,), 3.0, device=device, dtype=dtype)
    excess = torch.full((5,), 1.0, device=device, dtype=dtype)
    perc = torch.full((5,), 3.0, device=device, dtype=dtype)
    k0 = torch.full((5,), 0.25, device=device, dtype=dtype)
    k1 = torch.full((5,), 0.05, device=device, dtype=dtype)
    k2 = torch.full((5,), 0.01, device=device, dtype=dtype)
    uzl = torch.full((5,), 5.0, device=device, dtype=dtype)

    return _compare_block(
        "routing",
        lambda *args: RoutingBlock.apply(*args),
        lambda *args: _routing_block_formula(*args),
        [sm_after, suz, slz, recharge, excess, perc, k0, k1, k2, uzl],
    )


if __name__ == "__main__":
    all_ok = True
    for fn in (test_snow_block, test_soil_block, test_routing_block):
        ok, _ = fn()
        all_ok = all_ok and ok
    print(f"Overall: {'OK' if all_ok else 'FAIL'}")
