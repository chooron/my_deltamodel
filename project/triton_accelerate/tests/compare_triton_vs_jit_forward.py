"""
Compare HBV Triton core and JIT core forward results on real CAMELS forcing.
Loads the pickled forcing/target arrays, runs a short forward pass for a single
basin, and reports the per-variable differences for q and key states.
"""

from pathlib import Path
import pickle
from typing import Dict

import torch
import sys
sys.path.append("/workspace/my_deltamodel")
from project.triton_accelerate.models.hbv_jit_core import hbv_timestep_loop
from project.triton_accelerate.models.hbv_triton_core import hbv_step_triton


def find_camels_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "data" / "camels_dataset",
        Path("/workspace/my_deltamodel/data/camels_dataset"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("camels_dataset not found in expected locations")


def load_forcing(device: torch.device) -> torch.Tensor:
    data_path = find_camels_path()
    with open(data_path, "rb") as f:
        forcing, target, _ = pickle.load(f)
    forcing_torch = torch.from_numpy(forcing).to(device=device, dtype=torch.float32)
    forcing_torch = forcing_torch.permute(1, 0, 2)  # (T, B, 3)
    return forcing_torch


def build_params(device: torch.device) -> Dict[str, torch.Tensor]:
    base_shape = (1, 1)
    return {
        "parTT": torch.zeros(base_shape, device=device),
        "parCFMAX": torch.full(base_shape, 2.5, device=device),
        "parCFR": torch.full(base_shape, 0.05, device=device),
        "parCWH": torch.full(base_shape, 0.1, device=device),
        "parFC": torch.full(base_shape, 300.0, device=device),
        "parBETA": torch.full(base_shape, 2.0, device=device),
        "parLP": torch.full(base_shape, 0.7, device=device),
        "parBETAET": torch.full(base_shape, 1.5, device=device),
        "parC": torch.full(base_shape, 0.05, device=device),
        "parPERC": torch.full(base_shape, 1.5, device=device),
        "parK0": torch.full(base_shape, 0.15, device=device),
        "parK1": torch.full(base_shape, 0.05, device=device),
        "parK2": torch.full(base_shape, 0.01, device=device),
        "parUZL": torch.full(base_shape, 20.0, device=device),
    }


def run_jit(P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor, params: Dict[str, torch.Tensor]):
    snow0 = torch.zeros_like(P[0])
    melt0 = torch.zeros_like(P[0])
    sm0 = torch.zeros_like(P[0])
    suz0 = torch.zeros_like(P[0])
    slz0 = torch.zeros_like(P[0])

    return hbv_timestep_loop(
        P,
        T,
        PET,
        snow0,
        melt0,
        sm0,
        suz0,
        slz0,
        params["parTT"],
        params["parCFMAX"],
        params["parCFR"],
        params["parCWH"],
        params["parFC"],
        params["parBETA"],
        params["parLP"],
        params["parBETAET"],
        params["parC"],
        params["parPERC"],
        params["parK0"],
        params["parK1"],
        params["parK2"],
        params["parUZL"],
        nearzero=1e-6,
    )


def run_triton(P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor, params: Dict[str, torch.Tensor]):
    snow = torch.zeros_like(P[0])
    melt = torch.zeros_like(P[0])
    sm = torch.zeros_like(P[0])
    suz = torch.zeros_like(P[0])
    slz = torch.zeros_like(P[0])

    snow_hist = []
    sm_hist = []
    suz_hist = []
    slz_hist = []
    q_hist = []

    with torch.no_grad():
        for t in range(P.shape[0]):
            snow, melt, sm, suz, slz, q = hbv_step_triton(
                P[t],
                T[t],
                PET[t],
                snow,
                melt,
                sm,
                suz,
                slz,
                params["parTT"],
                params["parCFMAX"],
                params["parCFR"],
                params["parCWH"],
                params["parFC"],
                params["parBETA"],
                params["parLP"],
                params["parBETAET"],
                params["parC"],
                params["parPERC"],
                params["parK0"],
                params["parK1"],
                params["parK2"],
                params["parUZL"],
            )
            snow_hist.append(snow)
            sm_hist.append(sm)
            suz_hist.append(suz)
            slz_hist.append(slz)
            q_hist.append(q)

    snow_stack = torch.stack(snow_hist, dim=0)
    sm_stack = torch.stack(sm_hist, dim=0)
    suz_stack = torch.stack(suz_hist, dim=0)
    slz_stack = torch.stack(slz_hist, dim=0)
    q_stack = torch.stack(q_hist, dim=0)

    return snow_stack, sm_stack, suz_stack, slz_stack, q_stack


def summarize_diff(name: str, ref: torch.Tensor, test: torch.Tensor) -> None:
    diff = (ref - test).abs()
    print(f"{name:10s} max abs diff: {diff.max().item():.6f} | mean abs diff: {diff.mean().item():.6f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemError("CUDA is required to run the Triton core")

    forcing = load_forcing(device)
    params = build_params(device)

    basin_idx = 0
    step_count = 10000  # adjust if you want a longer comparison

    subset = forcing[:step_count, basin_idx : basin_idx + 1]
    P = subset[:, :, 0:1]
    T = subset[:, :, 1:2]
    PET = subset[:, :, 2:3]

    (
        q_jit,
        q0_jit,
        q1_jit,
        q2_jit,
        aet_jit,
        recharge_jit,
        excs_jit,
        evapfactor_jit,
        tosoil_jit,
        perc_jit,
        swe_jit,
        sm_jit,
        capillary_jit,
        soil_wetness_jit,
        snow_final_jit,
        melt_final_jit,
        sm_final_jit,
        suz_final_jit,
        slz_final_jit,
    ) = run_jit(P, T, PET, params)

    snow_triton, sm_triton, suz_triton, slz_triton, q_triton = run_triton(P, T, PET, params)

    print("=== Forward comparison (JIT reference vs Triton) ===")
    summarize_diff("q", q_jit, q_triton)
    summarize_diff("snow", swe_jit, snow_triton)
    summarize_diff("sm", sm_jit, sm_triton)
    summarize_diff("suz_final", suz_final_jit, suz_triton[-1])
    summarize_diff("slz_final", slz_final_jit, slz_triton[-1])

    print("\nSample q (first 5 timesteps):")
    for i in range(min(5, step_count)):
        jit_val = q_jit[i, 0, 0].item()
        triton_val = q_triton[i, 0, 0].item()
        print(f"t={i:3d} | jit={jit_val:.6f} | triton={triton_val:.6f} | diff={abs(jit_val - triton_val):.6f}")


if __name__ == "__main__":
    main()
