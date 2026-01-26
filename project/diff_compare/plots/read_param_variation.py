"""
提取同一 nmul 下多个 epoch 的 nn_model.params，并沿新维度拼接后保存为 npz。

默认提取 epoch: 0、1、5、10、15、20...100，可按需修改 epochs 列表。
输出 npz 包含:
- epochs: 实际成功读取的 epoch 列表
- params: shape [num_epoch, 559, 1, nmul]
"""

import os
import numpy as np
import torch
from scipy.stats import qmc

COLLIE1_PARAMS_BOUNDS = {
    "Smax": [1.0, 2000.0],
}

HYMOD_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "b_exp": [0.0, 10.0],  # Soil depth distribution parameter [-]
    "a_split": [0.0, 1.0],  # Runoff distribution fraction [-]
    "kf": [0.0, 1.0],  # Fast flow time parameter [d-1]
    "ks": [0.0, 1.0],  # Base flow time parameter [d-1]
}

XINANJIANG_PARAMS_BOUNDS = {
    "aim": [0.0, 1.0],
    "par_a": [-0.49, 0.49],
    "par_b": [0.0, 10.0],
    "stot": [1.0, 2000.0],
    "fwm": [0.01, 0.99],
    "flm": [0.01, 0.99],
    "par_c": [0.01, 0.99],
    "ex": [0.0, 10.0],
    "ki": [0.0, 1.0],
    "kg": [0.0, 1.0],
    "ci": [0.0, 1.0],
    "cg": [0.0, 1.0],
}

HBV96_PARAMS_BOUNDS = {
    "tt": [-3.0, 5.0],           # TT, threshold temperature for snowfall [oC]
    "tti": [0.0, 17.0],          # TTI, interval length of rain-snow spectrum [oC]
    "ttm": [-3.0, 3.0],          # TTM, threshold temperature for snowmelt [oC]
    "cfr": [0.0, 1.0],           # CFR, coefficient of refreezing of melted snow [-]
    "cfmax": [0.0, 20.0],        # CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
    "whc": [0.0, 1.0],           # WHC, maximum water holding content of snow pack [-]
    "cflux": [0.0, 4.0],         # CFLUX, maximum rate of capillary rise [mm/d]
    "fc": [1.0, 2000.0],         # FC, maximum soil moisture storage [mm]
    "lp": [0.05, 0.95],          # LP, wilting point as fraction of FC [-]
    "beta": [0.0, 10.0],         # BETA, non-linearity coefficient of upper zone recharge [-]
    "k0": [0.0, 1.0],            # K0, runoff coefficient from upper zone [d-1]
    "alpha": [0.0, 4.0],         # ALPHA, non-linearity coefficient of runoff from upper zone [-]
    "perc": [0.0, 20.0],         # PERC, maximum rate of percolation to lower zone [mm/d]
    "k1": [0.0, 1.0],            # K1, runoff coefficient from lower zone [d-1]
    "maxbas": [1.0, 120.0],      # MAXBAS, flow routing delay [d]
}

# ------------------------- 可配置区 -------------------------
param_num_map = {"collie1": 1, "hymod": 5, "xinanjiang": 12, "hbv96": 15}
param_bounds_map = {
    "collie1": COLLIE1_PARAMS_BOUNDS,
    "hymod": HYMOD_PARAMS_BOUNDS,
    "xinanjiang": XINANJIANG_PARAMS_BOUNDS,
    "hbv96": HBV96_PARAMS_BOUNDS,
}
num_basins = 559
epochs = [0, 1] + list(range(5, 101, 5))  # 如需调整，修改此列表
nmul_list = [16, 32, 64, 128, 256]
models = ["hbv96"]

base_dir_tpl = (
    "/workspace/my_deltamodel/project/diff_compare/ablation/nmul/camels_559/train1989-1998/no_multi/"
    "Calibrate_E100_R365_B200_n{nmul}_noLn_noWU_42/{model}/KgeLoss/stat"
)
filename_tpl = "dUnifyV2_Ep{epoch}.pt"
output_dir = os.path.join(os.path.dirname(__file__), "npz")
os.makedirs(output_dir, exist_ok=True)
# -----------------------------------------------------------


def load_params(path: str, nmul: int):
    """读取单个 checkpoint 的 nn_model.params 张量。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "nn_model.params" not in ckpt:
            raise KeyError(f"Missing 'nn_model.params' in {path}")
        tensor = ckpt["nn_model.params"]
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")
    if tensor.ndim != 3:
        raise ValueError(
            f"Unexpected tensor dim {tensor.ndim} for {path}, expect 3"
        )
    if tensor.shape[-1] != nmul:
        raise ValueError(
            f"nmul mismatch: got {tensor.shape[-1]}, expect {nmul} for {path}"
        )
    return tensor


def build_init_params(model_name: str, nmul: int):
    """使用 LatinHypercube 生成与训练一致的初始化 (作为 epoch 0)。"""
    param_num = param_num_map.get(model_name)
    if param_num is None:
        raise KeyError(f"param_num_map missing entry for {model_name}")

    sampler = qmc.LatinHypercube(d=param_num)
    total_samples = num_basins * nmul
    sample_np = sampler.random(n=total_samples)

    u = torch.from_numpy(sample_np).float()  # [total, param_num]
    u = u.view(num_basins, nmul, param_num).transpose(
        1, 2
    )  # [num_basins, param_num, nmul]
    u = u * 0.9 + 0.05  # 防止 logit 溢出
    init_val = torch.log(u / (1 - u))
    return init_val


def descale_params(tensor: torch.Tensor, model_name: str) -> torch.Tensor:
    """Sigmoid -> scale to parameter bounds for the given model."""
    bounds = param_bounds_map.get(model_name)
    if bounds is None:
        raise KeyError(f"param_bounds_map missing entry for {model_name}")

    names = list(bounds.keys())
    if tensor.shape[1] != len(names):
        raise ValueError(
            f"Param count mismatch for {model_name}: tensor has {tensor.shape[1]}, bounds has {len(names)}"
        )

    tensor = torch.sigmoid(tensor)  # map logits to [0,1]
    for i, name in enumerate(names):
        low, high = bounds[name]
        tensor[:, i, :] = tensor[:, i, :] * (high - low) + low
    return tensor


def process_single(model_name: str, nmul: int):
    base_dir = base_dir_tpl.format(nmul=nmul, model=model_name)
    save_path = os.path.join(
        output_dir, f"param_snapshots_{model_name}_n{nmul}.npz"
    )

    tensors = []
    loaded_epochs = []
    missing = []
    for ep in epochs:
        if ep == 0:
            tensor = build_init_params(model_name, nmul)
            tensor = descale_params(tensor, model_name)
            tensors.append(tensor)
            loaded_epochs.append(ep)
            continue

        fpath = os.path.join(base_dir, filename_tpl.format(epoch=ep))
        try:
            tensor = load_params(fpath, nmul)
        except FileNotFoundError:
            missing.append(ep)
            continue
        tensor = descale_params(tensor, model_name)
        tensors.append(tensor)
        loaded_epochs.append(ep)

    if not tensors:
        raise RuntimeError("No checkpoint loaded; check paths/epochs")

    stacked = torch.stack(tensors, dim=0).numpy()  # [num_epoch, 559, 1, nmul]
    np.savez(save_path, epochs=np.array(loaded_epochs), params=stacked)

    print(f"Saved {len(loaded_epochs)} epochs to {save_path}")
    print(f"params shape: {stacked.shape}")
    if missing:
        print(f"Missing epochs (file not found): {missing}")
    return missing


def main():
    all_missing = {}
    for model_name in models:
        for nmul in nmul_list:
            try:
                missing = process_single(model_name, nmul)
                if missing:
                    all_missing[(model_name, nmul)] = missing
            except Exception as exc:  # noqa: BLE001
                print(f"[{model_name}, n{nmul}] failed: {exc}")

    if all_missing:
        print("Summary of missing epochs:")
        for (model_name, nmul), miss in all_missing.items():
            print(f"  {model_name}, n{nmul}: {miss}")


if __name__ == "__main__":
    main()
