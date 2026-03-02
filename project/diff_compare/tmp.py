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

# ------------------------- 可配置区 -------------------------
nmul = 16
model_name = "collie1"
param_num_map = {"collie1": 1, "hymod": 5, "xinanajing": 12}
num_basins = 559
epochs = [0, 1] + list(range(5, 101, 5))  # 如需调整，修改此列表
base_dir = "/workspace/my_deltamodel/project/diff_compare/ablation/nmul/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B200_n16_noLn_noWU_42/collie1/KgeLoss/stat"
filename_tpl = "dUnifyV2_Ep{epoch}.pt"
save_path = os.path.join(os.path.dirname(__file__), "npz", "param_snapshots_collie1_n16.npz")
# -----------------------------------------------------------


def load_params(path: str):
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
        raise ValueError(f"Unexpected tensor dim {tensor.ndim} for {path}, expect 3")
    if tensor.shape[-1] != nmul:
        raise ValueError(f"nmul mismatch: got {tensor.shape[-1]}, expect {nmul} for {path}")
    return tensor


def build_init_params():
    """使用 LatinHypercube 生成与训练一致的初始化 (作为 epoch 0)。"""
    param_num = param_num_map.get(model_name)
    if param_num is None:
        raise KeyError(f"param_num_map missing entry for {model_name}")

    sampler = qmc.LatinHypercube(d=param_num)
    total_samples = num_basins * nmul
    sample_np = sampler.random(n=total_samples)

    u = torch.from_numpy(sample_np).float()  # [total, param_num]
    u = u.view(num_basins, nmul, param_num).transpose(1, 2)  # [num_basins, param_num, nmul]
    u = u * 0.9 + 0.05  # 防止 logit 溢出
    init_val = torch.log(u / (1 - u))
    return init_val


def main():
    tensors = []
    loaded_epochs = []
    missing = []
    for ep in epochs:
        if ep == 0:
            tensor = build_init_params()
            tensors.append(tensor)
            loaded_epochs.append(ep)
            continue

        fpath = os.path.join(base_dir, filename_tpl.format(epoch=ep))
        try:
            tensor = load_params(fpath)
        except FileNotFoundError:
            missing.append(ep)
            continue
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


if __name__ == "__main__":
    main()