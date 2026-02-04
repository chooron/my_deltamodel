import os
import sys
import pickle
import pandas as pd
import torch
import numpy as np
import json
from tqdm import tqdm

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dmg.models.hydromodel import STFN_INFO, INIT_INFO  # noqa


def run_batch_test_objs(model_keyword="xinanjiang"):
    # --- 1. 配置路径 ---
    data_dir = os.path.join(project_root, "data")
    forcing_v2_path = os.path.join(data_dir, "camels_forcing_v2.pkl")
    gage_id_path = os.path.join(data_dir, "gage_id.npy")
    subset_id_path = os.path.join(data_dir, "559sub_id.txt")
    params_dir = os.path.join(project_root, "project/diff_compare/tests/params")
    save_dir = os.path.join(project_root, "project/diff_compare/tests/sim")

    output_path = os.path.join(save_dir, f"{model_keyword}_sim_results.npz")
    if os.path.exists(output_path):
        print(
            f"Result file {output_path} already exists, skipping simulation for {model_keyword}."
        )
        return

    # 动态搜索参数文件
    obj_files = sorted(
        [
            f
            for f in os.listdir(params_dir)
            if (model_keyword + "_") in f and f.endswith(".csv")
        ]
    )
    print(f"Found and sorted parameter files: {obj_files}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 加载基础 ID 和映射 ---
    print("Loading Basin IDs and Subset...")
    all_gage_ids = np.load(gage_id_path, allow_pickle=True)
    all_gage_ids = [str(int(float(gid))).zfill(8) for gid in all_gage_ids]

    with open(subset_id_path, "r") as f:
        target_ids = json.load(f)
    target_ids = [str(tid).zfill(8) for tid in target_ids]

    # 建立映射
    id_to_idx = {gid: i for i, gid in enumerate(all_gage_ids)}
    target_indices = [id_to_idx[tid] for tid in target_ids if tid in id_to_idx]
    valid_ids = [tid for tid in target_ids if tid in id_to_idx]

    num_basins = len(valid_ids)
    print(f"Total target basins: {num_basins}")

    # --- 3. 加载 Forcing ---
    print(f"Loading forcing from {forcing_v2_path}...")
    with open(forcing_v2_path, "rb") as f:
        data_v2 = pickle.load(f)

    forcing_all = data_v2["forcing"][target_indices]
    dates = pd.to_datetime(data_v2["dates"])

    target_start = 1989
    target_end = 2009
    time_mask = (dates.year >= target_start) & (dates.year <= target_end)

    forcing = forcing_all[:, time_mask, :]
    num_steps = forcing.shape[1]
    print(f"Time steps: {num_steps} ({target_start}-{target_end})")

    # --- 4. 加载参数 ---
    print("Loading parameters...")
    all_obj_p_tensors = []

    for obj_file in obj_files:
        csv_path = os.path.join(params_dir, obj_file)
        df = pd.read_csv(csv_path)
        df["gauge_id_str"] = df["gauge_id"].apply(
            lambda x: str(int(x)).zfill(8)
        )
        df = df.set_index("gauge_id_str")
        df_ordered = df.reindex(valid_ids)

        # 提取参数 (跳过前两列 ID)
        p_block = torch.tensor(
            df_ordered.iloc[:, 3:].values, device=device, dtype=torch.float32
        )
        all_obj_p_tensors.append(p_block)

    full_p_tensor = torch.cat(all_obj_p_tensors, dim=0)
    total_batch_size = full_p_tensor.shape[0]

    # 转换为 list 供模型 unpack
    params = [p.unsqueeze(1) for p in torch.unbind(full_p_tensor, dim=1)]

    # --- 5. 准备模拟输入 ---
    forcing_torch = torch.tensor(forcing, device=device, dtype=torch.float32)
    forcing_torch = forcing_torch.repeat(3, 1, 1)
    print(forcing_torch.shape)

    model_step = STFN_INFO[model_keyword]
    create_initial_state = INIT_INFO[model_keyword]
    curr_states = list(create_initial_state(total_batch_size, 1, device))
    Q_sim_list = []

    # --- 6. 运行模拟 ---
    print(f"Starting simulation (Batch: {total_batch_size})...")
    for t in tqdm(range(num_steps), desc="Simulating"):
        P_t = forcing_torch[:, t, 0:1]
        T_t = forcing_torch[:, t, 1:2]
        PET_t = forcing_torch[:, t, 2:3]

        Qsim, Ea, *curr_states = model_step(
            P_t, T_t, PET_t, *params, *curr_states
        )
        Q_sim_list.append(Qsim.detach().cpu().numpy())

    # --- 7. 数据重组 ---
    Q_sim_all = np.concatenate(Q_sim_list, axis=1)  # [N*3, T]
    Q_sim_split = Q_sim_all.reshape(3, num_basins, num_steps)  # [3, N, T]
    final_results = np.transpose(Q_sim_split, (1, 2, 0))  # [N, T, 3]

    np.savez_compressed(output_path, data=final_results)

    print("\n" + "=" * 40)
    print("Simulation Complete.")


if __name__ == "__main__":
    for model_keyword in STFN_INFO.keys():
        run_batch_test_objs(model_keyword)
