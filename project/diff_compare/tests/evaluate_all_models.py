import numpy as np
import xarray as xr
import os
import sys
import pandas as pd
from tqdm import tqdm

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)


def calc_nse(obs, sim):
    """
    计算 Nash-Sutcliffe Efficiency (NSE)，此处作为 R2 使用。
    obs: 基准数据 (MATLAB)
    sim: 测试数据 (Python)
    """
    # 过滤无效值
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs_c = obs[mask]
    sim_c = sim[mask]

    if len(obs_c) < 2:
        return np.nan

    ss_res = np.sum((obs_c - sim_c) ** 2)
    ss_tot = np.sum((obs_c - np.mean(obs_c)) ** 2)

    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0

    return 1 - (ss_res / ss_tot)


def evaluate_models():
    save_dir = os.path.join(project_root, "project/diff_compare/tests/sim")
    matlab_results_dir = r"G:\Dataset\MARRMoTResult"

    if not os.path.exists(save_dir):
        print(f"Error: Simulation directory {save_dir} not found.")
        return

    # 获取所有 Python 模拟结果文件
    sim_files = [
        f for f in os.listdir(save_dir) if f.endswith("_sim_results.npz")
    ]

    results_list = []

    for sim_file in tqdm(sim_files, desc="Evaluating models"):
        model_keyword = sim_file.replace("_sim_results.npz", "")
        py_path = os.path.join(save_dir, sim_file)

        # 1. 加载 Python 结果 (Basins, Time, Objs)
        try:
            loaded = np.load(py_path)
            py_results = loaded["data"]
        except Exception as e:
            print(f"Error loading {py_path}: {e}")
            continue

        # 2. 寻找对应的 MATLAB 结果
        matlab_files = [
            f
            for f in os.listdir(matlab_results_dir)
            if model_keyword in f and f.endswith(".nc")
        ]

        if not matlab_files:
            print(f"Warning: No MATLAB benchmark found for {model_keyword}")
            continue

        # 使用第一个匹配的文件
        mat_path = os.path.join(matlab_results_dir, matlab_files[0])

        try:
            ds = xr.open_dataset(mat_path)
            # MATLAB shape: (Objs, Time, Basins)
            mat_q_raw = ds["Sim_q"].values
            # 转置为 (Basins, Time, Objs)
            mat_results = np.transpose(mat_q_raw, (2, 1, 0))
            ds.close()
        except Exception as e:
            print(f"Error reading MATLAB file {mat_path}: {e}")
            continue

        # 3. 对齐时间步长 (如果不一致则截断)
        min_steps = min(py_results.shape[1], mat_results.shape[1])
        py_results = py_results[:, :min_steps, :]
        mat_results = mat_results[:, :min_steps, :]

        # 4. 计算每个 Obj 的平均 R2
        num_objs = py_results.shape[2]
        obj_r2s = []

        for obj_idx in range(num_objs):
            basin_r2s = []
            for b_idx in range(py_results.shape[0]):
                r2 = calc_nse(
                    mat_results[b_idx, :, obj_idx],
                    py_results[b_idx, :, obj_idx],
                )
                if not np.isnan(r2):
                    basin_r2s.append(r2)

            avg_r2 = np.median(basin_r2s) if basin_r2s else np.nan
            obj_r2s.append(avg_r2)

        # 5. 汇总
        row = {
            "Model": model_keyword,
        }
        for i, r2_val in enumerate(obj_r2s):
            row[f"Obj{i + 1}_R2"] = r2_val

        row["Mean_R2"] = np.nanmean(obj_r2s) if obj_r2s else np.nan
        results_list.append(row)

    # 6. 保存为 CSV
    df_results = pd.DataFrame(results_list)

    # 保留 4 位小数
    numeric_cols = [c for c in df_results.columns if c != "Model"]
    df_results[numeric_cols] = df_results[numeric_cols].round(4)

    output_csv = os.path.join(
        project_root, "project/diff_compare/tests/model_comparison_r2.csv"
    )
    df_results.to_csv(output_csv, index=False)

    print("\n" + "=" * 40)
    print(f"Evaluation Complete!")
    print(f"Summary saved to: {output_csv}")
    print(df_results.to_string(index=False))
    print("=" * 40)


if __name__ == "__main__":
    evaluate_models()
