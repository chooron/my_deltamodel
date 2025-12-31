import numpy as np
import xarray as xr
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

model_keyword = "tcm"

save_dir = os.path.join(project_root, "project/diff_compare/tests/sim")
loaded = np.load(os.path.join(save_dir, f"{model_keyword}_sim_results.npz"))
final_results = loaded["data"]  # 必须使用保存时指定的键名 'data'
matlab_results_dir = r"G:\Dataset\MARRMoTResult"
matlab_files = sorted(
    [
        f
        for f in os.listdir(matlab_results_dir)
        if model_keyword in f and f.endswith(".nc")
    ]
)
print(f"Found MATLAB files: {matlab_files}")

target_nc_file = os.path.join(matlab_results_dir, matlab_files[0])
print(f"Reading benchmark: {target_nc_file}")

ds = xr.open_dataset(target_nc_file)

# MATLAB shape: (3, 7670, 559) -> (Objs, Time, Basins)
sim_q_matlab_raw = ds["Sim_q"].values

# 转置为 (Basins, Time, Objs) 与 python 对齐
sim_q_matlab_aligned = np.transpose(sim_q_matlab_raw, (2, 1, 0))

plt.plot(sim_q_matlab_aligned[10, :5000, 1])
plt.plot(final_results[10, :5000, 1])
plt.show()
