from project.hydro_selection import load_config  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
import os

#------------------------------------------#
# Define two config paths to compare
CONFIG_PATH_1 = r'conf/config_dhbvmoev1_ann.yaml'
CONFIG_PATH_2 = r'conf/config_dhbv_ann.yaml'  # 修改为第二个配置文件路径
#------------------------------------------#

# 加载两个配置
config_1 = load_config(CONFIG_PATH_1)
config_2 = load_config(CONFIG_PATH_2)

out_path_1 = config_1['out_path']
out_path_2 = config_2['out_path']

# 读取两个 streamflow.npy
streamflow_1 = np.load(os.path.join(out_path_1, 'streamflow.npy'))
streamflow_2 = np.load(os.path.join(out_path_2, 'streamflow.npy'))

print(f"Streamflow 1 shape: {streamflow_1.shape}")
print(f"Streamflow 2 shape: {streamflow_2.shape}")

# 选择要对比的流域索引 (0-670)
basin_idx = 0  # 可以修改为任意 0-670 之间的值

# 提取该流域的数据
basin_flow_1 = streamflow_1[:, basin_idx].squeeze()
basin_flow_2 = streamflow_2[:, basin_idx].squeeze()

# 绘制对比折线图
plt.figure(figsize=(14, 6))
plt.plot(basin_flow_1, label=f'Config 1: {os.path.basename(CONFIG_PATH_1)}', alpha=0.8)
plt.plot(basin_flow_2, label=f'Config 2: {os.path.basename(CONFIG_PATH_2)}', alpha=0.8, linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Streamflow')
plt.title(f'Streamflow Comparison for Basin {basin_idx}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_path_1, f'streamflow_comparison_basin_{basin_idx}.png'), dpi=150)
plt.show()

# 绘制差异图
plt.figure(figsize=(14, 4))
diff = basin_flow_1 - basin_flow_2
plt.plot(diff, color='red', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Time Step')
plt.ylabel('Difference (Config1 - Config2)')
plt.title(f'Streamflow Difference for Basin {basin_idx}')
plt.tight_layout()
plt.savefig(os.path.join(out_path_1, f'streamflow_diff_basin_{basin_idx}.png'), dpi=150)
plt.show()

# 打印统计信息
print(f"\n=== Statistics for Basin {basin_idx} ===")
print(f"Config 1 mean: {np.mean(basin_flow_1):.4f}, std: {np.std(basin_flow_1):.4f}")
print(f"Config 2 mean: {np.mean(basin_flow_2):.4f}, std: {np.std(basin_flow_2):.4f}")
print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")
print(f"Mean absolute difference: {np.mean(np.abs(diff)):.6f}")
print(f"Correlation: {np.corrcoef(basin_flow_1, basin_flow_2)[0, 1]:.6f}")

# 绘制散点图
plt.figure(figsize=(8, 8))
plt.scatter(basin_flow_1, basin_flow_2, alpha=0.5, s=10)
max_val = max(np.max(basin_flow_1), np.max(basin_flow_2))
plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
plt.xlabel(f'Streamflow - {os.path.basename(CONFIG_PATH_1)}')
plt.ylabel(f'Streamflow - {os.path.basename(CONFIG_PATH_2)}')
plt.title(f'Scatter Plot for Basin {basin_idx}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_path_1, f'streamflow_scatter_basin_{basin_idx}.png'), dpi=150)
plt.show()
