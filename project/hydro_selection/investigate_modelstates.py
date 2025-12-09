from project.hydro_selection import load_config  # noqa: E402
import numpy as np
import matplotlib.pyplot as plt
import os

#------------------------------------------#
# Define model settings here.
CONFIG_PATH = r'conf/config_dhbvmoev1_ann.yaml'
#------------------------------------------#
# model training
config = load_config(CONFIG_PATH)

out_path = config['out_path']

# 读取 Qsimmu.npy
qsimmu_path = os.path.join(out_path, 'Qsimmu.npy')
Qsimmu = np.load(qsimmu_path)  # shape: (ts, 671, 16)

print(f"Qsimmu shape: {Qsimmu.shape}")

# 选择要绘制的流域索引 (0-670)
basin_idx = 100  # 可以修改为任意 0-670 之间的值

# 提取该流域的数据，shape: (ts, 16)
basin_data = Qsimmu[:, basin_idx, :]

# 绘制折线图
plt.figure(figsize=(14, 6))
for expert_idx in range(basin_data.shape[1]):
    plt.plot(basin_data[:, expert_idx], label=f'Expert {expert_idx}', alpha=0.7)

plt.xlabel('Time Step')
plt.ylabel('Qsim')
plt.title(f'Qsimmu for Basin {basin_idx} - All 16 Experts')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(out_path, f'qsimmu_basin_{basin_idx}.png'), dpi=150)
# plt.show()

# 绘制 Qsimmu 箱线图 (16个 Expert)
plt.figure(figsize=(12, 6))
boxplot_data_qsim = [basin_data[:, i] for i in range(16)]
plt.boxplot(boxplot_data_qsim, labels=[f'E{i}' for i in range(16)])
plt.xlabel('Expert')
plt.ylabel('Qsim')
plt.title(f'Qsimmu Distribution for Basin {basin_idx}')
plt.tight_layout()
plt.savefig(os.path.join(out_path, f'qsimmu_boxplot_basin_{basin_idx}.png'), dpi=150)
plt.show()

# 读取 moe_weights.npy 并绘制热力图
moe_weights_path = os.path.join(out_path, 'moe_weights.npy')
moe_weights = np.load(moe_weights_path)  # shape: (ts, 671, 16)

print(f"moe_weights shape: {moe_weights.shape}")

# 提取该流域的权重数据，shape: (ts, 16)
basin_weights = moe_weights[:, basin_idx, :]

# 绘制热力图 (16 x ts)
plt.figure(figsize=(14, 6))
# 转置为 (16, ts) 以便 y 轴是 Expert，x 轴是时间步
plt.imshow(basin_weights.T, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='Weight')
plt.xlabel('Time Step')
plt.ylabel('Expert')
plt.yticks(range(16), [f'Expert {i}' for i in range(16)])
plt.title(f'MoE Weights Heatmap for Basin {basin_idx}')
plt.tight_layout()
plt.savefig(os.path.join(out_path, f'moe_weights_basin_{basin_idx}.png'), dpi=150)
# plt.show()

# 绘制权重箱线图 (16个 Expert)
plt.figure(figsize=(12, 6))
# basin_weights shape: (ts, 16), 每列是一个 Expert 的权重分布
boxplot_data = [basin_weights[:, i] for i in range(16)]
plt.boxplot(boxplot_data, labels=[f'E{i}' for i in range(16)])
plt.xlabel('Expert')
plt.ylabel('Weight')
plt.title(f'MoE Weights Distribution for Basin {basin_idx}')
plt.tight_layout()
plt.savefig(os.path.join(out_path, f'moe_weights_boxplot_basin_{basin_idx}.png'), dpi=150)
plt.show()

# 验证: Qsimmu * moe_weights 在 Expert 维度求和，与 streamflow 对比
# 读取 streamflow.npy
streamflow_path = os.path.join(out_path, 'streamflow.npy')
streamflow = np.load(streamflow_path)  # shape: (ts, 671) 或 (ts, 671, 1)

print(f"streamflow shape: {streamflow.shape}")

# 计算加权求和: (ts, 671, 16) * (ts, 671, 16) -> sum over dim=2 -> (ts, 671)
weighted_sum = np.sum(Qsimmu * moe_weights, axis=2)
mean_sum = np.mean(Qsimmu, axis=2)

print(f"weighted_sum shape: {weighted_sum.shape}")

# 提取该流域的数据
basin_weighted = weighted_sum[:, basin_idx]
basin_streamflow = streamflow[:, basin_idx].squeeze()  # 去掉可能的多余维度

# 绘制对比折线图
plt.figure(figsize=(14, 6))
plt.plot(basin_weighted, label='Qsimmu * weights (sum)', alpha=0.8)
plt.plot(mean_sum[:, basin_idx], label='Qsimmu mean (over experts)', alpha=0.8)
# plt.plot(basin_streamflow, label='streamflow', alpha=0.8, linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Streamflow')
plt.title(f'Verification: Weighted Sum vs Streamflow for Basin {basin_idx}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_path, f'verification_basin_{basin_idx}.png'), dpi=150)
plt.show()

# 打印差异统计
diff = basin_weighted - basin_streamflow
print(f"Max absolute difference: {np.max(np.abs(diff)):.6e}")
print(f"Mean absolute difference: {np.mean(np.abs(diff)):.6e}")
print(f"Are they approximately equal? {np.allclose(basin_weighted, basin_streamflow)}")