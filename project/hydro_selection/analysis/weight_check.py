import numpy as np
import matplotlib.pyplot as plt
import os

# 数据文件路径（目录）
data_dir = (
    r"/workspace/my_deltamodel/project/hydro_selection/output/camels_671/train1980-1995/no_multi/MultiHeadNet_E50_R365_B100_n4_noLn_noWU_42/BlendHydroV2/NseBatchLoss/stat/test1995-2010_Ep50"
)

# 模型名称（对应4个模型）
model_names = ["HBV", "SHM", "EXPHYDRO", "HYMOD"]

# 读取每个模型的权重数据
print(f"从目录读取权重文件: {data_dir}")
weights_list = []
for model_name in model_names:
    weight_file = os.path.join(data_dir, f"{model_name}_weights.npy")
    print(f"  读取: {weight_file}")
    model_weights = np.load(weight_file)
    print(f"    形状: {model_weights.shape}")
    weights_list.append(model_weights)

# 将所有模型的权重堆叠成一个数组
# 假设每个模型的权重形状为 (Time, Grid)，堆叠后为 (Time, Grid, 4)
weights = np.stack(weights_list, axis=-1)

# 打印数据基本信息
print(f"\n合并后数据形状: {weights.shape}") # 应该是 (Time, Grid, 4) - 4个模型
print(f"数据类型: {weights.dtype}")
print(f"数据范围: [{weights.min()}, {weights.max()}]")
print(f"数据均值: {weights.mean()}")
print(f"数据标准差: {weights.std()}")

# 可视化（针对单个流域）
import argparse

parser = argparse.ArgumentParser(description="查看某个流域的模型权重随时间的变化")
parser.add_argument("--basin", type=int, default=11, help="流域索引（0-based）")
parser.add_argument("--save_dir", type=str, default='.', help="保存可视化图像的目录")
args = parser.parse_args()

basin_idx = args.basin
if basin_idx < 0 or basin_idx >= weights.shape[1]:
    raise ValueError(f"无效的流域索引: {basin_idx}，应在 [0, {weights.shape[1]-1}] 范围内")

os.makedirs(args.save_dir, exist_ok=True)

# 选择当前流域的数据：shape -> (time, 4)
basin_weights = weights[:, basin_idx, :]
print(f"\n流域 {basin_idx} 数据形状: {basin_weights.shape}")
print(f"各模型平均权重:")
for i, name in enumerate(model_names):
    print(f"  {name}: {basin_weights[:, i].mean():.4f} (std: {basin_weights[:, i].std():.4f})")

time_len = basin_weights.shape[0]
t = np.arange(time_len)

# 设置颜色（4种不同颜色对应4个模型）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红

# ============================================================
# 图1：堆叠面积图 - 展示4个模型权重的占比变化
# ============================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 6))
ax1.stackplot(t, basin_weights.T, labels=model_names, colors=colors, alpha=0.8)
ax1.set_title(f"Basin {basin_idx}: Model Weights Over Time (Stacked Area)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Time Step", fontsize=12)
ax1.set_ylabel("Weight Value", fontsize=12)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1])
fig1.tight_layout()
path1 = os.path.join(args.save_dir, f"basin_{basin_idx}_model_weights_stack.png")
fig1.savefig(path1, dpi=300, bbox_inches='tight')
print(f"\n已保存: {path1}")

# ============================================================
# 图2：折线图 - 展示4个模型权重的独立变化趋势
# ============================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
for i, (name, color) in enumerate(zip(model_names, colors)):
    ax2.plot(t, basin_weights[:, i], label=name, color=color, linewidth=2, alpha=0.8)
ax2.set_title(f"Basin {basin_idx}: Model Weights Over Time (Line Chart)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Time Step", fontsize=12)
ax2.set_ylabel("Weight Value", fontsize=12)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1])
fig2.tight_layout()
path2 = os.path.join(args.save_dir, f"basin_{basin_idx}_model_weights_lines.png")
fig2.savefig(path2, dpi=300, bbox_inches='tight')
print(f"已保存: {path2}")

# ============================================================
# 图3：箱线图 - 展示每个模型权重的分布统计
# ============================================================
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
bp = ax3.boxplot([basin_weights[:, i] for i in range(4)], 
                  labels=model_names,
                  patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

# 设置箱体颜色
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax3.set_title(f"Basin {basin_idx}: Model Weight Distribution", fontsize=14, fontweight='bold')
ax3.set_ylabel("Weight Value", fontsize=12)
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
ax3.set_ylim([0, 1])
fig3.tight_layout()
path3 = os.path.join(args.save_dir, f"basin_{basin_idx}_model_weights_boxplot.png")
fig3.savefig(path3, dpi=300, bbox_inches='tight')
print(f"已保存: {path3}")

# ============================================================
# 图4：热图 - 展示权重随时间的变化（更直观的视觉效果）
# ============================================================
fig4, ax4 = plt.subplots(1, 1, figsize=(14, 5))
im = ax4.imshow(basin_weights.T, aspect='auto', cmap='viridis', 
                interpolation='bilinear', vmin=0, vmax=1)
ax4.set_yticks(range(4))
ax4.set_yticklabels(model_names)
ax4.set_xlabel("Time Step", fontsize=12)
ax4.set_ylabel("Model", fontsize=12)
ax4.set_title(f"Basin {basin_idx}: Model Weights Heatmap", fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Weight Value', fontsize=11)
fig4.tight_layout()
path4 = os.path.join(args.save_dir, f"basin_{basin_idx}_model_weights_heatmap.png")
fig4.savefig(path4, dpi=300, bbox_inches='tight')
print(f"已保存: {path4}")

# ============================================================
# 图5：饼图 - 展示各模型的平均权重占比
# ============================================================
avg_weights = basin_weights.mean(axis=0)
fig5, ax5 = plt.subplots(1, 1, figsize=(8, 8))
wedges, texts, autotexts = ax5.pie(avg_weights, labels=model_names, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 12})
# 加粗百分比文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(13)
ax5.set_title(f"Basin {basin_idx}: Average Model Weight Distribution", 
              fontsize=14, fontweight='bold', pad=20)
fig5.tight_layout()
path5 = os.path.join(args.save_dir, f"basin_{basin_idx}_model_weights_pie.png")
fig5.savefig(path5, dpi=300, bbox_inches='tight')
print(f"已保存: {path5}")

# ============================================================
# 打印统计摘要
# ============================================================
print("\n" + "="*60)
print(f"统计摘要 - 流域 {basin_idx}")
print("="*60)
for i, name in enumerate(model_names):
    w = basin_weights[:, i]
    print(f"\n{name}:")
    print(f"  平均值: {w.mean():.4f}")
    print(f"  标准差: {w.std():.4f}")
    print(f"  最小值: {w.min():.4f}")
    print(f"  最大值: {w.max():.4f}")
    print(f"  中位数: {np.median(w):.4f}")

print("\n权重和检查（应接近1.0）:")
weight_sums = basin_weights.sum(axis=1)
print(f"  均值: {weight_sums.mean():.6f}")
print(f"  标准差: {weight_sums.std():.6f}")
print(f"  最小值: {weight_sums.min():.6f}")
print(f"  最大值: {weight_sums.max():.6f}")

plt.show()
