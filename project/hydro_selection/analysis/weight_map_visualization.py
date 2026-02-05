import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ���据文件路径（目录）
data_dir = (
    r"/workspace/my_deltamodel/project/hydro_selection/output/camels_671/train1980-1995/no_multi/"
    + "MultiHeadNet_E50_R365_B100_n4_noLn_noWU_42/BlendHydroV2/NseBatchLoss/stat/test1995-2010_Ep50"
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

# 将所有模型的权重堆���成一个数组
# 假设每个模型的权重形状为 (Time, Grid)，堆叠后为 (Time, Grid, 4)
weights = np.stack(weights_list, axis=-1)

print(f"\n合并后数据形状: {weights.shape}")  # 应该是 (Time, Grid, 4) - 4个模型
print(f"数据类型: {weights.dtype}")
print(f"数据范围: [{weights.min()}, {weights.max()}]")

# 根据时间步划分季节
# test1995-2010，共5110个时间步，假设是日数据
# 创建时间索引（从1995-01-01到2010-12-31）
import pandas as pd
start_date = pd.Timestamp('1995-01-01')
time_steps = weights.shape[0]
dates = pd.date_range(start=start_date, periods=time_steps, freq='D')

# 定义季节：春季(3-5月)、夏季(6-8月)、秋季(9-11月)、冬季(12-2月)
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:  # 12, 1, 2
        return 'Winter'

seasons = dates.map(lambda x: get_season(x.month))
season_names = ['Spring', 'Summer', 'Autumn', 'Winter']

print(f"\n时间范围: {dates[0]} 到 {dates[-1]}")
print(f"各季节时间步数:")
for season in season_names:
    count = (seasons == season).sum()
    print(f"  {season}: {count}")

# 计算每个季节每个流域每个模型的最大权重
# 结果: 字典 {season: (Grid, 4)}
seasonal_max_weights = {}
for season in season_names:
    season_mask = (seasons == season)
    season_weights = weights[season_mask, :, :]  # (Season_Time, Grid, 4)
    seasonal_max_weights[season] = season_weights.max(axis=0)  # (Grid, 4)
    print(f"\n{season} 最大权重形状: {seasonal_max_weights[season].shape}")

# 读取CAMELS流域位置数据（shapefile）
camels_shp_path = "/workspace/my_deltamodel/data/camels_loc/camels_671_loc.shp"
print(f"\n读取CAMELS流域位置数据: {camels_shp_path}")
gdf = gpd.read_file(camels_shp_path)
print(f"  流域数量: {len(gdf)}")
print(f"  列名: {gdf.columns.tolist()}")

# 读取gage_id顺序（确保与权重数据对应）
gage_id_file = "/workspace/my_deltamodel/data/gage_id.txt"
with open(gage_id_file, "r") as f:
    gage_ids = [line.strip() for line in f.readlines()]
print(f"\n读取gage_id文件: {gage_id_file}")
print(f"  gage_id数量: {len(gage_ids)}")

# 确保gage_id数量与权重数据匹配
if len(gage_ids) != seasonal_max_weights['Spring'].shape[0]:
    raise ValueError(
        f"gage_id数量 ({len(gage_ids)}) 与权重数据流域数量 ({seasonal_max_weights['Spring'].shape[0]}) 不匹配"
    )

# 将权重数据添加到GeoDataFrame
# 首先需要确保gdf中的gage_id与我们的gage_ids顺序一致
# 假设shapefile中有一个gage_id列（可能名称不同，需要检查）
# 常见的列名: 'GAGE_ID', 'gage_id', 'STAID', 'gauge_id'
gage_id_col = None
for col in ['GAGE_ID', 'gage_id', 'STAID', 'gauge_id', 'hru_id']:
    if col in gdf.columns:
        gage_id_col = col
        break

if gage_id_col is None:
    print(f"警告: 未找到gage_id列，使用前几列: {gdf.columns[:5].tolist()}")
    print(f"GeoDataFrame前几行:\n{gdf.head()}")
    # 尝试使用第一个非几何列
    for col in gdf.columns:
        if col != 'geometry':
            gage_id_col = col
            break

print(f"\n使用gage_id列: {gage_id_col}")

# 将gage_id转换为字符串格式（8位，前面补0）
gdf[gage_id_col] = gdf[gage_id_col].astype(str).str.zfill(8)

# 创建一个字典，将gage_id映射到权重索引
gage_id_to_idx = {gid: idx for idx, gid in enumerate(gage_ids)}

# 为每个季节的每个模型添加权重列
for season in season_names:
    for i, model_name in enumerate(model_names):
        col_name = f'{season}_{model_name}_weight'
        gdf[col_name] = gdf[gage_id_col].map(
            lambda gid: seasonal_max_weights[season][gage_id_to_idx[gid], i] if gid in gage_id_to_idx else np.nan
        )

# 检查是否有缺失值
for season in season_names:
    for model_name in model_names:
        col_name = f'{season}_{model_name}_weight'
        n_missing = gdf[col_name].isna().sum()
        if n_missing > 0:
            print(f"警告: {season} {model_name} 有 {n_missing} 个流域缺失权重数据")

# ============================================================
# 创建4个季节的地图可视化（每个季节一个图，包含4个模型的子图）
# ============================================================

# 创建输出目录
output_dir = "/workspace/my_deltamodel/project/hydro_selection/analysis"
os.makedirs(output_dir, exist_ok=True)

# 获取美国大陆的边界（用于背景）
usa_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

for season in season_names:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 为每个模型使用不同的颜色方案
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds']

    for i, (model_name, cmap) in enumerate(zip(model_names, colormaps)):
        ax = axes[i]

        # 获取当前季节当前模型的权重
        weight_col = f'{season}_{model_name}_weight'

        # 提取经纬度和权重数据
        lons = gdf['lon'].values
        lats = gdf['lat'].values
        weights_values = gdf[weight_col].values

        # 使用scatter绘制散点图
        scatter = ax.scatter(
            lons,
            lats,
            c=weights_values,
            cmap=cmap,
            s=30,  # 固定散点大小
            edgecolors='black',
            linewidths=0.3,
            vmin=0,
            vmax=1,
            alpha=0.8
        )

        # 设置标题
        ax.set_title(f'{model_name} - Maximum Weight', fontsize=14, fontweight='bold')

        # 移除坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])

        # 添加colorbar，调整长度使其与图高度相当
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04, aspect=15)
        cbar.set_label('Maximum Weight', fontsize=11)

        # 设置坐标轴范围（保持一致）
        ax.set_xlim(usa_bounds[0], usa_bounds[2])
        ax.set_ylim(usa_bounds[1], usa_bounds[3])

        # 添加统计信息
        max_weight_mean = gdf[weight_col].mean()
        max_weight_std = gdf[weight_col].std()
        ax.text(
            0.02, 0.98,
            f'Median: {max_weight_mean:.3f}\nStd: {max_weight_std:.3f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # 添加总标题
    fig.suptitle(f'{season} - Model Weights Distribution', fontsize=16, fontweight='bold', y=0.995)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存图像
    output_path = os.path.join(output_dir, f"model_weights_map_{season}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n已保存{season}地图可视化: {output_path}")

    plt.close(fig)

# ============================================================
# 打印统计摘要
# ============================================================
print("\n" + "=" * 60)
print("各季节各模型最大权重统计摘要")
print("=" * 60)
for season in season_names:
    print(f"\n{season}:")
    print("-" * 60)
    for model_name in model_names:
        weight_col = f'{season}_{model_name}_weight'
        weights_data = gdf[weight_col].dropna()
        print(f"  {model_name}:")
        print(f"    中位数: {weights_data.mean():.4f}, 标准差: {weights_data.std():.4f}")
        print(f"    最小值: {weights_data.min():.4f}, 最大值: {weights_data.max():.4f}")

print("\n所有季节地图可视化已完成！")
