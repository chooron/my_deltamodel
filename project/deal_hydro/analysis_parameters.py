import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore
from dmg import ModelHandler  # noqa: E402
from dmg.core.utils import (  # noqa: E402
    import_data_loader,
    set_randomseed,
)
from project.deal_hydro import load_config  # noqa: E402

# ------------------------------------------#
# Define model settings here.
CONFIG_PATH = r"conf/config_unify_ann.yaml"
# ------------------------------------------#
# model training
config = load_config(CONFIG_PATH)
config["mode"] = "train"
set_randomseed(config["random_seed"])
model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config["data_loader"])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
param_pred_model = model.model_dict["UnifyV2"].nn_model
model_input = data_loader.train_dataset
norm_attr = model_input["c_nn_norm"] # torch.Size([559, 35])
pred_params = param_pred_model(model_input)[1] # torch.Size 559 16 15
print(pred_params.shape)
attr_names = config["delta_model"]["nn_model"].get("attributes", [])
param_names = [
    "tt",
    "tti",
    "ttm",
    "cfr",
    "cfmax",
    "whc",
    "cflux",
    "fc",
    "lp",
    "beta",
    "k0",
    "alpha",
    "perc",
    "k1",
    "maxbas",
]

# HBV96 参数边界（用于反缩放）
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


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _descale_params(params_normalized, param_names, bounds):
    """
    将归一化参数 [0, 1] 映射回实际物理范围

    Parameters
    ----------
    params_normalized : np.ndarray or torch.Tensor
        归一化参数，形状 [..., n_params] 或 [..., n_params, n_samples]
    param_names : list
        参数名称列表
    bounds : dict
        参数边界字典

    Returns
    -------
    params_scaled : np.ndarray
        缩放后的参数，与输入形状相同
    """
    is_tensor = isinstance(params_normalized, torch.Tensor)
    if is_tensor:
        params_normalized = _to_numpy(params_normalized)

    params_scaled = np.zeros_like(params_normalized)

    # 处理不同的输入形状
    if params_normalized.ndim == 2:
        # 形状: [n_basins, n_params]
        for i, name in enumerate(param_names):
            if name in bounds:
                min_val, max_val = bounds[name]
                params_scaled[:, i] = params_normalized[:, i] * (max_val - min_val) + min_val
            else:
                params_scaled[:, i] = params_normalized[:, i]
    elif params_normalized.ndim == 3:
        # 形状: [n_basins, n_params, n_samples]
        for i, name in enumerate(param_names):
            if name in bounds:
                min_val, max_val = bounds[name]
                params_scaled[:, i, :] = params_normalized[:, i, :] * (max_val - min_val) + min_val
            else:
                params_scaled[:, i, :] = params_normalized[:, i, :]
    else:
        raise ValueError(f"Unsupported parameter shape: {params_normalized.shape}")

    return params_scaled


# 原始归一化参数 [0, 1]
params_normalized_np = _to_numpy(pred_params)  # shape (559, 16, 15) or (559, 15)

# 反缩放到物理范围
params_np = _descale_params(params_normalized_np, param_names, HBV96_PARAMS_BOUNDS)
print(f"Parameters descaled to physical range: {params_np.shape}")
print(f"  Range: [{params_np.min():.4f}, {params_np.max():.4f}]")

attrs_np = _to_numpy(norm_attr)  # shape (559, 35)

num_params = params_np.shape[1]
num_attrs = attrs_np.shape[1]
attr_labels = attr_names if len(attr_names) == num_attrs else [f"attr_{i + 1}" for i in range(num_attrs)]
param_labels = param_names if len(param_names) == num_params else [f"param_{i + 1}" for i in range(num_params)]
corr_matrix = np.empty((num_params, num_attrs))

for p_idx in range(num_params):
    for a_idx in range(num_attrs):
        corr_matrix[p_idx, a_idx] = np.corrcoef(
            params_np[:, p_idx], attrs_np[:, a_idx]
        )[0, 1]

corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

fig, ax = plt.subplots(figsize=(12, 8))
heatmap = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
ax.set_xlabel("Attributes")
ax.set_ylabel("Parameters")
ax.set_xticks(range(num_attrs))
ax.set_xticklabels(attr_labels, rotation=90, fontsize=7)
ax.set_yticks(range(num_params))
ax.set_yticklabels(param_labels, fontsize=8)
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label("Pearson r")
fig.tight_layout()

fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "param_attr_corr_heatmap.png")
fig.savefig(fig_path, dpi=300)

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import gaussian_kde

# ==========================================
# 1. 核心算法：几何中心对象选择 (Geometric Medoid)
# ==========================================
print("正在执行几何中心对象选择 (Geometric Medoid Selection)...")

def get_geometric_medoid(params_ensemble):
    """
    从集成预测中选择几何中心对象。
    Args:
        params_ensemble: Tensor [Batch, Heads, Params]
    Returns:
        medoid_params: Tensor [Batch, Params]
    """
    # 1. Z-Score 标准化 (在 Ensemble 维度，消除参数量级差异)
    mean = params_ensemble.mean(dim=1, keepdim=True)
    std = params_ensemble.std(dim=1, keepdim=True) + 1e-6
    z_scores = (params_ensemble - mean) / std

    # 2. 计算距离矩阵 (Batch, Heads, Heads)
    # 利用广播机制计算两两之间的欧氏距离
    diff = z_scores.unsqueeze(2) - z_scores.unsqueeze(1)
    dist_matrix = torch.norm(diff, dim=-1)
    
    # 3. 计算距离之和，找最小的 Index
    sum_dist = dist_matrix.sum(dim=-1) # [Batch, Heads]
    best_idx = torch.argmin(sum_dist, dim=1) # [Batch]
    
    # 4. 提取对应的参数
    batch_size, num_heads, num_params = params_ensemble.shape
    idx_expanded = best_idx.view(-1, 1, 1).expand(-1, 1, num_params)
    medoid_params = torch.gather(params_ensemble, 1, idx_expanded).squeeze(1)
    
    return medoid_params

# 使用 Medoid 替代 Random Selection
with torch.no_grad():
    # pred_params: [559, 16, 15]
    best_params_tensor = get_geometric_medoid(pred_params)

# 将选择的最佳参数也反缩放到物理范围
best_params_np = _to_numpy(best_params_tensor)  # [559, 15] (归一化的)
params_np = _descale_params(best_params_np, param_names, HBV96_PARAMS_BOUNDS)  # 反缩放到物理范围
print(f"Best parameters (medoid) descaled to physical range: {params_np.shape}")
print(f"  Range: [{params_np.min():.4f}, {params_np.max():.4f}]")

attrs_np = _to_numpy(norm_attr)  # [559, 35]

# ==========================================
# 2. 深度分析：互信息 (Mutual Information) vs Pearson
# ==========================================
print("正在计算互信息 (捕捉非线性关系)... 这可能需要一分钟...")

num_params = params_np.shape[1]
num_attrs = attrs_np.shape[1]

# 初始化矩阵
mi_matrix = np.zeros((num_params, num_attrs))
pearson_matrix = np.zeros((num_params, num_attrs))

# 计算两种指标
for p_idx in range(num_params):
    # 计算 Pearson (线性)
    for a_idx in range(num_attrs):
        pearson_matrix[p_idx, a_idx] = np.corrcoef(params_np[:, p_idx], attrs_np[:, a_idx])[0, 1]
    
    # 计算 Mutual Information (非线性)
    # random_state 保证结果可复现
    mi_scores = mutual_info_regression(attrs_np, params_np[:, p_idx], random_state=42)
    mi_matrix[p_idx, :] = mi_scores

# 打印每个参数与流域属性的 Pearson 相关系数（两位小数）
print("\n参数-属性 Pearson 相关系数：")
for p_idx, p_name in enumerate(param_labels):
    row_vals = [f"{attr_labels[a_idx]}={pearson_matrix[p_idx, a_idx]:.2f}" for a_idx in range(num_attrs)]
    print(f"{p_name}: " + ", ".join(row_vals))

# 绘图：左右对比 (左边 Pearson, 右边 MI)
fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1]})

# Plot 1: Pearson Correlation (你之前的图)
sns.heatmap(pearson_matrix, ax=axes[0], cmap="coolwarm", center=0, vmin=-0.8, vmax=0.8,
            xticklabels=attr_labels, yticklabels=param_labels, cbar_kws={'label': 'Pearson r'})
axes[0].set_title("(A) Linear Correlation (Pearson)\nShows Direction (+/-)", fontsize=14)

# Plot 2: Mutual Information (新图)
# MI 值非负，越高代表关系越紧密（无论是否线性）
sns.heatmap(mi_matrix, ax=axes[1], cmap="viridis", vmin=0, 
            xticklabels=attr_labels, yticklabels=False, cbar_kws={'label': 'Mutual Information (Nats)'})
axes[1].set_title("(B) Non-Linear Dependency (Mutual Info)\nShows Strength of Mapping", fontsize=14)

plt.tight_layout()
fig_path_mi = os.path.join(fig_dir, "correlation_vs_mutual_info.png")
plt.savefig(fig_path_mi, dpi=300)
print(f"图表已保存: {fig_path_mi}")

# ==========================================
# 3. 物理机制可视化：关键参数-属性散点图
# ==========================================
print("正在绘制物理机制散点图...")

# 定义你想深入挖掘的物理对子 (Param_Name, Attr_Name)
# 基于你之前的热图结果，这些是有物理意义的亮点
pairs_to_plot = [
    ("cfmax", "lai_max"),      # 融雪因子 vs 叶面积指数 (植被遮阴效应)
    ("k0", "slope_mean"),  # 汇流时间 vs 坡度 (地形排水快慢)
    ("fc", "clay_frac")        # 蓄水容量 vs 粘土含量 (土壤持水性)
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (p_name, a_name) in enumerate(pairs_to_plot):
    # 获取索引
    try:
        p_idx = param_names.index(p_name)
        # 处理属性名可能不一致的情况，这里做个简单的模糊匹配或直接查找
        # 假设 attr_names 是完整的，如果找不到会报错，实际使用需确保名字对齐
        if a_name in attr_names:
            a_idx = attr_names.index(a_name)
        else:
            # 如果名字匹配不上，尝试找个近似的或者跳过
            print(f"Warning: Attribute {a_name} not found. Skipping.")
            continue
            
        x_data = attrs_np[:, a_idx]
        y_data = params_np[:, p_idx]
        
        # 计算点密度，为了画图好看
        xy = np.vstack([x_data, y_data])
        z = gaussian_kde(xy)(xy)
        
        # 散点图
        axes[i].scatter(x_data, y_data, c=z, s=20, cmap='Spectral_r', alpha=0.6)
        
        # 拟合一条非线性趋势线 (Lowess)
        sns.regplot(x=x_data, y=y_data, ax=axes[i], scatter=False, lowess=True, 
                    line_kws={'color': 'black', 'linewidth': 2, 'linestyle': '--'})
        
        axes[i].set_xlabel(f"Attribute: {a_name} (Normalized)", fontsize=12)
        axes[i].set_ylabel(f"Parameter: {p_name}", fontsize=12)
        axes[i].set_title(f"{p_name} vs. {a_name}\nChecking Physical Consistency", fontsize=13)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
    except ValueError as e:
        print(f"Error plotting {p_name} vs {a_name}: {e}")

plt.tight_layout()
fig_path_scatter = os.path.join(fig_dir, "physical_mechanism_scatter.png")
plt.savefig(fig_path_scatter, dpi=300)
print(f"图表已保存: {fig_path_scatter}")

print("分析完成！请查看 figures 文件夹。")