"""
直接从 .pt 文件读取 Calibrate 模型参数并绘制分布

功能：
1. 直接加载 .pt 文件（无需转换为 npz）
2. 应用 sigmoid 变换将参数从 logit 空间转换到 [0, 1]
3. 将参数从 [0, 1] 映射回实际物理范围
4. 根据 metrics.json 筛选精度前 N% 的流域
5. 绘制筛选后流域的参数分布直方图
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# HBV96 参数边界（从 hbv96.py 复制）
HBV96_PARAMS_BOUNDS = {
    "tt": [-3.0, 5.0],  # TT, threshold temperature for snowfall [oC]
    "tti": [0.0, 17.0],  # TTI, interval length of rain-snow spectrum [oC]
    "ttm": [-3.0, 3.0],  # TTM, threshold temperature for snowmelt [oC]
    "cfr": [0.0, 1.0],  # CFR, coefficient of refreezing of melted snow [-]
    "cfmax": [
        0.0,
        20.0,
    ],  # CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
    "whc": [0.0, 1.0],  # WHC, maximum water holding content of snow pack [-]
    "cflux": [0.0, 4.0],  # CFLUX, maximum rate of capillary rise [mm/d]
    "fc": [1.0, 2000.0],  # FC, maximum soil moisture storage [mm]
    "lp": [0.05, 0.95],  # LP, wilting point as fraction of FC [-]
    "beta": [
        0.0,
        10.0,
    ],  # BETA, non-linearity coefficient of upper zone recharge [-]
    "k0": [0.0, 1.0],  # K0, runoff coefficient from upper zone [d-1]
    "alpha": [
        0.0,
        4.0,
    ],  # ALPHA, non-linearity coefficient of runoff from upper zone [-]
    "perc": [
        0.0,
        20.0,
    ],  # PERC, maximum rate of percolation to lower zone [mm/d]
    "k1": [0.0, 1.0],  # K1, runoff coefficient from lower zone [d-1]
    "maxbas": [1.0, 120.0],  # MAXBAS, flow routing delay [d]
}

# 参数名称列表
PARAM_NAMES = [
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


def load_metrics_json(metrics_path: str, metric_key: str = "nse"):
    """
    从 metrics.json 文件加载性能指标

    Parameters
    ----------
    metrics_path : str
        metrics.json 文件路径
    metric_key : str
        要读取的指标键名，默认 "nse"

    Returns
    -------
    metrics : np.ndarray
        性能指标数组，形状 [num_basins, num_members]
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    print(f"Loading metrics from: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        content = f.read().lstrip("\ufeff").strip()

    data = json.loads(content)
    if isinstance(data, str):
        data = json.loads(data)

    if metric_key not in data:
        raise ValueError(
            f"Metric key '{metric_key}' not found in metrics.json. Available keys: {list(data.keys())}"
        )

    metrics_list = data[metric_key]
    metrics_array = np.array(metrics_list, dtype=float)

    print(f"  Loaded {len(metrics_list)} metric values")
    print(
        f"  Metric range: [{np.nanmin(metrics_array):.4f}, {np.nanmax(metrics_array):.4f}]"
    )

    return metrics_array


def filter_top_members_per_basin(metrics: np.ndarray, top_percent: float = 0.3):
    """
    对每个流域，筛选出前 N% 表现最好的 ensemble 成员索引

    Parameters
    ----------
    metrics : np.ndarray
        性能指标数组，形状 [num_basins, num_members] 或 [num_basins * num_members]
    top_percent : float
        保留的顶部百分比，默认 0.3 (30%)

    Returns
    -------
    top_member_mask : np.ndarray
        布尔掩码数组，形状 [num_basins, num_members]，True 表示该成员被选中
    """
    # 如果是 1D 数组，需要 reshape
    # JSON格式：[basin0_member0, basin0_member1, ..., basin0_member127, basin1_member0, ...]
    if metrics.ndim == 1:
        num_members = 128  # 根据配置文件
        num_basins = len(metrics) // num_members
        # 直接reshape为 [num_basins, num_members]
        metrics_2d = metrics.reshape(num_basins, num_members)
    else:
        metrics_2d = metrics
        num_basins, num_members = metrics_2d.shape

    print(f"\nFiltering top {top_percent*100:.0f}% members for each basin:")
    print(f"  Metrics shape: {metrics_2d.shape} (basins × members)")
    print(f"  First basin metrics sample: {metrics_2d[0, :5]}")  # 打印第一个basin的前5个成员

    # 创建掩码数组
    top_member_mask = np.zeros_like(metrics_2d, dtype=bool)

    # 对每个流域单独筛选
    num_to_select = max(1, int(num_members * top_percent))

    for basin_idx in range(num_basins):
        basin_metrics = metrics_2d[basin_idx, :]
        # 获取前 N% 的索引
        top_indices = np.argsort(basin_metrics)[-num_to_select:]
        top_member_mask[basin_idx, top_indices] = True

    total_selected = np.sum(top_member_mask)
    print(f"  Selecting top {num_to_select} members per basin")
    print(f"  Total selected samples: {total_selected} out of {num_basins * num_members}")

    return top_member_mask


def load_calibrate_params(pt_file_path: str):
    """
    从 .pt 文件加载 Calibrate 模型参数

    Parameters
    ----------
    pt_file_path : str
        .pt 文件路径

    Returns
    -------
    params_scaled : torch.Tensor
        缩放后的参数，形状 [num_basins, ny, num_start]
    params_normalized : torch.Tensor
        归一化参数 [0, 1]，形状 [num_basins, ny, num_start]
    """
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"File not found: {pt_file_path}")

    print(f"Loading: {pt_file_path}")

    # 加载 .pt 文件
    checkpoint = torch.load(pt_file_path, map_location="cpu")

    # 提取参数
    if isinstance(checkpoint, dict) and "nn_model.params" in checkpoint:
        params_raw = checkpoint[
            "nn_model.params"
        ]  # Shape: [num_basins, ny, num_start]
    else:
        raise ValueError(
            f"Unexpected checkpoint structure. Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}"
        )

    print(f"  Raw params shape: {params_raw.shape}")
    print(
        f"  Raw params range: [{params_raw.min().item():.6f}, {params_raw.max().item():.6f}]"
    )

    # 应用 sigmoid 变换 (与 Calibrate.forward 中的逻辑一致)
    params_normalized = torch.sigmoid(params_raw)

    print(
        f"  Normalized params range: [{params_normalized.min().item():.6f}, {params_normalized.max().item():.6f}]"
    )

    return params_raw, params_normalized


def descale_params(
    params_normalized: torch.Tensor, param_names: list, bounds: dict
):
    """
    将归一化参数 [0, 1] 映射回实际物理范围

    Parameters
    ----------
    params_normalized : torch.Tensor
        归一化参数，形状 [num_basins, ny, num_start]
    param_names : list
        参数名称列表
    bounds : dict
        参数边界字典

    Returns
    -------
    params_scaled : torch.Tensor
        缩放后的参数，形状 [num_basins, ny, num_start]
    """
    num_basins, ny, num_start = params_normalized.shape
    params_scaled = torch.zeros_like(params_normalized)

    for i, name in enumerate(param_names):
        if name in bounds:
            min_val, max_val = bounds[name]
            # 线性映射: [0, 1] -> [min_val, max_val]
            params_scaled[:, i, :] = (
                params_normalized[:, i, :] * (max_val - min_val) + min_val
            )
        else:
            print(
                f"Warning: Parameter {name} not found in bounds, using normalized values"
            )
            params_scaled[:, i, :] = params_normalized[:, i, :]

    return params_scaled


def plot_parameter_distributions(
    params_scaled: torch.Tensor,
    param_names: list,
    bounds: dict,
    member_mask: np.ndarray = None,
    basin_idx: int = None,
    save_path: str = None,
):
    """
    绘制参数分布图（支持筛选 ensemble 成员）

    Parameters
    ----------
    params_scaled : torch.Tensor
        缩放后的参数，形状 [num_basins, ny, num_members]
    param_names : list
        参数名称列表
    bounds : dict
        参数边界字典
    member_mask : np.ndarray
        成员掩码，形状 [num_basins, num_members]，True 表示该成员被选中
        如果为 None，则使用所有成员
    basin_idx : int
        要绘制的流域索引，如果为 None 则绘制所有流域
    save_path : str
        保存路径

    Returns
    -------
    None
    """
    params_np = params_scaled.numpy()  # [num_basins, ny, num_members]

    # 如果指定了流域索引，只处理该流域
    if basin_idx is not None:
        params_np = params_np[basin_idx:basin_idx+1, :, :]
        if member_mask is not None:
            member_mask = member_mask[basin_idx:basin_idx+1, :]
        title_suffix = f"Basin {basin_idx}"
    else:
        title_suffix = "All Basins"

    # 应用成员掩码筛选
    if member_mask is not None:
        # 提取被选中的参数样本
        selected_samples = []
        for b_idx in range(params_np.shape[0]):
            basin_params = params_np[b_idx, :, :]  # [ny, num_members]
            basin_mask = member_mask[b_idx, :]  # [num_members]
            selected = basin_params[:, basin_mask]  # [ny, num_selected]
            selected_samples.append(selected)

        # 合并所有流域的选中样本
        all_samples = np.concatenate(selected_samples, axis=1)  # [ny, total_selected]
        num_selected = all_samples.shape[1]
    else:
        # 使用所有样本
        all_samples = params_np.reshape(params_np.shape[0], params_np.shape[1], -1)
        all_samples = all_samples.transpose(1, 0, 2).reshape(params_np.shape[1], -1)
        num_selected = all_samples.shape[1]

    print(f"\nPlotting parameter distributions for {title_suffix}")
    print(f"  Total samples per parameter: {num_selected}")

    # 创建子图
    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        param_values = all_samples[i, :]

        # 绘制直方图
        ax.hist(
            param_values,
            bins=30,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )

        # 添加参数边界线
        if name in bounds:
            min_val, max_val = bounds[name]
            ax.axvline(
                min_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Min={min_val:.2f}",
            )
            ax.axvline(
                max_val,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"Max={max_val:.2f}",
            )

        # 设置标题和标签
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Parameter Value", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Parameter Distributions - {title_suffix} (Top {num_selected} samples)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()
    plt.close()


def main():
    """主函数"""
    # ==========================================
    # 配置路径
    # ==========================================
    # .pt 文件路径
    pt_file = "/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B100_n128_noLn_noWU_42/hbv96/KgeLoss/stat/dUnifyV1_Ep100.pt"

    # metrics.json 文件路径
    metrics_file = "/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B100_n128_noLn_noWU_42/hbv96/KgeLoss/stat/train1989-1998_Ep100/metrics.json"

    # 输出目录
    figures_dir = "/workspace/my_deltamodel/project/deal_hydro/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # 筛选参数
    top_percent = 0.29  # 保留每个流域前 30% 的 ensemble 成员
    metric_key = "kge"  # 使用 NSE 作为筛选指标
    basin_idx = 0  # 指定要绘制的流域索引（0-558），设置为 None 绘制所有流域

    print("=" * 80)
    print("HBV96 Parameter Distribution Analysis (Filtered by Performance)")
    print("=" * 80)

    # ==========================================
    # 1. 加载性能指标并筛选 ensemble 成员
    # ==========================================
    print("\n[Step 1] Loading performance metrics...")
    metrics = load_metrics_json(metrics_file, metric_key=metric_key)

    print("\n[Step 2] Filtering top-performing ensemble members for each basin...")
    member_mask = filter_top_members_per_basin(metrics, top_percent=top_percent)

    # ==========================================
    # 2. 加载参数
    # ==========================================
    print("\n[Step 3] Loading calibrated parameters...")
    params_raw, params_normalized = load_calibrate_params(pt_file)

    # ==========================================
    # 3. 反归一化参数
    # ==========================================
    print("\n[Step 4] Descaling parameters to physical range...")
    params_scaled = descale_params(
        params_normalized, PARAM_NAMES, HBV96_PARAMS_BOUNDS
    )

    print(f"  Scaled params shape: {params_scaled.shape}")
    print(
        f"  Scaled params range: [{params_scaled.min().item():.6f}, {params_scaled.max().item():.6f}]"
    )

    # ==========================================
    # 4. 绘制筛选后的参数分布
    # ==========================================
    print(
        f"\n[Step 5] Plotting parameter distributions for top {top_percent * 100:.0f}% members..."
    )

    if basin_idx is not None:
        save_path = os.path.join(
            figures_dir,
            f"calibrate_params_basin{basin_idx}_top{int(top_percent * 100)}pct_{metric_key}.png",
        )
    else:
        save_path = os.path.join(
            figures_dir,
            f"calibrate_params_allbasins_top{int(top_percent * 100)}pct_{metric_key}.png",
        )

    plot_parameter_distributions(
        params_scaled,
        PARAM_NAMES,
        HBV96_PARAMS_BOUNDS,
        member_mask=member_mask,
        basin_idx=basin_idx,
        save_path=save_path,
    )

    # ==========================================
    # 5. 打印统计信息
    # ==========================================
    print("\n" + "=" * 80)
    print(f"Parameter Statistics (Top {top_percent*100:.0f}% members per basin)")
    print("=" * 80)

    # 提取筛选后的参数
    params_np = params_scaled.numpy()

    if basin_idx is not None:
        # 单个流域
        basin_params = params_np[basin_idx, :, :]  # [ny, num_members]
        basin_mask = member_mask[basin_idx, :]
        selected_params = basin_params[:, basin_mask]  # [ny, num_selected]
        print(f"Basin {basin_idx}:")
    else:
        # 所有流域
        selected_samples = []
        for b_idx in range(params_np.shape[0]):
            basin_params = params_np[b_idx, :, :]
            basin_mask = member_mask[b_idx, :]
            selected = basin_params[:, basin_mask]
            selected_samples.append(selected)
        selected_params = np.concatenate(selected_samples, axis=1)
        print(f"All {params_np.shape[0]} basins:")

    for i, name in enumerate(PARAM_NAMES):
        param_values = selected_params[i, :]
        bounds = HBV96_PARAMS_BOUNDS.get(name, [None, None])
        print(
            f"{name:10s}: Mean={param_values.mean():.4f}, "
            f"Std={param_values.std():.4f}, "
            f"Min={param_values.min():.4f}, "
            f"Max={param_values.max():.4f}, "
            f"Bounds={bounds}"
        )

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
