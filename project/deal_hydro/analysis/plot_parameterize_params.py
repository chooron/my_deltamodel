"""
绘制 MC Dropout 参数采样分布

功能：
1. 加载 MC Dropout 参数采样结果
2. 选择指定流域，绘制每个参数的分布直方图
3. 显示均值、标准差、P10、P90 等统计量
4. 保存图表到 figures 文件夹
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# HBV96 参数边界
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


def descale_params(params_normalized: np.ndarray, param_names: list, bounds: dict):
    """
    将归一化参数 [0, 1] 映射回实际物理范围

    Parameters
    ----------
    params_normalized : np.ndarray
        归一化参数
    param_names : list
        参数名称列表
    bounds : dict
        参数边界字典

    Returns
    -------
    params_scaled : np.ndarray
        缩放后的参数
    """
    params_scaled = np.zeros_like(params_normalized)

    # 处理不同的输入形状
    if params_normalized.ndim == 3:
        # 形状: (n_samples, n_basins, n_params)
        for i, name in enumerate(param_names):
            if name in bounds:
                min_val, max_val = bounds[name]
                params_scaled[:, :, i] = params_normalized[:, :, i] * (max_val - min_val) + min_val
            else:
                params_scaled[:, :, i] = params_normalized[:, :, i]
    elif params_normalized.ndim == 2:
        # 形状: (n_basins, n_params)
        for i, name in enumerate(param_names):
            if name in bounds:
                min_val, max_val = bounds[name]
                params_scaled[:, i] = params_normalized[:, i] * (max_val - min_val) + min_val
            else:
                params_scaled[:, i] = params_normalized[:, i]
    else:
        raise ValueError(f"Unsupported parameter shape: {params_normalized.shape}")

    return params_scaled


def load_mc_dropout_parameters(mc_dropout_dir: str, dataset: str = "train",
                               param_names: list = None, param_bounds: dict = None):
    """
    加载 MC Dropout 参数采样结果并反缩放到物理范围

    Parameters
    ----------
    mc_dropout_dir : str
        MC Dropout 结果目录
    dataset : str
        数据集名称
    param_names : list, optional
        参数名称列表
    param_bounds : dict, optional
        参数边界字典

    Returns
    -------
    params_samples : np.ndarray
        参数采样（物理范围）
    params_stats : dict
        参数统计量（物理范围）
    """
    params_file = os.path.join(
        mc_dropout_dir, f"{dataset}_parameters_samples.npz"
    )
    stats_file = os.path.join(mc_dropout_dir, f"{dataset}_parameters_stats.npz")

    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameters file not found: {params_file}")

    params_data = np.load(params_file)
    params_samples = params_data["samples"]  # (n_samples, n_basins, n_params)

    stats_data = np.load(stats_file)
    params_stats = {
        "mean": stats_data["mean"],
        "std": stats_data["std"],
        "p10": stats_data["p10"],
        "p90": stats_data["p90"],
    }

    # 如果提供了参数边界，进行反缩放
    if param_names is not None and param_bounds is not None:
        print("Descaling parameters to physical range...")
        params_samples = descale_params(params_samples, param_names, param_bounds)
        params_stats["mean"] = descale_params(params_stats["mean"], param_names, param_bounds)
        params_stats["std"] = descale_params(params_stats["std"], param_names, param_bounds)
        params_stats["p10"] = descale_params(params_stats["p10"], param_names, param_bounds)
        params_stats["p90"] = descale_params(params_stats["p90"], param_names, param_bounds)
        print(f"  Descaled range: [{params_samples.min():.4f}, {params_samples.max():.4f}]")

    return params_samples, params_stats


def plot_parameter_distributions(
    params_samples: np.ndarray,
    params_stats: dict,
    basin_idx: int,
    param_names: list = None,
    param_bounds: dict = None,
    save_path: str = None,
    dataset: str = "train",
):
    """
    绘制指定流域的参数分布图

    Parameters
    ----------
    params_samples : np.ndarray
        参数采样 (n_samples, n_basins, n_params)
    params_stats : dict
        参数统计量
    basin_idx : int
        流域索引
    param_names : list, optional
        参数名称列表
    param_bounds : dict, optional
        参数边界字典，键为参数名，值为 [min, max]
    save_path : str, optional
        保存路径
    dataset : str
        数据集名称 (train/eval)
    """
    n_samples, n_basins, n_params = params_samples.shape

    if basin_idx >= n_basins:
        raise ValueError(
            f"Basin index {basin_idx} out of range [0, {n_basins - 1}]"
        )

    # 提取指定流域的参数
    basin_params = params_samples[:, basin_idx, :]  # (n_samples, n_params)
    basin_mean = params_stats["mean"][basin_idx, :]
    basin_std = params_stats["std"][basin_idx, :]
    basin_p10 = params_stats["p10"][basin_idx, :]
    basin_p90 = params_stats["p90"][basin_idx, :]

    # 如果没有提供参数名称，使用默认名称
    if param_names is None:
        param_names = [f"Param_{i + 1}" for i in range(n_params)]

    # 计算子图布局
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Parameter Distributions - Basin {basin_idx} ({dataset.upper()} Dataset) - Physical Scale",
        fontsize=16,
        fontweight="bold",
    )

    # 展平 axes 以便迭代
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_params > 1 else [axes]

    for i in range(n_params):
        ax = axes[i]
        param_values = basin_params[:, i]
        param_name = param_names[i]

        # 获取参数边界
        if param_bounds and param_name in param_bounds:
            bounds = param_bounds[param_name]
            xlim = bounds
        else:
            xlim = None

        # 绘制直方图
        n_bins = min(30, n_samples // 3)
        ax.hist(
            param_values,
            bins=n_bins,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            density=True,
            label="Samples",
            range=xlim,  # 设置直方图的范围
        )

        # 设置 x 轴范围
        if xlim:
            ax.set_xlim(xlim)

        # 添加统计线
        ylim = ax.get_ylim()
        ax.axvline(
            basin_mean[i],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {basin_mean[i]:.4f}",
        )
        ax.axvline(
            basin_p10[i],
            color="green",
            linestyle=":",
            linewidth=1.5,
            label=f"P10: {basin_p10[i]:.4f}",
        )
        ax.axvline(
            basin_p90[i],
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=f"P90: {basin_p90[i]:.4f}",
        )

        # 设置标题和标签
        ax.set_title(f"{param_names[i]}\nStd: {basin_std[i]:.4f}", fontsize=12)
        ax.set_xlabel("Parameter Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


def main():
    """主函数"""
    # 配置路径
    mc_dropout_dir = "/workspace/my_deltamodel/project/deal_hydro/output/camels_559/train1989-1998/no_multi/Parameterize_E100_R365_B100_n1_noLn_noWU_42/hbv96/KgeBatchLoss/stat/test1999-2009_Ep100/mc_dropout"
    figures_dir = "/workspace/my_deltamodel/project/deal_hydro/figures"

    # 确保输出目录存在
    os.makedirs(figures_dir, exist_ok=True)

    # 参数设置
    basin_idx = 0  # 选择第一个流域，可以修改
    dataset = "train"  # 可选 "train" 或 "eval"

    # 参数名称（根据实际模型调整）
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

    print(f"Loading MC Dropout results from: {mc_dropout_dir}")
    print(f"Dataset: {dataset}")
    print(f"Basin index: {basin_idx}")

    # 加载数据并反缩放到物理范围
    params_samples, params_stats = load_mc_dropout_parameters(
        mc_dropout_dir, dataset, param_names=param_names, param_bounds=HBV96_PARAMS_BOUNDS
    )
    n_samples, n_basins, n_params = params_samples.shape

    print(
        f"Loaded {n_samples} samples for {n_basins} basins with {n_params} parameters"
    )

    # 调整参数名称列表长度
    param_names = param_names[:n_params]

    # 绘制分布图
    save_path = os.path.join(
        figures_dir, f"parameterize_params_basin{basin_idx}.png"
    )
    plot_parameter_distributions(
        params_samples,
        params_stats,
        basin_idx,
        param_names=param_names,
        param_bounds=HBV96_PARAMS_BOUNDS,
        save_path=save_path,
        dataset=dataset,
    )

    print("\nParameter statistics for basin", basin_idx, "(Physical Scale)")
    print("-" * 80)
    for i, name in enumerate(param_names):
        bounds = HBV96_PARAMS_BOUNDS.get(name, [None, None])
        print(
            f"{name:12s}: Mean={params_stats['mean'][basin_idx, i]:.4f}, "
            f"Std={params_stats['std'][basin_idx, i]:.4f}, "
            f"P10={params_stats['p10'][basin_idx, i]:.4f}, "
            f"P90={params_stats['p90'][basin_idx, i]:.4f}, "
            f"Bounds={bounds}"
        )


if __name__ == "__main__":
    main()
