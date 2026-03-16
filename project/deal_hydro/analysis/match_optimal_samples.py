"""
MC Dropout 最优样本匹配分析

功能：
1. 从 train metrics_samples 中找出每个流域的最优样本（基于指定指标，如 NSE）
2. 获取该最优样本的索引
3. 使用相同索引从 eval metrics_samples 中提取对应的指标值
4. 使用相同索引从 eval predictions/parameters 中提取对应的结果
5. 保存结果到 CSV 文件

注意：直接使用 MC Dropout 已计算好的 metrics_samples.npz，无需重新计算
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def load_mc_dropout_data(mc_dropout_dir: str, dataset: str):
    """加载 MC Dropout 数据"""
    # 加载预测采样
    preds_file = os.path.join(mc_dropout_dir, f"{dataset}_predictions_samples.npz")
    preds_data = np.load(preds_file)
    preds_samples = preds_data['samples']  # (n_samples, n_timesteps, n_basins)

    # 加载参数采样
    params_file = os.path.join(mc_dropout_dir, f"{dataset}_parameters_samples.npz")
    params_data = np.load(params_file)
    params_samples = params_data['samples']  # (n_samples, n_basins, n_params)

    # 加载指标采样（已经计算好的）
    metrics_file = os.path.join(mc_dropout_dir, f"{dataset}_metrics_samples.npz")
    metrics_data = np.load(metrics_file)

    # 将 metrics_data 转换为字典，每个指标是 (n_samples,) 的数组（全局平均）
    # 注意：如果需要每个流域的指标，需要重新计算或修改 FasterTrainer 保存逻辑
    metrics_dict = {key: metrics_data[key] for key in metrics_data.keys()}

    return preds_samples, params_samples, metrics_dict


def calculate_basin_level_metrics(preds_samples: np.ndarray, obs: np.ndarray, metric_name: str = 'nse'):
    """
    从预测采样计算每个流域的指标

    Parameters
    ----------
    preds_samples : np.ndarray
        预测采样 (n_samples, n_timesteps, n_basins)
    obs : np.ndarray
        观测值 (n_timesteps, n_basins)
    metric_name : str
        指标名称

    Returns
    -------
    all_metrics : np.ndarray
        每个样本每个流域的指标 (n_samples, n_basins)
    """
    from dmg.core.calc.metrics import Metrics

    n_samples, n_timesteps, n_basins = preds_samples.shape
    all_metrics = np.zeros((n_samples, n_basins))

    for sample_idx in range(n_samples):
        sample_preds = preds_samples[sample_idx]  # (n_timesteps, n_basins)

        # 转置为 (n_basins, n_timesteps)
        preds_T = np.swapaxes(sample_preds, 0, 1)
        obs_T = np.swapaxes(obs, 0, 1)

        # 计算指标
        metrics = Metrics(preds_T, obs_T, [metric_name])
        metric_attr = f"{metric_name}_vals"
        all_metrics[sample_idx] = getattr(metrics, metric_attr)

    return all_metrics


def find_optimal_samples_from_metrics(
    all_metrics: np.ndarray,
    maximize: bool = True
):
    """
    从指标矩阵中找出每个流域的最优样本索引

    Parameters
    ----------
    all_metrics : np.ndarray
        指标矩阵 (n_samples, n_basins)
    maximize : bool
        是否最大化指标（True 表示越大越好，如 NSE、KGE）

    Returns
    -------
    optimal_indices : np.ndarray
        每个流域的最优样本索引 (n_basins,)
    optimal_metrics : np.ndarray
        每个流域的最优指标值 (n_basins,)
    """
    # 找出每个流域的最优样本
    if maximize:
        optimal_indices = np.nanargmax(all_metrics, axis=0)  # (n_basins,)
        optimal_metrics = np.nanmax(all_metrics, axis=0)
    else:
        optimal_indices = np.nanargmin(all_metrics, axis=0)
        optimal_metrics = np.nanmin(all_metrics, axis=0)

    return optimal_indices, optimal_metrics


def extract_optimal_eval_results(
    eval_all_metrics: np.ndarray,
    eval_preds: np.ndarray,
    eval_params: np.ndarray,
    optimal_indices: np.ndarray,
):
    """
    根据 train 最优索引提取 eval 对应结果

    Parameters
    ----------
    eval_all_metrics : np.ndarray
        eval 指标矩阵 (n_samples, n_basins)，可以是多个指标的字典
    eval_preds : np.ndarray
        测试集预测 (n_samples, n_timesteps, n_basins)
    eval_params : np.ndarray
        测试集参数 (n_samples, n_basins, n_params)
    optimal_indices : np.ndarray
        最优样本索引 (n_basins,)

    Returns
    -------
    optimal_eval_preds : np.ndarray
        最优预测 (n_timesteps, n_basins)
    optimal_eval_params : np.ndarray
        最优参数 (n_basins, n_params)
    optimal_eval_metrics : dict
        最优指标值（每个流域）
    """
    n_samples, n_timesteps, n_basins = eval_preds.shape
    n_params = eval_params.shape[2]

    # 提取每个流域的最优预测和参数
    optimal_eval_preds = np.zeros((n_timesteps, n_basins))
    optimal_eval_params = np.zeros((n_basins, n_params))

    for basin_idx in range(n_basins):
        sample_idx = optimal_indices[basin_idx]
        optimal_eval_preds[:, basin_idx] = eval_preds[sample_idx, :, basin_idx]
        optimal_eval_params[basin_idx, :] = eval_params[sample_idx, basin_idx, :]

    # 提取最优指标值
    optimal_eval_metrics = {}
    if isinstance(eval_all_metrics, dict):
        # 如果是字典，提取每个指标
        for metric_name, metric_matrix in eval_all_metrics.items():
            optimal_vals = np.zeros(n_basins)
            for basin_idx in range(n_basins):
                sample_idx = optimal_indices[basin_idx]
                optimal_vals[basin_idx] = metric_matrix[sample_idx, basin_idx]
            optimal_eval_metrics[metric_name] = optimal_vals
    else:
        # 如果是单个矩阵
        optimal_vals = np.zeros(n_basins)
        for basin_idx in range(n_basins):
            sample_idx = optimal_indices[basin_idx]
            optimal_vals[basin_idx] = eval_all_metrics[sample_idx, basin_idx]
        optimal_eval_metrics['metric'] = optimal_vals

    return optimal_eval_preds, optimal_eval_params, optimal_eval_metrics


def save_results_to_csv(
    optimal_indices: np.ndarray,
    train_optimal_metrics: np.ndarray,
    eval_optimal_metrics: dict,
    eval_optimal_params: np.ndarray,
    train_all_metrics: np.ndarray,
    save_dir: str,
    metric_name: str = 'nse',
    param_names: list = None
):
    """保存结果到 CSV 文件"""
    n_basins = len(optimal_indices)
    n_params = eval_optimal_params.shape[1]

    if param_names is None:
        param_names = [f"param_{i+1}" for i in range(n_params)]

    # 创建主结果 DataFrame
    results_df = pd.DataFrame({
        'basin_id': np.arange(n_basins),
        'optimal_sample_idx': optimal_indices,
        f'train_{metric_name}_optimal': train_optimal_metrics,
    })

    # 添加 eval 指标
    for metric_key, metric_vals in eval_optimal_metrics.items():
        results_df[f'eval_{metric_key}_at_train_optimal'] = metric_vals

    # 添加最优参数
    for i, param_name in enumerate(param_names[:n_params]):
        results_df[f'optimal_{param_name}'] = eval_optimal_params[:, i]

    # 添加统计信息
    results_df['train_metric_mean'] = np.nanmean(train_all_metrics, axis=0)
    results_df['train_metric_std'] = np.nanstd(train_all_metrics, axis=0)
    results_df['train_metric_range'] = np.nanmax(train_all_metrics, axis=0) - np.nanmin(train_all_metrics, axis=0)

    # 保存主结果
    main_csv = os.path.join(save_dir, f"optimal_train_eval_correspondence_{metric_name}.csv")
    results_df.to_csv(main_csv, index=False)
    print(f"Main results saved to: {main_csv}")

    # 保存汇总统计
    summary_stats = {
        f'train_{metric_name}_mean': np.nanmean(train_optimal_metrics),
        f'train_{metric_name}_std': np.nanstd(train_optimal_metrics),
        f'train_{metric_name}_median': np.nanmedian(train_optimal_metrics),
    }

    for metric_key, metric_vals in eval_optimal_metrics.items():
        summary_stats[f'eval_{metric_key}_mean'] = np.nanmean(metric_vals)
        summary_stats[f'eval_{metric_key}_std'] = np.nanstd(metric_vals)
        summary_stats[f'eval_{metric_key}_median'] = np.nanmedian(metric_vals)

    summary_df = pd.DataFrame([summary_stats])
    summary_csv = os.path.join(save_dir, f"optimal_correspondence_summary_{metric_name}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to: {summary_csv}")

    return results_df, summary_df


def main():
    """主函数"""
    # 配置路径
    mc_dropout_dir = "/workspace/my_deltamodel/project/deal_hydro/output/camels_559/train1989-1998/no_multi/Parameterize_E100_R365_B100_n1_noLn_noWU_42/hbv96/KgeBatchLoss/stat/test1999-2009_Ep100/mc_dropout"
    csv_dir = "/workspace/my_deltamodel/project/deal_hydro/analysis/csv"

    # 确保输出目录存在
    os.makedirs(csv_dir, exist_ok=True)

    # 参数设置
    metric_name = 'kge'  # 优化指标，可选 'nse', 'kge', 'rmse' 等
    maximize = True  # NSE 和 KGE 越大越好

    # 参数名称（HBV96 模型）
    param_names = [
        'BETA', 'FC', 'K0', 'K1', 'K2', 'LP',
        'PERC', 'UZL', 'TT', 'CFMAX', 'CFR', 'CWH'
    ]

    print("=" * 80)
    print("MC Dropout Optimal Sample Matching Analysis")
    print("=" * 80)

    # 1. 加载 train 数据
    print("\n[1/4] Loading train MC Dropout data...")
    train_preds, train_params, train_metrics_dict = load_mc_dropout_data(mc_dropout_dir, "train")
    print(f"  Train predictions shape: {train_preds.shape}")
    print(f"  Train parameters shape: {train_params.shape}")
    print(f"  Train metrics available: {list(train_metrics_dict.keys())}")

    # 2. 加载 eval 数据
    print("\n[2/4] Loading eval MC Dropout data...")
    eval_preds, eval_params, eval_metrics_dict = load_mc_dropout_data(mc_dropout_dir, "eval")
    print(f"  Eval predictions shape: {eval_preds.shape}")
    print(f"  Eval parameters shape: {eval_params.shape}")
    print(f"  Eval metrics available: {list(eval_metrics_dict.keys())}")

    # 3. 提取流域级别指标
    print(f"\n[3/4] Extracting basin-level metrics...")

    # 检查是否有流域级别的指标
    train_metric_key = f"{metric_name}_basin"
    if train_metric_key not in train_metrics_dict:
        raise ValueError(
            f"Basin-level metric '{train_metric_key}' not found in train metrics. "
            f"Available keys: {list(train_metrics_dict.keys())}\n"
            f"Please re-run MC Dropout evaluation with updated FasterTrainer."
        )

    # 提取 train 流域级别指标
    train_all_metrics = train_metrics_dict[train_metric_key]  # (n_samples, n_basins)
    print(f"  Train {metric_name} shape: {train_all_metrics.shape}")

    # 提取 eval 所有流域级别指标
    eval_all_metrics = {}
    for key in eval_metrics_dict.keys():
        if key.endswith('_basin'):
            metric_base_name = key[:-6]  # 移除 '_basin'
            eval_all_metrics[metric_base_name] = eval_metrics_dict[key]
            print(f"  Eval {metric_base_name} shape: {eval_metrics_dict[key].shape}")

    # 4. 找出 train 最优样本
    print(f"\n[4/4] Finding optimal samples on train (metric: {metric_name})...")
    optimal_indices, train_optimal_metrics = find_optimal_samples_from_metrics(
        train_all_metrics, maximize
    )
    print(f"  Optimal indices shape: {optimal_indices.shape}")
    print(f"  Train optimal {metric_name} - Mean: {np.nanmean(train_optimal_metrics):.4f}, "
          f"Std: {np.nanstd(train_optimal_metrics):.4f}")

    # 5. 提取 eval 对应结果
    print(f"\n[5/5] Extracting corresponding eval results...")
    optimal_eval_preds, optimal_eval_params, optimal_eval_metrics = extract_optimal_eval_results(
        eval_all_metrics, eval_preds, eval_params, optimal_indices
    )
    print(f"  Optimal eval predictions shape: {optimal_eval_preds.shape}")
    print(f"  Optimal eval parameters shape: {optimal_eval_params.shape}")
    print(f"  Eval metrics at train optimal:")
    for metric_key, metric_vals in optimal_eval_metrics.items():
        print(f"    {metric_key.upper()}: Mean={np.nanmean(metric_vals):.4f}, "
              f"Std={np.nanstd(metric_vals):.4f}")

    # 6. 保存结果
    print(f"\n[6/6] Saving results to CSV...")
    results_df, summary_df = save_results_to_csv(
        optimal_indices, train_optimal_metrics, optimal_eval_metrics,
        optimal_eval_params, train_all_metrics, csv_dir, metric_name, param_names
    )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nResults saved to: {csv_dir}")
    print(f"  - optimal_train_eval_correspondence_{metric_name}.csv")
    print(f"  - optimal_correspondence_summary_{metric_name}.csv")

    # 显示前几行结果
    print("\nFirst 5 basins:")
    print(results_df.head().to_string())


if __name__ == "__main__":
    main()
