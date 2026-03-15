"""测试 _calc_sample_metrics 方法是否正确返回 _mean 和 _basin 键"""
import os
import sys
import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

# 创建模拟数据
n_timesteps = 100
n_basins = 10

predictions = np.random.randn(n_timesteps, n_basins)
observations = torch.randn(n_timesteps, n_basins)

# 创建模拟配置
config = {
    "test": {},  # 没有指定 metrics，应该使用默认的 NSE 和 KGE
    "delta_model": {
        "phy_model": {
            "warm_up": 0
        }
    }
}

# 导入 FasterTrainer 并创建实例
from dmg.trainers.faster_trainer import FasterTrainer

# 创建一个最小的 trainer 实例来测试方法
class MockTrainer:
    def __init__(self, config):
        self.config = config

    def _calc_sample_metrics(self, predictions, observations):
        """复制 FasterTrainer 的方法"""
        from dmg.core.calc.metrics import Metrics

        obs_np = observations.cpu().numpy() if isinstance(observations, torch.Tensor) else observations

        # 使用 Metrics 类计算指标
        metrics_to_compute = self.config["test"].get("metrics", None)

        # 确保形状一致
        if predictions.ndim == 2 and obs_np.ndim == 3:
            obs_np = obs_np.squeeze(-1)

        # 移除 warmup（如果 target 包含 warmup）
        warm_up = self.config["delta_model"]["phy_model"]["warm_up"]
        if obs_np.shape[0] > predictions.shape[0]:
            obs_np = obs_np[warm_up:warm_up + predictions.shape[0], :]

        # 创建 Metrics 对象
        metrics_calc = Metrics(
            np.swapaxes(predictions, 1, 0),
            np.swapaxes(obs_np, 1, 0),
            metrics_to_compute,
        )

        # 提取计算的指标（同时保存全局平均和流域级别）
        result = {}
        if metrics_to_compute is None:
            # 如果没有指定，计算默认指标
            result["nse_mean"] = float(np.nanmean(metrics_calc.nse_vals))
            result["nse_basin"] = metrics_calc.nse_vals  # (n_basins,)
            result["kge_mean"] = float(np.nanmean(metrics_calc.kge_vals))
            result["kge_basin"] = metrics_calc.kge_vals  # (n_basins,)
        else:
            # 只计算指定的指标
            for metric_name in metrics_to_compute:
                metric_attr = f"{metric_name}_vals"
                if hasattr(metrics_calc, metric_attr):
                    values = getattr(metrics_calc, metric_attr)
                    result[f"{metric_name}_mean"] = float(np.nanmean(values))
                    result[f"{metric_name}_basin"] = values  # (n_basins,)

        return result

trainer = MockTrainer(config)
result = trainer._calc_sample_metrics(predictions, observations)

print("Test _calc_sample_metrics:")
print(f"  Keys: {list(result.keys())}")
for key, value in result.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value).__name__} = {value:.4f}")

# 测试保存逻辑
print("\nTest save logic:")
metrics_list = [result, result, result]  # 模拟 3 个样本

metrics_mean = {}
metrics_basin = {}

for key in metrics_list[0].keys():
    if key.endswith('_mean'):
        metric_name = key[:-5]
        metrics_mean[metric_name] = np.array([m[key] for m in metrics_list])
        print(f"  {metric_name}_mean: shape={metrics_mean[metric_name].shape}")
    elif key.endswith('_basin'):
        metric_name = key[:-6]
        metrics_basin[metric_name] = np.array([m[key] for m in metrics_list])
        print(f"  {metric_name}_basin: shape={metrics_basin[metric_name].shape}")

print(f"\nmetrics_mean keys: {list(metrics_mean.keys())}")
print(f"metrics_basin keys: {list(metrics_basin.keys())}")
