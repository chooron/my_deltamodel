"""
FasterTrainer — 优化的训练器

基于 Trainer 的改进版本，主要优化：
- 更高效的训练循环
- 改进的内存管理
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid, create_dl_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class FasterTrainer(BaseTrainer):
    """优化的训练器，完整向后兼容 Trainer。

    在原 Trainer 基础上进行了性能优化。
    所有接口（evaluate、inference、load_states、init_optimizer 等）
    与 Trainer 保持一致。
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = import_data_sampler(config["data_sampler"])(config)
        self.is_in_train = False

        if "train" in config["mode"]:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            log.info("Initializing experiment")
            self.epochs = self.config["train"]["epochs"]

            self.loss_func = loss_func or load_criterion(
                self.train_dataset["target"],
                config["loss_function"],
                device=config["device"],
            )
            self.model.loss_func = self.loss_func

            self.optimizer = optimizer or self.init_optimizer()
            if config["delta_model"]["nn_model"]["lr_scheduler"]:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            self.load_states()

    # ── 以下方法与 Trainer 完全一致 ──────────────────────────────────────────

    def init_optimizer(self) -> torch.optim.Optimizer:
        name = self.config["train"]["optimizer"]
        optimizer_dict = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "Adadelta": torch.optim.Adadelta,
            "RMSprop": torch.optim.RMSprop,
        }

        cls = optimizer_dict[name]
        if cls is None:
            raise ValueError(
                f"Optimizer '{name}' not recognized. "
                f"Available options are: {list(optimizer_dict.keys())}"
            )

        try:
            self.optimizer = cls(
                self.model.get_parameters(),
                lr=self.config["train"]["learning_rate"],
                weight_decay=self.config["train"].get("weight_decay", 0.0),
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing optimizer: {e}") from e
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        name = self.config["delta_model"]["train"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        cls = scheduler_dict[name]
        if cls is None:
            raise ValueError(
                f"Scheduler '{name}' not recognized. "
                f"Available options are: {list(scheduler_dict.keys())}"
            )

        try:
            self.scheduler = cls(
                self.optimizer,
                **self.config["delta_model"]["train"]["lr_scheduler_params"],
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        path = self.config["model_path"]
        for file in os.listdir(path):
            if "train_state" in file:
                checkpoint = torch.load(os.path.join(path, file))

                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.model.load_model(epoch=checkpoint["epoch"])
                self.start_epoch = checkpoint["epoch"] + 1

                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                torch.set_rng_state(checkpoint["random_state"])
                if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                return
            else:
                self.start_epoch = 1

    def train(self) -> None:
        self.is_in_train = True

        if self.config.get("data_sampler") == "DlSampler":
            n_samples, n_minibatch, n_timesteps = create_dl_training_grid(
                self.train_dataset["xc_nn_norm"],
                self.config,
            )
        else:
            n_samples, n_minibatch, n_timesteps = create_training_grid(
                self.train_dataset["xc_nn_norm"],
                self.config,
            )

        log.info(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs"
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

    # ── [改动 2] train_one_epoch：增加 TBPTT 分支 ──────────────────────────

    def train_one_epoch(
        self, epoch: int, n_samples: int, n_minibatch: int, n_timesteps: int
    ) -> None:
        """Train model for one epoch.

        [改动 2] use_tbptt=True 时调用 _train_step_tbptt()；
                 use_tbptt=False 时走原有逻辑，行为与 Trainer 完全相同。
        """
        start_time = time.perf_counter()
        prog_str = f"Epoch {epoch}/{self.epochs}"

        self.current_epoch = epoch
        self.total_loss = 0.0

        for mb in tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=prog_str,
            leave=False,
            dynamic_ncols=True,
        ):
            self.current_batch = mb

            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            # 标准训练流程
            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.get_parameters(), max_norm=1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_val = loss.item()

            self.total_loss += loss_val

            if self.verbose:
                tqdm.tqdm.write(
                    f"Epoch {epoch}, batch {mb} | loss: {loss_val:.6f}"
                )

        if self.use_scheduler:
            self.scheduler.step()

        if self.verbose:
            log.info(f"\n ---- \n Epoch {epoch} total loss: {self.total_loss}")
        self._log_epoch_stats(
            epoch, self.model.loss_dict, n_minibatch, start_time
        )

        if epoch % self.config["train"]["save_epoch"] == 0:
            self.model.save_model(epoch)
            save_train_state(
                self.config,
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )

    def evaluate(self) -> None:
        """Standard evaluation without MC Dropout."""
        self.is_in_train = False

        batch_predictions = []
        observations = self.eval_dataset["target"]

        n_samples = self.eval_dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Validating Model: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.eval_dataset, batch_start, batch_end
        )

        log.info("Saving model outputs + Calculating metrics")
        save_outputs(
            self.config, batch_predictions, observations, create_dirs=True
        )
        self.predictions = self._batch_data(batch_predictions)
        self.calc_metrics(batch_predictions, observations)

    def evaluate_mc_dropout(self, n_samples: int = 100) -> None:
        """
        MC Dropout 评估：对 train 和 eval 数据集进行多次前向传播。

        Parameters
        ----------
        n_samples : int
            MC Dropout 采样次数，默认 100

        保存内容
        --------
        - parameters_samples.npz: 所有采样的参数 (n_samples, n_basins, n_params)
        - predictions_samples.npz: 所有采样的预测 (n_samples, n_timesteps, n_basins)
        - metrics_samples.npz: 每次采样的评估指标
        - parameters_stats.npz: 参数统计量 (mean, std, p10, p90)
        - predictions_stats.npz: 预测统计量 (mean, std, p10, p90)
        """
        self.is_in_train = False
        log.info(f"Starting MC Dropout evaluation with {n_samples} samples")

        # 获取模型
        model_name = list(self.model.model_dict.keys())[0]
        dpl_model = self.model.model_dict[model_name]
        nn_model = dpl_model.nn_model
        phy_model = dpl_model.phy_model

        # 设置为 train 模式以启用 Dropout
        nn_model.train()
        phy_model.eval()

        # 准备输出目录
        out_path = self.config["out_path"]
        mc_dropout_dir = os.path.join(out_path, "mc_dropout")
        os.makedirs(mc_dropout_dir, exist_ok=True)

        # 对 train 和 eval 数据集分别处理
        for dataset_name, dataset in [("train", self.train_dataset), ("eval", self.eval_dataset)]:
            if dataset is None:
                continue

            log.info(f"Processing {dataset_name} dataset with MC Dropout")

            # 批量前向传播收集所有采样
            all_params_samples, all_preds_samples, all_metrics = self._mc_dropout_forward(
                dataset, n_samples
            )

            # 保存结果
            self._save_mc_dropout_results(
                mc_dropout_dir, dataset_name, all_params_samples,
                all_preds_samples, all_metrics, dataset["target"]
            )

        log.info(f"MC Dropout evaluation complete. Results saved to {mc_dropout_dir}")

    def _mc_dropout_forward(
        self, dataset: dict, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        对整个数据集进行 MC Dropout 前向传播。

        Returns
        -------
        all_params_samples : np.ndarray
            参数采样 (n_samples, n_basins, n_params)
        all_preds_samples : np.ndarray
            预测采样 (n_samples, n_timesteps, n_basins)
        all_metrics : list
            每次采样的指标列表
        """
        model_name = list(self.model.model_dict.keys())[0]
        dpl_model = self.model.model_dict[model_name]
        nn_model = dpl_model.nn_model
        phy_model = dpl_model.phy_model

        n_basins = dataset["xc_nn_norm"].shape[1]
        n_timesteps_full = dataset["x_phy"].shape[0]
        warm_up = self.config["delta_model"]["phy_model"]["warm_up"]

        # actual output timesteps (model no longer includes warmup)
        n_timesteps_output = n_timesteps_full - warm_up
        n_params = nn_model.ny

        # 预分配数组
        all_params_samples = np.zeros((n_samples, n_basins, n_params), dtype=np.float32)
        all_preds_samples = np.zeros((n_samples, n_timesteps_output, n_basins), dtype=np.float32)
        all_metrics = []

        # 批处理设置 - MC Dropout 使用全量流域大小以充分利用 GPU
        batch_size = n_basins  # 使用全量流域大小，不分批
        batch_start = np.arange(0, n_basins, batch_size)
        batch_end = np.append(batch_start[1:], n_basins)

        # MC Dropout 采样循环
        for sample_idx in tqdm.tqdm(range(n_samples), desc="MC Dropout Sampling"):
            batch_predictions = []
            batch_parameters = []

            # 对每个 batch 进行前向传播
            for i in range(len(batch_start)):
                dataset_sample = self.sampler.get_validation_sample(
                    dataset, batch_start[i], batch_end[i]
                )

                with torch.no_grad():
                    # NN 预测参数
                    _, params = nn_model(dataset_sample)  # (batch, n_params, 1)
                    batch_parameters.append(params.squeeze(-1).cpu())

                    # 物理模型前向传播
                    output = phy_model(dataset_sample, (None, params))
                    batch_predictions.append(output)

            # 合并所有 batch 的参数
            params_full = torch.cat(batch_parameters, dim=0).numpy()  # (n_basins, n_params)
            all_params_samples[sample_idx] = params_full

            # 合并所有 batch 的预测
            preds_full = self._batch_data(batch_predictions, target_key="streamflow")  # (n_timesteps, n_basins)
            all_preds_samples[sample_idx] = preds_full

            # 计算当前采样的指标
            metrics = self._calc_sample_metrics(preds_full, dataset["target"])
            all_metrics.append(metrics)

        return all_params_samples, all_preds_samples, all_metrics

    def _calc_sample_metrics(self, predictions: np.ndarray, observations: torch.Tensor) -> dict:
        """
        计算单次采样的评估指标。

        Returns
        -------
        dict
            包含两类指标：
            - '{metric}_mean': 全局平均值（标量）
            - '{metric}_basin': 每个流域的值（数组）
        """
        obs_np = observations.cpu().numpy() if isinstance(observations, torch.Tensor) else observations

        # 使用 Metrics 类计算指标
        metrics_to_compute = self.config["test"].get("metrics", None)

        # 确保形状一致
        if predictions.ndim == 2 and obs_np.ndim == 3:
            obs_np = obs_np.squeeze(-1)

        if obs_np.shape[0] > predictions.shape[0]:
            obs_np = obs_np[:predictions.shape[0], :]

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
            result["nse_mean"] = float(np.nanmean(metrics_calc.nse))
            result["nse_basin"] = metrics_calc.nse  # (n_basins,)
            result["kge_mean"] = float(np.nanmean(metrics_calc.kge))
            result["kge_basin"] = metrics_calc.kge  # (n_basins,)
        else:
            # 只计算指定的指标
            for metric_name in metrics_to_compute:
                if hasattr(metrics_calc, metric_name):
                    values = getattr(metrics_calc, metric_name)
                    result[f"{metric_name}_mean"] = float(np.nanmean(values))
                    result[f"{metric_name}_basin"] = values  # (n_basins,)

        return result

    def _save_mc_dropout_results(
        self,
        save_dir: str,
        dataset_name: str,
        params_samples: np.ndarray,
        preds_samples: np.ndarray,
        metrics_list: list,
        observations: torch.Tensor,
    ) -> None:
        """保存 MC Dropout 结果到压缩的 npz 文件。"""

        # 1. 保存参数采样 (压缩)
        params_file = os.path.join(save_dir, f"{dataset_name}_parameters_samples.npz")
        np.savez_compressed(
            params_file,
            samples=params_samples,  # (n_samples, n_basins, n_params)
        )
        log.info(f"Saved parameters samples to {params_file}")

        # 2. 保存预测采样 (压缩)
        preds_file = os.path.join(save_dir, f"{dataset_name}_predictions_samples.npz")
        np.savez_compressed(
            preds_file,
            samples=preds_samples,  # (n_samples, n_timesteps, n_basins)
        )
        log.info(f"Saved predictions samples to {preds_file}")

        # 3. 保存每次采样的指标
        metrics_file = os.path.join(save_dir, f"{dataset_name}_metrics_samples.npz")

        # 分离全局平均指标和流域级别指标
        metrics_mean = {}  # 全局平均 (n_samples,)
        metrics_basin = {}  # 流域级别 (n_samples, n_basins)

        for key in metrics_list[0].keys():
            if key.endswith('_mean'):
                # 全局平均指标
                metric_name = key[:-5]  # 移除 '_mean' 后缀
                metrics_mean[metric_name] = np.array([m[key] for m in metrics_list])
            elif key.endswith('_basin'):
                # 流域级别指标
                metric_name = key[:-6]  # 移除 '_basin' 后缀
                metrics_basin[metric_name] = np.array([m[key] for m in metrics_list])

        # 保存两种格式的指标
        save_dict = {}
        # 全局平均指标 (n_samples,)
        for metric_name, values in metrics_mean.items():
            save_dict[f"{metric_name}_mean"] = values
        # 流域级别指标 (n_samples, n_basins)
        for metric_name, values in metrics_basin.items():
            save_dict[f"{metric_name}_basin"] = values

        np.savez_compressed(metrics_file, **save_dict)
        log.info(f"Saved metrics samples to {metrics_file}")
        log.info(f"  - Global mean metrics: {list(metrics_mean.keys())}")
        log.info(f"  - Basin-level metrics: {list(metrics_basin.keys())}")

        # 4. 计算并保存参数统计量
        params_stats_file = os.path.join(save_dir, f"{dataset_name}_parameters_stats.npz")
        np.savez_compressed(
            params_stats_file,
            mean=params_samples.mean(axis=0),
            std=params_samples.std(axis=0),
            p10=np.percentile(params_samples, 10, axis=0),
            p90=np.percentile(params_samples, 90, axis=0),
        )
        log.info(f"Saved parameters statistics to {params_stats_file}")

        # 5. 计算并保存预测统计量
        preds_stats_file = os.path.join(save_dir, f"{dataset_name}_predictions_stats.npz")
        np.savez_compressed(
            preds_stats_file,
            mean=preds_samples.mean(axis=0),
            std=preds_samples.std(axis=0),
            p10=np.percentile(preds_samples, 10, axis=0),
            p90=np.percentile(preds_samples, 90, axis=0),
        )
        log.info(f"Saved predictions statistics to {preds_stats_file}")

        # 6. 保存指标统计摘要
        summary_file = os.path.join(save_dir, f"{dataset_name}_metrics_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"MC Dropout Evaluation Summary - {dataset_name.upper()} Dataset\n")
            f.write("=" * 60 + "\n\n")

            # 全局平均指标统计
            f.write("Global Mean Metrics (averaged across all basins):\n")
            f.write("-" * 60 + "\n")
            for metric_name, values in metrics_mean.items():
                f.write(f"{metric_name.upper()}:\n")
                f.write(f"  Mean: {values.mean():.4f}\n")
                f.write(f"  Std:  {values.std():.4f}\n")
                f.write(f"  Min:  {values.min():.4f}\n")
                f.write(f"  Max:  {values.max():.4f}\n")
                f.write(f"  P10:  {np.percentile(values, 10):.4f}\n")
                f.write(f"  P90:  {np.percentile(values, 90):.4f}\n\n")

            # 流域级别指标统计
            f.write("\nBasin-Level Metrics Statistics:\n")
            f.write("-" * 60 + "\n")
            for metric_name, values in metrics_basin.items():
                # values shape: (n_samples, n_basins)
                f.write(f"{metric_name.upper()}:\n")
                f.write(f"  Overall Mean: {np.nanmean(values):.4f}\n")
                f.write(f"  Overall Std:  {np.nanstd(values):.4f}\n")
                f.write(f"  Best Basin Mean: {np.nanmax(np.nanmean(values, axis=0)):.4f}\n")
                f.write(f"  Worst Basin Mean: {np.nanmin(np.nanmean(values, axis=0)):.4f}\n\n")

        log.info(f"Saved metrics summary to {summary_file}")

    def inference(self) -> None:
        self.is_in_train = False

        batch_predictions = []

        n_samples = self.dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(
            0, n_samples, self.config["simulation"]["batch_size"]
        )
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.dataset, batch_start, batch_end
        )

        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        data = {}
        try:
            if target_key:
                return torch.cat(
                    [x[target_key] for x in batch_list], dim=1
                ).cpu().numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    pass
                else:
                    pass
                data[key] = (
                    torch.cat([d[key] for d in batch_list], dim=1).cpu().numpy()
                )
            return data

        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray,
    ):
        batch_predictions = []
        model_name = self.config["delta_model"]["phy_model"]["model"][0]
        for i in tqdm.tqdm(
            range(len(batch_start)),
            desc="Forwarding",
            leave=False,
            dynamic_ncols=True,
        ):
            self.current_batch = i

            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[i],
                batch_end[i],
            )
            if self.config["test"]["split_dataset"]:
                total_time_steps = dataset_sample["x_phy"].shape[0]
                prediction_time_chunks = []
                prediction_length = self.config["delta_model"]["rho"]
                warmup_length = self.config["delta_model"]["phy_model"]["warm_up"]
                time_starts = range(
                    0,
                    total_time_steps - prediction_length - warmup_length + 1,
                    prediction_length,
                )
                for t_start in time_starts:
                    t_end = t_start + prediction_length + warmup_length
                    time_window_input = {
                        key: tensor[t_start:t_end, ...]
                        if len(tensor.shape) > 2
                        else tensor
                        for key, tensor in dataset_sample.items()
                    }
                    prediction_window = self.model(time_window_input, eval=True)
                    prediction_valid_part = {
                        key: tensor.cpu().detach()
                        for key, tensor in prediction_window.items()
                    }
                    prediction_time_chunks.append(prediction_valid_part)
                collated_chunks = {key: [] for key in prediction_time_chunks[0]}
                for chunk in prediction_time_chunks:
                    for key, ten in chunk.items():
                        collated_chunks[key].append(ten)
                prediction = {
                    key: torch.cat(tensors, dim=0)
                    for key, tensors in collated_chunks.items()
                }
                batch_predictions.append(prediction)
            else:
                prediction = self.model(dataset_sample, eval=True)
                prediction = {
                    key: tensor.cpu().detach()
                    for key, tensor in prediction.items()
                }
                batch_predictions.append(prediction)
        return batch_predictions

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        target_name = self.config["train"]["target"][0]
        predictions = self._batch_data(batch_predictions, target_name)
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)

        target = target[: len(predictions), :]

        # 从配置中获取要计算的指标列表
        metrics_to_compute = self.config["test"].get("metrics", None)
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
            metrics_to_compute,
        )

        metrics.dump_metrics(self.config["out_path"])

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        avg_loss_dict = {
            key: value / n_minibatch + 1 for key, value in loss_dict.items()
        }
        loss = ", ".join(
            f"{key}: {value:.6f}" for key, value in avg_loss_dict.items()
        )
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(
            torch.cuda.memory_reserved(device=self.config["device"]) * 0.000001
        )

        log.info(
            f"Loss after epoch {epoch}: {loss} \n"
            f"~ Runtime {elapsed:.2f} s, {mem_aloc} Mb reserved GPU memory",
        )
