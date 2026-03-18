import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import sys
import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid, create_dl_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_outputsv2, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


# try:
#     from ray import tune
#     from ray.air import Checkpoint
# except ImportError:
#     log.warning('Ray Tune is not installed or is misconfigured. Tuning will be disabled.')


class CalTrainer(BaseTrainer):
    """Generic, unified trainer for neural networks and differentiable models.

    Inspired by the Hugging Face Trainer class.

    Retrieves and formats data, initializes optimizers/schedulers/loss functions,
    and runs training and testing/inference loops.

    Parameters
    ----------
    config
        Configuration settings for the model and experiment.
    model
        Learnable model object. If not provided, a new model is initialized.
    train_dataset
        Training dataset dictionary.
    eval_dataset
        Testing/inference dataset dictionary.
    dataset
        Inference dataset dictionary.
    loss_func
        Loss function object. If not provided, a new loss function is initialized.
    optimizer
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    verbose
        Whether to print verbose output.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
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

            log.debug("Initializing experiment")
            self.epochs = self.config["train"]["epochs"]

            # Loss function
            self.loss_func = loss_func or load_criterion(
                self.train_dataset["target"],
                config["loss_function"],
                device=config["device"],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config["delta_model"]["nn_model"]["lr_scheduler"]:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            # self.start_epoch = self.config['train']['start_epoch'] + 1
            # if self.start_epoch > 1:
            self.load_states()
        elif "test" in config["mode"]:
            self.load_test_states()

        

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer.

        Adding additional optimizers is possible by extending the optimizer_dict.

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer object.
        """
        name = self.config["train"]["optimizer"]
        optimizer_dict = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "Adadelta": torch.optim.Adadelta,
            "RMSprop": torch.optim.RMSprop,
        }

        # Fetch optimizer class
        cls = optimizer_dict[name]
        if cls is None:
            raise ValueError(
                f"Optimizer '{name}' not recognized. "
                f"Available options are: {list(optimizer_dict.keys())}"
            )

        # Initialize
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
        """Initialize a learning rate scheduler for the optimizer.

        torch.optim.lr_scheduler.LRScheduler
            Initialized learning rate scheduler object.
        """
        name = self.config["delta_model"]["nn_model"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        }

        # Fetch scheduler class
        cls = scheduler_dict[name]
        if cls is None:
            raise ValueError(
                f"Scheduler '{name}' not recognized. "
                f"Available options are: {list(scheduler_dict.keys())}"
            )

        # Initialize
        try:
            self.scheduler = cls(
                self.optimizer,
                **self.config["delta_model"]["nn_model"]["lr_scheduler_params"],
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint to resume
        training if a checkpoint file exists.
        """
        path = self.config["model_path"]
        for file in os.listdir(path):
            # Check for state checkpoint: looks like `train_state_epoch_XX.pt`.
            if "train_state" in file:
                # and (str(self.start_epoch - 1) in file):
                # log.info("Loading trainer states --> Resuming Training from" /
                #          f" epoch {self.start_epoch}")

                checkpoint = torch.load(os.path.join(path, file), map_location=self.config["device"])

                # Restore optimizer states
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
                # # Restore model
                self.model.load_model(epoch=checkpoint["epoch"])
                self.start_epoch = checkpoint["epoch"] + 1

                if self.scheduler:
                    self.scheduler.load_state_dict(
                        checkpoint["scheduler_state_dict"]
                    )

                # Restore random states
                torch.set_rng_state(checkpoint["random_state"].cpu().byte())
                if (
                    torch.cuda.is_available()
                    and "cuda_random_state" in checkpoint
                ):
                    torch.cuda.set_rng_state_all(
                        checkpoint["cuda_random_state"].cpu().byte()
                    )
                log.debug(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                return
            else:
                self.start_epoch = 1

    def load_test_states(self) -> None:
        """
        Load model states for testing using the specified test epoch.

        This is separate from load_states to avoid modifying training logic.
        """
        path = self.config["model_path"]
        test_epoch = self.config["test"].get("test_epoch", None)

        if test_epoch is None:
            raise ValueError("'test_epoch' must be set in config['test'].")

        # Load model by explicit file name format: d{model}_Ep{epoch}.pt
        model_name = self.config["delta_model"]["phy_model"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]

        checkpoint_file = f"d{model_name}_Ep{int(test_epoch)}.pt"
        checkpoint_path = os.path.join(path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path} not found.")

        self.model.load_model(epoch=int(test_epoch))
        print(f"Loaded test checkpoint: {checkpoint_path}")

    def _emit_progress(self, message: str) -> None:
        """Emit progress even when the entry script does not configure logging."""
        if log.hasHandlers() and log.isEnabledFor(logging.INFO):
            log.info(message)
        else:
            print(message, flush=True)

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True

        # Setup a training grid (number of samples, minibatches, and timesteps)
        # 根据 data_sampler 类型选择合适的训练网格计算函数
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

        # 训练开始摘要
        n_basins = self.train_dataset["xc_nn_norm"].shape[1]
        optimizer_name = self.config["train"]["optimizer"]
        lr = self.config["train"]["learning_rate"]
        scheduler_name = self.config["delta_model"]["nn_model"].get("lr_scheduler", "None")
        self._emit_progress(
            f"[Train Start] epochs={self.epochs} | optimizer={optimizer_name} | lr={lr} | "
            f"scheduler={scheduler_name} | n_basins={n_basins}"
        )
        sys.stdout.flush()
        sys.stderr.flush()

        self._train_start_time = time.perf_counter()
        self._final_loss = 0.0

        self._emit_progress(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs"
        )
        sys.stdout.flush()
        sys.stderr.flush()

        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(
                epoch,
                n_samples,
                n_minibatch,
                n_timesteps,
            )

        # 训练结束摘要
        total_time = time.perf_counter() - self._train_start_time
        self._emit_progress(
            f"[Train End] total_time={total_time:.1f}s | best_epoch=N/A | final_loss={self._final_loss:.4f}"
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def train_one_epoch(
        self, epoch, n_samples, n_minibatch, n_timesteps
    ) -> None:
        """Train model for one epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        n_samples
            Number of samples in the training dataset.
        n_minibatch
            Number of minibatches in the training dataset.
        n_timesteps
            Number of timesteps in the training dataset.
        """
        start_time = time.perf_counter()
        prog_str = f"Epoch {epoch}/{self.epochs}"

        self.current_epoch = epoch
        self.total_loss = 0.0
        self.model.loss_dict = {key: 0.0 for key in self.model.loss_dict}

        # 获取实际的 num_start：仅从模型属性读取，否则自然报错
        model_name = self.config["delta_model"]["phy_model"]["model"]
        _model_name = model_name[0] if isinstance(model_name, list) else model_name
        _nn = getattr(self.model.model_dict.get(_model_name, None), "nn_model", None)
        num_start = _nn.num_start

        # Iterate through epoch in minibatches.
        # 彻底禁用 tqdm 进度条，避免控制字符污染日志文件
        for mb in range(1, n_minibatch + 1):
            self.current_batch = mb

            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            # ============================================================
            # [核心修改] 仅在 Multi-Start 时对 Target 进行复制
            # ============================================================
            if num_start > 1:
                dataset_sample["target"] = dataset_sample[
                    "target"
                ].repeat_interleave(num_start, dim=1)
            # ============================================================

            # Forward pass through model.
            _ = self.model(dataset_sample)

            # Loss 计算
            loss = self.model.calc_loss(dataset_sample)

            # ============================================================
            # [修改 1] Loss NaN 保护：跳过当前 Batch 而不是报错
            # ============================================================
            if torch.isnan(loss) or torch.isinf(loss):
                log.debug(
                    f"[Warning] Batch {mb}: Loss is NaN/Inf. Skipping this batch."
                )
                self.optimizer.zero_grad()
                continue  # 直接跳过，不进行 backward，保护模型不崩

            # 反向传播
            loss.backward()

            # ============================================================
            # [修改 2] 梯度 NaN 保护 (即你想要的 nan_to_num)
            # ============================================================
            # 遍历所有参数的梯度，将 NaN 替换为 0，防止“一颗老鼠屎坏了一锅粥”
            # 这样坏掉的参数（梯度为NaN）不会更新，好的参数（梯度正常）继续优化
            for param in self.model.get_parameters():
                if param.grad is not None:
                    # 原地修改梯度：NaN -> 0, 正无穷 -> 1, 负无穷 -> -1 (数值可根据需要调整)
                    torch.nan_to_num_(
                        param.grad, nan=0.0, posinf=1.0, neginf=-1.0
                    )

            # 梯度裁剪 (保持原有逻辑)
            torch.nn.utils.clip_grad_norm_(
                self.model.get_parameters(), max_norm=1.0
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.total_loss += loss.item()

            # CosineAnnealingWarmRestarts 需要 per-batch step 以获得平滑余弦曲线
            if self.use_scheduler and isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            ):
                self.scheduler.step(epoch - 1 + mb / n_minibatch)

            # 移除了 per-batch 的 verbose 日志输出，避免日志冗余
            # 统计信息已由 _log_epoch_stats 每个 epoch 输出一次

        if self.use_scheduler and not isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        ):
            self.scheduler.step()

        # 移除了 per-epoch 的 verbose 日志输出，避免日志冗余
        # 统计信息已由 _log_epoch_stats 每个 epoch 输出一次

        # 记录 final_loss 供 Train End 摘要使用
        self._final_loss = self.total_loss / max(n_minibatch, 1)

        self._log_epoch_stats(
            epoch, self.model.loss_dict, n_minibatch, start_time
        )

        # Save model and trainer states.
        if epoch % self.config["train"]["save_epoch"] == 0:
            self.model.save_model(epoch)
            save_train_state(
                self.config,
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )

    def _evaluate_dataset(
        self,
        dataset: dict,
        out_path,
        start_time: str,
        end_time: str,
    ) -> None:
        """Evaluate a single dataset and save results to out_path.

        Parameters
        ----------
        dataset
            Dataset dictionary to evaluate.
        out_path
            Output directory path for saving results.
        start_time
            Start time string for logging.
        end_time
            End time string for logging.
        """
        model_name = self.config["delta_model"]["phy_model"]["model"]
        _model_name = model_name[0] if isinstance(model_name, list) else model_name
        _nn = getattr(self.model.model_dict.get(_model_name, None), "nn_model", None)
        num_start = _nn.num_start

        observations = dataset["target"]
        if num_start > 1:
            observations = observations.repeat_interleave(num_start, dim=1)

        n_samples = dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Evaluating {start_time} ~ {end_time}: {len(batch_start)} batches")
        batch_predictions = self._forward_loop(dataset, batch_start, batch_end)

        # Temporarily override out_path in config for saving
        orig_out_path = self.config["out_path"]
        self.config["out_path"] = out_path
        out_path.mkdir(parents=True, exist_ok=True)

        log.info("Saving model outputs + Calculating metrics")
        if self.config.get("save_output", False):
            save_outputsv2(
                self.config, batch_predictions, observations, create_dirs=True
            )
        self.calc_metrics(batch_predictions, observations)

        self.config["out_path"] = orig_out_path

        # Free memory after each dataset evaluation
        del batch_predictions, observations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self) -> None:
        """Run model evaluation on both train and eval datasets."""
        self.is_in_train = False

        base_outpath = Path(self.config["out_path"]).parents[0]
        test_epoch = self.config["test"].get("test_epoch", "")

        datasets_to_eval = []

        if self.train_dataset is not None:
            train_start = self.config["train"].get("start_time", "1989/01/01")
            train_end = self.config["train"].get("end_time", "1998/12/31")
            # Format: train{YYYY}-{YYYY}_Ep{epoch}
            s_year = train_start.split("/")[0]
            e_year = train_end.split("/")[0]
            folder = base_outpath / f"train{s_year}-{e_year}_Ep{test_epoch}"
            datasets_to_eval.append((self.train_dataset, folder, train_start, train_end))

        if self.eval_dataset is not None:
            eval_start = self.config["test"].get("start_time", "1999/01/01")
            eval_end = self.config["test"].get("end_time", "2009/12/31")
            s_year = eval_start.split("/")[0]
            e_year = eval_end.split("/")[0]
            folder = base_outpath / f"test{s_year}-{e_year}_Ep{test_epoch}"
            datasets_to_eval.append((self.eval_dataset, folder, eval_start, eval_end))

        for dataset, out_path, start_time, end_time in datasets_to_eval:
            self._evaluate_dataset(dataset, out_path, start_time, end_time)
            print(f"Metrics and predictions saved to {out_path}")

    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch
        n_samples = self.dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(
            0, n_samples, self.config["simulation"]["batch_size"]
        )
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.dataset, batch_start, batch_end
        )

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary.

        Parameters
        ----------
        batch_list
            List of dictionaries from each forward batch containing inputs and
            model predictions.
        target_key
            Key to extract from each batch dictionary.
        """
        data = {}
        try:
            if target_key:
                return torch.cat(
                    [x[target_key] for x in batch_list], dim=1
                ).numpy()

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
        """Forward loop used in model evaluation and inference.

        Parameters
        ----------
        data
            Dictionary containing model input data.
        batch_start
            Start indices for each batch.
        batch_end
            End indices for each batch.
        """
        # Track predictions accross batches
        batch_predictions = []
        # Save the batch predictions
        model_name = self.config["delta_model"]["phy_model"]["model"][0]
        # 彻底禁用 tqdm 进度条，避免控制字符污染日志文件
        for i in range(len(batch_start)):
            self.current_batch = i

            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[i],
                batch_end[i],
            )
            if self.config["test"]["split_dataset"]:
                total_time_steps = dataset_sample["x_phy"].shape[0]
                # split to 730
                prediction_time_chunks = []
                prediction_length = self.config["delta_model"]["rho"]
                warmup_length = self.config["delta_model"]["phy_model"][
                    "warm_up"
                ]
                # subtime_length = prediction_length + warmup_length
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
                        key: tensor[warmup_length:, ...].cpu().detach()
                        if tensor.shape[0] > warmup_length
                        else tensor.cpu().detach()
                        for key, tensor in prediction_window[model_name].items()
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
        """Calculate and save model performance metrics.

        Parameters
        ----------
        batch_predictions
            List of dictionaries containing model predictions.
        observations
            Target variable observation data.
        """
        target_name = self.config["train"]["target"][0]
        predictions = self._batch_data(batch_predictions, target_name)
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)

        # Remove warm-up data
        target = target[self.config["delta_model"]["phy_model"]["warm_up"] :, :]
        target = target[: len(predictions), :]
        # Compute metrics
        metrics_to_compute = self.config["test"].get("metrics", None)
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
            metrics_to_compute,
        )

        # Save all metrics and aggregated statistics.
        metrics.dump_metrics(self.config["out_path"])

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """每个 epoch 结束时输出一行简洁日志，频率由 log_interval 控制。"""
        log_interval = self.config["train"].get("log_interval", 1)  # 默认每个epoch都打印
        if epoch % log_interval != 0:
            return

        # 当前学习率
        lr = self.optimizer.param_groups[0]["lr"]

        # 优先使用当前 epoch 内部累计的平均 loss，避免读取跨 epoch 累积值。
        avg_loss = self._final_loss

        elapsed = time.perf_counter() - start_time
        if torch.cuda.is_available() and str(self.config["device"]).startswith("cuda"):
            mem_mb = int(
                torch.cuda.memory_reserved(device=self.config["device"]) * 0.000001
            )
        else:
            mem_mb = 0

        # 检测 Warm Restart（CosineAnnealingWarmRestarts 周期结束）
        warm_restart_tag = ""
        if self.use_scheduler and isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        ):
            T_0 = self.scheduler.T_0
            T_mult = self.scheduler.T_mult
            # 当 epoch 是重启点时标记
            t = T_0
            while t <= epoch:
                if epoch == t:
                    warm_restart_tag = "  ← Warm Restart"
                    break
                t += T_0 * (T_mult ** (t // T_0))

        self._emit_progress(
            f"[Epoch {epoch:>4}/{self.epochs}] loss={avg_loss:.4f} | "
            f"lr={lr:.2e} | time={elapsed:.1f}s | mem={mem_mb}MB{warm_restart_tag}"
        )
        # 强制刷新输出到文件
        sys.stdout.flush()
        sys.stderr.flush()
