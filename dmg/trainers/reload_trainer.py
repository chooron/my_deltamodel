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


# try:
#     from ray import tune
#     from ray.air import Checkpoint
# except ImportError:
#     log.warning('Ray Tune is not installed or is misconfigured. Tuning will be disabled.')


class ReloadTrainer(BaseTrainer):
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

            log.info("Initializing experiment")
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
        name = self.config["delta_model"]["train"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
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
                **self.config["delta_model"]["train"]["lr_scheduler_params"],
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_pretrained_submodel(self) -> None:
        """
        从指定路径加载预训练的子模型权重（如MultiHeadNet）。
        pt文件使用phy_model名称命名，但加载到nn_model中。
        只加载模型权重，不加载optimizer和scheduler状态。
        """
        if "pretrained_model" not in self.config:
            return
        
        pretrained_config = self.config["pretrained_model"]
        if not pretrained_config.get("enabled", False):
            return
        
        model_path = pretrained_config.get("path")
        phy_model_name = pretrained_config.get("phy_model", "BlendHydroV0")
        nn_model_name = pretrained_config.get("nn_model", "MultiHeadNet")
        epoch = pretrained_config.get("epoch", 0)
        
        if not model_path or not os.path.exists(model_path):
            log.warning(f"Pretrained model path not found: {model_path}")
            return
        
        # 使用phy_model名称构造模型文件路径
        if epoch > 0:
            model_file = os.path.join(model_path, f"d{phy_model_name}_Ep{epoch}.pt")
        else:
            # 尝试查找最新的epoch
            model_files = [f for f in os.listdir(model_path) if f.startswith(f"d{phy_model_name}_Ep") and f.endswith(".pt")]
            if not model_files:
                log.warning(f"No pretrained model found for {phy_model_name} in {model_path}")
                return
            # 提取epoch号并找到最大的
            epochs = [int(f.split("Ep")[1].split(".")[0]) for f in model_files]
            epoch = max(epochs)
            model_file = os.path.join(model_path, f"d{phy_model_name}_Ep{epoch}.pt")
        
        if not os.path.exists(model_file):
            log.warning(f"Pretrained model file not found: {model_file}")
            return
        
        try:
            # 加载预训练权重
            state_dict = torch.load(model_file, map_location=self.config["device"])
            
            # 从state_dict中提取nn_model的权重
            # state_dict可能包含完整的DplModel状态，需要提取nn_model部分
            nn_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('nn_model.'):
                    # 移除'nn_model.'前缀
                    new_key = key[len('nn_model.'):]
                    nn_state_dict[new_key] = value
            
            # 如果没有找到nn_model前缀，尝试直接使用整个state_dict
            if not nn_state_dict:
                nn_state_dict = state_dict
            
            # 加载到当前模型的nn_model中
            if hasattr(self.model, 'model_dict'):
                loaded = False
                for name, submodel in self.model.model_dict.items():
                    if hasattr(submodel, 'nn_model'):
                        # 加载到DplModel的nn_model中
                        submodel.nn_model.load_state_dict(nn_state_dict, strict=False)
                        log.info(f"Loaded pretrained {nn_model_name} weights from {phy_model_name} epoch {epoch} into {name}.nn_model")
                        loaded = True
                    elif hasattr(submodel, 'param_net'):
                        # 可能是其他参数网络
                        submodel.param_net.load_state_dict(nn_state_dict, strict=False)
                        log.info(f"Loaded pretrained {nn_model_name} weights from {phy_model_name} epoch {epoch} into {name}.param_net")
                        loaded = True
                
                if not loaded:
                    log.warning(f"Could not find nn_model or param_net to load weights into")
            
            log.info(f"Successfully loaded pretrained {nn_model_name} from {phy_model_name}_Ep{epoch}.pt")
            
        except Exception as e:
            log.error(f"Error loading pretrained model from {model_file}: {e}")
            raise

    def load_states(self) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint to resume
        training if a checkpoint file exists.

        如果配置了pretrained_model，会先加载预训练的子模型权重。
        """
        # 首先尝试加载预训练的子模型（如果配置了的话）
        self.load_pretrained_submodel()

        path = self.config["model_path"]
        if not os.path.exists(path):
            log.warning(f"Model path does not exist: {path}")
            self.start_epoch = 1
            return

        for file in os.listdir(path):
            # Check for state checkpoint: looks like `train_state_epoch_XX.pt`.
            if "train_state" in file:
                checkpoint = torch.load(os.path.join(path, file), map_location=self.config["device"])

                # Restore optimizer states
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
                # Restore model (如果没有加载预训练模型，则正常恢复训练)
                # 注意：如果已加载预训练模型，这里会覆盖预训练的权重
                if not self.config.get("pretrained_model", {}).get("enabled", False):
                    self.model.load_model(epoch=checkpoint["epoch"])

                self.start_epoch = checkpoint["epoch"] + 1

                if self.scheduler:
                    self.scheduler.load_state_dict(
                        checkpoint["scheduler_state_dict"]
                    )

                # Restore random states (with error handling for compatibility)
                try:
                    random_state = checkpoint.get("random_state")
                    if random_state is not None:
                        # Ensure it's a ByteTensor
                        if not isinstance(random_state, torch.ByteTensor):
                            if isinstance(random_state, torch.Tensor):
                                random_state = random_state.byte()
                            else:
                                log.warning(f"Random state has unexpected type: {type(random_state)}, skipping")
                                random_state = None

                        if random_state is not None:
                            torch.set_rng_state(random_state)

                    # Restore CUDA random states (note: key name is 'cuda_state' not 'cuda_random_state')
                    if torch.cuda.is_available():
                        cuda_state = checkpoint.get("cuda_state") or checkpoint.get("cuda_random_state")
                        if cuda_state is not None:
                            torch.cuda.set_rng_state_all(cuda_state)
                except Exception as e:
                    log.warning(f"Failed to restore random states: {e}. Continuing with current random state.")

                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                return

        # 如果没有找到checkpoint，从epoch 1开始
        self.start_epoch = 1

    def _calc_weighted_multi_streamflow_loss(
        self,
        model_output: dict,
        dataset_sample: dict,
    ) -> torch.Tensor:
        """计算加权的多模型streamflow损失。

        主损失来自 'streamflow'，辅助损失来自所有 '*_streamflow' 输出。
        总损失 = 主损失 + auxiliary_loss_weight * sum(辅助损失)

        Parameters
        ----------
        model_output
            模型输出字典，包含 'streamflow' 和多个 {model_name}_streamflow
        dataset_sample
            数据样本，包含target

        Returns
        -------
        torch.Tensor
            加权后的总损失值
        """
        target = dataset_sample["target"]

        # 获取辅助损失权重系数（默认为0.1）
        auxiliary_loss_weight = self.config.get("train", {}).get("auxiliary_loss_weight", 0.1)

        # 1. 计算主损失 (streamflow)
        if "streamflow" not in model_output:
            raise ValueError("Model output must contain 'streamflow' key")

        main_pred = model_output["streamflow"]
        main_loss = self.loss_func(
            main_pred,
            target,
            sample_ids=dataset_sample.get('batch_sample'),
        )

        # 2. 计算辅助损失 (所有 *_streamflow)
        auxiliary_losses = []
        streamflow_keys = [k for k in model_output.keys()
                          if k.endswith("_streamflow") and k != "streamflow"]

        for key in streamflow_keys:
            pred = model_output[key]
            loss_i = self.loss_func(
                pred,
                target,
                sample_ids=dataset_sample.get('batch_sample'),
            )
            auxiliary_losses.append(loss_i)

        # 3. 计算总损失
        if auxiliary_losses:
            auxiliary_loss_sum = torch.stack(auxiliary_losses).sum()
            total_loss = main_loss + auxiliary_loss_weight * auxiliary_loss_sum
        else:
            total_loss = main_loss
            auxiliary_loss_sum = torch.tensor(0.0, device=main_loss.device)

        # 4. 更新loss_dict用于记录
        if hasattr(self.model, "loss_dict"):
            self.model.loss_dict["main_loss"] = main_loss.item()
            self.model.loss_dict["auxiliary_loss_sum"] = auxiliary_loss_sum.item()
            self.model.loss_dict["total_loss"] = total_loss.item()

            # 记录每个辅助损失
            for i, key in enumerate(streamflow_keys):
                self.model.loss_dict[f"{key}_loss"] = auxiliary_losses[i].item()

        return total_loss

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True

        # Setup a training grid (number of samples, minibatches, and timesteps)
        # 根据 data_sampler 类型选择合适的训练网格计算函数
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset["xc_nn_norm"],
            self.config,
        )

        log.info(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs"
        )

        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(
                epoch,
                n_samples,
                n_minibatch,
                n_timesteps,
            )

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

        # Iterate through epoch in minibatches.
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

            # Forward pass through model.
            model_output = self.model(dataset_sample)

            # 计算加权的多模型streamflow损失
            loss = self._calc_weighted_multi_streamflow_loss(model_output, dataset_sample)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.get_parameters(), max_norm=1.0
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.total_loss += loss.item()

            if self.verbose:
                tqdm.tqdm.write(
                    f"Epoch {epoch}, batch {mb} | loss: {loss.item()}"
                )

        if self.use_scheduler:
            self.scheduler.step()

        if self.verbose:
            log.info(f"\n ---- \n Epoch {epoch} total loss: {self.total_loss}")
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

    def evaluate(self) -> None:
        """Run model evaluation and return both metrics and model outputs."""
        self.is_in_train = False

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.eval_dataset["target"]

        # Get start and end indices for each batch
        n_samples = self.eval_dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Validating Model: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.eval_dataset, batch_start, batch_end
        )

        # Save predictions and calculate metrics
        log.info("Saving model outputs + Calculating metrics")
        if self.config.get("save_output", False):
            save_outputs(
                self.config, batch_predictions, observations, create_dirs=True
            )
        self.predictions = self._batch_data(batch_predictions)

        # Calculate metrics
        self.calc_metrics(batch_predictions, observations)

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
        for i in tqdm.tqdm(
            range(len(batch_start)),
            desc="Forwarding",
            leave=False,
            dynamic_ncols=True,
        ):
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
                    for key, tensor in prediction[model_name].items()
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
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
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
        """Log statistics after each epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        loss_dict
            Dictionary containing loss values.
        n_minibatch
            Number of minibatches.
        start_time
            Start time of the epoch.
        """
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
