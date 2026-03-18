import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid, create_dl_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_outputsv2, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class NewTrainer(BaseTrainer):
    """Trainer based on Trainer, with evaluate() supporting both train and eval datasets.

    Changes from Trainer:
    - evaluate() evaluates both train_dataset and eval_dataset (like CalTrainer)
    - Uses save_outputsv2 for output saving
    - No tqdm in train loop (cleaner log output)
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
        elif "test" in config["mode"]:
            self.load_test_states()

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
        name = self.config["delta_model"]["nn_model"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
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
                **self.config["delta_model"]["nn_model"]["lr_scheduler_params"],
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        path = self.config["model_path"]
        for file in os.listdir(path):
            if "train_state" in file:
                checkpoint = torch.load(
                    os.path.join(path, file), map_location=self.config["device"]
                )
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.model.load_model(epoch=checkpoint["epoch"])
                self.start_epoch = checkpoint["epoch"] + 1
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                torch.set_rng_state(checkpoint["random_state"].cpu().byte())
                if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
                    torch.cuda.set_rng_state_all(
                        checkpoint["cuda_random_state"].cpu().byte()
                    )
                log.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                return
            else:
                self.start_epoch = 1

    def load_test_states(self) -> None:
        path = self.config["model_path"]
        test_epoch = self.config["test"].get("test_epoch", None)
        if test_epoch is None:
            raise ValueError("'test_epoch' must be set in config['test'].")
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
        if log.hasHandlers() and log.isEnabledFor(logging.INFO):
            log.info(message)
        else:
            print(message, flush=True)

    def train(self) -> None:
        self.is_in_train = True

        if self.config.get("data_sampler") == "DlSampler":
            n_samples, n_minibatch, n_timesteps = create_dl_training_grid(
                self.train_dataset["xc_nn_norm"], self.config
            )
        else:
            n_samples, n_minibatch, n_timesteps = create_training_grid(
                self.train_dataset["xc_nn_norm"], self.config
            )

        self._train_start_time = time.perf_counter()
        self._final_loss = 0.0
        self._emit_progress(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs"
        )
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

        total_time = time.perf_counter() - self._train_start_time
        self._emit_progress(
            f"[Train End] total_time={total_time:.1f}s | final_loss={self._final_loss:.4f}"
        )
        sys.stdout.flush()

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        start_time = time.perf_counter()
        self.current_epoch = epoch
        self.total_loss = 0.0
        self.model.loss_dict = {key: 0.0 for key in self.model.loss_dict}

        for mb in range(1, n_minibatch + 1):
            self.current_batch = mb

            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset, n_samples, n_timesteps
            )

            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)

            if torch.isnan(loss) or torch.isinf(loss):
                log.debug(f"[Warning] Batch {mb}: Loss is NaN/Inf. Skipping.")
                self.optimizer.zero_grad()
                continue

            self._emit_progress(f"  [Epoch {epoch} Batch {mb}/{n_minibatch}] loss={loss.item():.4f}")
            loss.backward()

            for param in self.model.get_parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)

            torch.nn.utils.clip_grad_norm_(self.model.get_parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss += loss.item()

            if self.use_scheduler and isinstance(
                self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            ):
                self.scheduler.step(epoch - 1 + mb / n_minibatch)

        if self.use_scheduler and not isinstance(
            self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            self.scheduler.step()

        self._final_loss = self.total_loss / max(n_minibatch, 1)
        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

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
        out_path: Path,
        start_time: str,
        end_time: str,
    ) -> None:
        observations = dataset["target"]
        n_samples = dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Evaluating {start_time} ~ {end_time}: {len(batch_start)} batches")
        batch_predictions = self._forward_loop(dataset, batch_start, batch_end)

        orig_out_path = self.config["out_path"]
        self.config["out_path"] = str(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        log.info("Saving model outputs + Calculating metrics")
        if self.config.get("save_output", False):
            save_outputsv2(
                self.config, batch_predictions, observations, create_dirs=True
            )
        self.calc_metrics(batch_predictions, observations)

        self.config["out_path"] = orig_out_path

        del batch_predictions, observations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self) -> None:
        """Evaluate on both train_dataset and eval_dataset if available."""
        self.is_in_train = False

        base_outpath = Path(self.config["out_path"]).parents[0]
        test_epoch = self.config["test"].get("test_epoch", "")

        datasets_to_eval = []

        if self.train_dataset is not None:
            train_start = self.config["train"].get("start_time", "1989/01/01")
            train_end = self.config["train"].get("end_time", "1998/12/31")
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
        self.is_in_train = False

        n_samples = self.dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["simulation"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)
        return self.predictions

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ):
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()
            for key in batch_list[0].keys():
                data[key] = torch.cat([d[key] for d in batch_list], dim=1).cpu().numpy()
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
        model_name = self.config["delta_model"]["phy_model"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]

        for i in range(len(batch_start)):
            self.current_batch = i
            dataset_sample = self.sampler.get_validation_sample(
                data, batch_start[i], batch_end[i]
            )
            if self.config["test"]["split_dataset"]:
                total_time_steps = dataset_sample["x_phy"].shape[0]
                prediction_length = self.config["delta_model"]["rho"]
                warmup_length = self.config["delta_model"]["phy_model"]["warm_up"]
                time_starts = range(
                    0,
                    total_time_steps - prediction_length - warmup_length + 1,
                    prediction_length,
                )
                prediction_time_chunks = []
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
                    for key, tensor in prediction[model_name].items()
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

        target = target[self.config["delta_model"]["phy_model"]["warm_up"]:, :]
        target = target[: len(predictions), :]

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
        log_interval = self.config["train"].get("log_interval", 1)
        if epoch % log_interval != 0:
            return

        lr = self.optimizer.param_groups[0]["lr"]
        avg_loss = self._final_loss
        elapsed = time.perf_counter() - start_time

        if torch.cuda.is_available() and str(self.config["device"]).startswith("cuda"):
            mem_mb = int(
                torch.cuda.memory_reserved(device=self.config["device"]) * 0.000001
            )
        else:
            mem_mb = 0

        self._emit_progress(
            f"[Epoch {epoch:>4}/{self.epochs}] loss={avg_loss:.4f} | "
            f"lr={lr:.2e} | time={elapsed:.1f}s | mem={mem_mb}MB"
        )
        sys.stdout.flush()
