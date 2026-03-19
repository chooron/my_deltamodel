"""
TwoStageTrainer - DiffBlendV2 两阶段训练器

Stage 1 (预训练): 固定均匀权重，训练参数网络学习物理参数
Stage 2 (权重精调): 差异化学习率，Gumbel-Softmax 温度退火学习过程权重

继承 BaseTrainer，复用项目的数据加载和模型管理基础设施。
"""

import logging
import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid, create_dl_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_outputsv2, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


# ===================================================================
# Tau 温度调度器
# ===================================================================

class TauScheduler:
    """Gumbel-Softmax 温度退火调度器。

    在 warmup 阶段维持高温（鼓励探索），之后指数衰减至低温（促进收敛）。

    参数:
        tau_start: 初始高温（探索阶段）
        tau_end: 最终低温（收敛阶段）
        warmup_epochs: 维持高温的 epoch 数
        total_epochs: stage2 总 epoch 数
    """

    def __init__(
        self,
        tau_start: float = 5.0,
        tau_end: float = 0.3,
        warmup_epochs: int = 20,
        total_epochs: int = 80,
    ):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # 预计算衰减率: tau_end = tau_start * exp(-decay_rate * decay_epochs)
        decay_epochs = max(total_epochs - warmup_epochs, 1)
        self.decay_rate = -math.log(tau_end / tau_start) / decay_epochs

    def get_tau(self, epoch: int) -> float:
        """获取当前 epoch 的温度值。"""
        if epoch < self.warmup_epochs:
            return self.tau_start
        # 指数衰减
        t = epoch - self.warmup_epochs
        return self.tau_start * math.exp(-self.decay_rate * t)


# ===================================================================
# 权重熵监控
# ===================================================================

# 过程名称与选项数
PROCESS_OPTIONS = {
    "rainsnow": 3, "snowbal": 3, "infiltration": 3,
    "evaporation": 3, "quickflow": 3, "baseflow": 2,
}


def log_weight_entropy(pred_dict: Dict[str, torch.Tensor], epoch: int) -> None:
    """打印各过程权重的均值分布和熵值，用于监控权重坍塌。

    对每个过程，收集所有选项的权重，计算:
    - 跨流域平均的权重分布
    - 熵值 H = -sum(w * log(w + eps))，与最大熵 log(n_opt) 对比
    """
    for proc, n_opt in PROCESS_OPTIONS.items():
        # 收集该过程所有选项的权重: w_{proc}_{i} -> [T, B]
        ws = []
        for i in range(n_opt):
            key = f"w_{proc}_{i}"
            if key in pred_dict:
                # 取第一个时间步（权重在时间维上是常量）
                ws.append(pred_dict[key][0].mean().item())
        if not ws:
            continue

        # 归一化（防止数值误差导致不归一）
        ws_arr = np.array(ws)
        ws_arr = ws_arr / (ws_arr.sum() + 1e-8)

        # 熵计算
        entropy = -np.sum(ws_arr * np.log(ws_arr + 1e-8))
        max_entropy = np.log(n_opt)
        pct = entropy / max_entropy * 100 if max_entropy > 0 else 0

        w_str = "[" + ", ".join(f"{w:.2f}" for w in ws_arr) + "]"
        pad = max(14 - len(proc), 0)
        log.info(
            f"[Epoch {epoch:3d}] {proc}:{' ' * pad} "
            f"w={w_str} H={entropy:.3f}/{max_entropy:.3f}({pct:.0f}%)"
        )


# ===================================================================
# 两阶段训练器
# ===================================================================

class TwoStageTrainer(BaseTrainer):
    """DiffBlendV2 两阶段训练器。

    Stage 1: 预训练物理参数（均匀权重，冻结 weight_logits）
    Stage 2: 差异化学习率精调权重（Tau 退火 + 熵监控）

    通过 config['two_stage'] 配置所有超参数。
    """

    def __init__(
        self,
        config: dict[str, Any],
        model = None,
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
        self.sampler = import_data_sampler(config["data_sampler"])(config)
        self.is_in_train = False

        # 两阶段配置
        ts: Any = config.get("two_stage", {})
        self.verbose = ts.get("verbose", True)
        self.stage1_epochs = ts.get("stage1_epochs", 50)
        self.stage2_epochs = ts.get("stage2_epochs", 80)
        self.stage1_lr = ts.get("stage1_lr", 1e-3)
        self.stage2_param_lr = ts.get("stage2_param_lr", 1e-5)
        self.stage2_weight_lr = ts.get("stage2_weight_lr", 1e-3)
        self.balance_loss_weight = ts.get("balance_loss_weight", 0.01)
        self.log_interval = ts.get("log_interval", 5)
        self.checkpoint_dir = config.get("model_path", "./checkpoints")

        # Tau 调度器
        self.tau_scheduler = TauScheduler(
            tau_start=ts.get("tau_start", 5.0),
            tau_end=ts.get("tau_end", 0.3),
            warmup_epochs=ts.get("stage2_warmup_epochs", 20),
            total_epochs=self.stage2_epochs,
        )

        if "train" in config["mode"]:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            self.epochs = self.stage1_epochs + self.stage2_epochs

            # 损失函数
            self.loss_func = loss_func or load_criterion(
                self.train_dataset["target"],
                config["loss_function"],
                device=config["device"],
            )
            self.model.loss_func = self.loss_func

        elif "test" in config["mode"]:
            self.load_test_states()

    # ---------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------

    def _get_phy_model(self) -> nn.Module:
        """获取物理模型 (DiffBlendV2) 的引用。"""
        model_name = self.config["delta_model"]["phy_model"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]
        return self.model.model_dict[model_name].phy_model

    def _get_nn_model(self) -> nn.Module:
        """获取参数预测网络的引用。"""
        model_name = self.config["delta_model"]["phy_model"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]
        return self.model.model_dict[model_name].nn_model

    def init_optimizer(self) -> torch.optim.Optimizer:
        """根据当前 stage 初始化优化器。"""
        raise NotImplementedError("Use _init_stage1_optimizer / _init_stage2_optimizer")

    def _init_stage1_optimizer(self) -> torch.optim.Optimizer:
        """Stage 1 优化器: 冻结 weight head，只训练 backbone、phy_head、rout_head。"""
        nn_model = self._get_nn_model()

        # 冻结 weight head
        for param in nn_model.heads["process_weight_logits"].parameters():
            param.requires_grad_(False)

        trainable = [p for p in nn_model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.stage1_lr)

    def _init_stage2_optimizer(self) -> torch.optim.Optimizer:
        """Stage 2 优化器: 解冻 weight head，差异化学习率精调。"""
        nn_model = self._get_nn_model()

        # 解冻 weight head
        for param in nn_model.heads["process_weight_logits"].parameters():
            param.requires_grad_(True)

        weight_head_params = list(
            nn_model.heads["process_weight_logits"].parameters()
        )
        other_params = [
            p for n, p in nn_model.named_parameters()
            if "process_weight_logits" not in n
        ]
        return torch.optim.Adam([
            {"params": other_params,       "lr": self.stage2_param_lr},
            {"params": weight_head_params, "lr": self.stage2_weight_lr},
        ])

    def _save_checkpoint(self, epoch: int, stage: int, loss: float) -> None:
        """保存 checkpoint。"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        phy_model = self._get_phy_model()
        nn_model = self._get_nn_model()
        path = os.path.join(
            self.checkpoint_dir, f"two_stage_s{stage}_ep{epoch}.pt"
        )
        torch.save({
            "model": phy_model.state_dict(),
            "param_network": nn_model.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "loss": loss,
        }, path)
        log.info(f"Checkpoint saved: {path}")

    def load_test_states(self) -> None:
        """加载测试状态。"""
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

    # ---------------------------------------------------------------
    # 单 epoch 训练
    # ---------------------------------------------------------------

    def _train_one_epoch(
        self,
        epoch: int,
        stage: int,
        n_samples: int,
        n_minibatch: int,
        n_timesteps: int,
    ) -> Dict[str, float]:
        """训练一个 epoch，返回平均损失字典。"""
        start_time = time.perf_counter()

        total_loss = 0.0
        valid_batches = 0
        verbose_log_interval = self.config.get("verbose_log_interval", 10)

        prog_str = f"Stage{stage} Epoch {epoch}"
        for mb in tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=prog_str, leave=False, dynamic_ncols=True,
        ):
            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset, n_samples, n_timesteps,
            )

            # 前向传播
            _ = self.model(dataset_sample)

            # 损失计算
            loss = self.model.calc_loss(dataset_sample)

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            loss.backward()

            # 梯度 NaN 保护
            for param in self.model.get_parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)

            torch.nn.utils.clip_grad_norm_(self.model.get_parameters(), max_norm=1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            valid_batches += 1

            if self.verbose and valid_batches % verbose_log_interval == 0:
                avg_so_far = total_loss / valid_batches
                print(
                    f"[Stage{stage}] Epoch {epoch} Batch {mb}/{n_minibatch}: "
                    f"loss={loss.item():.6f}  avg={avg_so_far:.6f}"
                )

        avg_loss = total_loss / max(valid_batches, 1)
        elapsed = time.perf_counter() - start_time

        if torch.cuda.is_available():
            mem = int(torch.cuda.memory_reserved(device=self.config["device"]) * 1e-6)
        else:
            mem = 0

        log.info(
            f"[Stage{stage}] Epoch {epoch}: loss={avg_loss:.6f} "
            f"({elapsed:.1f}s, {mem}Mb GPU)"
        )

        return {"total": avg_loss}

    # ---------------------------------------------------------------
    # 主训练入口
    # ---------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """两阶段训练主函数。

        返回训练历史字典，包含每个 epoch 的损失记录。
        """
        self.is_in_train = True
        history: Dict[str, list] = {
            "stage1_loss": [],
            "stage2_loss": [],
        }

        # 构建训练网格
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset["xc_nn_norm"], self.config,
        )

        phy_model = self._get_phy_model()
        save_epoch = self.config["train"].get("save_epoch", 5)

        # ==================== Stage 1: 预训练物理参数 ====================
        log.info(f"===== Stage 1: 预训练物理参数 ({self.stage1_epochs} epochs) =====")
        phy_model.stage = 1
        self.optimizer = self._init_stage1_optimizer()

        for epoch in range(1, self.stage1_epochs + 1):
            loss_dict = self._train_one_epoch(
                epoch, stage=1,
                n_samples=n_samples,
                n_minibatch=n_minibatch,
                n_timesteps=n_timesteps,
            )
            history["stage1_loss"].append(loss_dict["total"])

            if epoch % save_epoch == 0:
                self._save_checkpoint(epoch, stage=1, loss=loss_dict["total"])
                self.model.save_model(epoch)

        # 保存 Stage 1 最终 checkpoint
        self._save_checkpoint(self.stage1_epochs, stage=1, loss=history["stage1_loss"][-1])

        # ==================== Stage 2: 权重精调 ====================
        log.info(f"===== Stage 2: 权重精调 ({self.stage2_epochs} epochs) =====")
        phy_model.stage = 2
        self.optimizer = self._init_stage2_optimizer()

        for epoch in range(1, self.stage2_epochs + 1):
            # 更新 Tau 温度
            tau = self.tau_scheduler.get_tau(epoch)
            phy_model.tau = tau

            loss_dict = self._train_one_epoch(
                epoch, stage=2,
                n_samples=n_samples,
                n_minibatch=n_minibatch,
                n_timesteps=n_timesteps,
            )
            history["stage2_loss"].append(loss_dict["total"])

            # 权重熵监控
            if epoch % self.log_interval == 0:
                log.info(f"[Stage2] tau={tau:.3f}")
                # 做一次前向传播获取权重分布
                with torch.no_grad():
                    sample = self.sampler.get_training_sample(
                        self.train_dataset, n_samples, n_timesteps,
                    )
                    pred_dict = self.model(sample)
                    # pred_dict 来自 ModelHandler，需要取第一个模型的输出
                    model_name = self.config["delta_model"]["phy_model"]["model"]
                    if isinstance(model_name, list):
                        model_name = model_name[0]
                    output = self.model.output_dict.get(model_name, pred_dict)
                    log_weight_entropy(output, epoch)

            global_epoch = self.stage1_epochs + epoch
            if epoch % save_epoch == 0:
                self._save_checkpoint(epoch, stage=2, loss=loss_dict["total"])
                self.model.save_model(global_epoch)

        log.info("===== 两阶段训练完成 =====")
        return history

    # ---------------------------------------------------------------
    # 评估 / 推理
    # ---------------------------------------------------------------

    def evaluate(self) -> None:
        """模型评估。"""
        self.is_in_train = False

        batch_predictions = []
        observations = self.eval_dataset["target"]

        n_samples = self.eval_dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Evaluating: {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.eval_dataset, batch_start, batch_end
        )

        if self.config.get("save_output", False):
            save_outputsv2(
                self.config, batch_predictions, observations, create_dirs=True
            )
        self.predictions = self._batch_data(batch_predictions)
        self.calc_metrics(batch_predictions, observations)

    def inference(self) -> None:
        """推理模式（同 evaluate）。"""
        self.evaluate()

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        """计算评估指标。"""
        predictions = np.expand_dims(
            self._batch_data(batch_predictions)[:, :, 0].cpu().numpy(), 2
        )
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
        target = target[self.config["delta_model"]["phy_model"]["warm_up"]:]
        target = target[:len(predictions)]

        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )
        metrics.dump_metrics(self.config["out_path"])

    def _forward_loop(self, dataset, batch_start, batch_end):
        """批次前向传播循环。"""
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for i in range(len(batch_start)):
                sample = self.sampler.get_validation_sample(
                    dataset, batch_start[i], batch_end[i],
                )
                pred = self.model(sample)
                model_name = self.config["delta_model"]["phy_model"]["model"]
                if isinstance(model_name, list):
                    model_name = model_name[0]
                output = self.model.output_dict.get(model_name, pred)
                predictions.append(output)
        self.model.train()
        return predictions

    def _batch_data(self, batch_predictions):
        """合并批次预测结果。"""
        target_name = self.config["train"]["target"][0]
        all_preds = []
        for bp in batch_predictions:
            if isinstance(bp, dict):
                all_preds.append(bp[target_name])
            else:
                all_preds.append(bp)
        return torch.cat(all_preds, dim=1).unsqueeze(-1)
