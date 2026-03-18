import logging
from typing import Any

import numpy as np
import torch

from dmg.trainers.cal_trainer import CalTrainer

log = logging.getLogger(__name__)


class EliteCalTrainer(CalTrainer):
    """
    在 CalTrainer 基础上增加 Top-K 精英池变异机制。
    通过周期性地将差成员重置为精英成员附近的扰动点，
    实现梯度下降与进化算法的混合优化策略。
    """

    def _get_num_start(self) -> int:
        """从模型中读取实际的 num_start（成员数）。"""
        model_name = self.config["delta_model"]["phy_model"]["model"]
        _model_name = model_name[0] if isinstance(model_name, list) else model_name
        _dpl_model = self.model.model_dict.get(_model_name, None)
        if _dpl_model is not None:
            _nn = getattr(_dpl_model, "nn_model", None)
            if _nn is not None and hasattr(_nn, "num_start"):
                return _nn.num_start
        raise RuntimeError(
            f"无法从模型 {_model_name} 中获取 num_start。"
            "请确保模型已正确初始化且 nn_model 具有 num_start 属性。"
        )

    def _compute_member_kge(self) -> np.ndarray:
        """
        对训练集做完整前向推理，逐成员计算 KGE。

        Returns
        -------
        np.ndarray
            shape=(n_basins, num_start) 的 KGE 矩阵，无效值填 np.nan。
        """
        num_start = self._get_num_start()
        warm_up = self.config["delta_model"]["phy_model"]["warm_up"]
        target_name = self.config["train"]["target"][0]

        dataset = self.train_dataset
        n_samples = dataset["xc_nn_norm"].shape[1]
        batch_size = self.config["test"]["batch_size"]
        batch_start = np.arange(0, n_samples, batch_size)
        batch_end = np.append(batch_start[1:], n_samples)

        # 收集所有 batch 的预测结果（shape: [T, B*num_start] 拼接后再整理）
        all_preds = []   # 每个元素 shape: [T, b*num_start] 或 [T, b, num_start]
        all_targets = []  # 每个元素 shape: [T, b]

        with torch.no_grad():
            for i in range(len(batch_start)):
                sample = self.sampler.get_validation_sample(
                    dataset, batch_start[i], batch_end[i]
                )
                prediction = self.model(sample, eval=True)

                # 取目标变量预测值，shape: [T, b*num_start] 或 [T, b, num_start]
                model_name = self.config["delta_model"]["phy_model"]["model"]
                _model_name = model_name[0] if isinstance(model_name, list) else model_name
                pred_tensor = prediction[target_name]  # [T, b*num_start]

                # target shape: [T, b, 1] -> [T, b]
                tgt = sample["target"][..., 0]  # [T, b]

                all_preds.append(pred_tensor.cpu().float())
                all_targets.append(tgt.cpu().float())

        # 拼接所有 batch：沿流域维度 (dim=1)
        # pred: [T, n_basins * num_start]，target: [T, n_basins]
        pred_full = torch.cat(all_preds, dim=1)    # [T, n_basins*num_start]
        tgt_full = torch.cat(all_targets, dim=1)   # [T, n_basins]

        T, total_cols = pred_full.shape
        n_basins = tgt_full.shape[1]

        # reshape pred -> [T, n_basins, num_start]
        pred_full = pred_full.view(T, n_basins, num_start)

        # pred 已由模型内部裁掉 warm-up，tgt 需手动裁剪对齐
        pred_np = pred_full.numpy()                    # [T-warm_up, n_basins, num_start]
        tgt_np = tgt_full[warm_up:].numpy()            # [T-warm_up, n_basins]

        # 防止长度仍不一致（边界情况），取最短对齐
        min_len = min(pred_np.shape[0], tgt_np.shape[0])
        pred_full = pred_np[:min_len]
        tgt_np = tgt_np[:min_len]

        T_valid, n_basins, _ = pred_full.shape
        kge_matrix = np.full((n_basins, num_start), np.nan)

        for b in range(n_basins):
            t_b = tgt_np[:, b]  # [T']
            valid_t = ~np.isnan(t_b)

            for m in range(num_start):
                p_m = pred_full[:, b, m]  # [T']
                valid_p = ~np.isnan(p_m)
                valid = valid_t & valid_p

                if valid.sum() < 2:
                    continue

                p = p_m[valid]
                t = t_b[valid]

                mean_p = p.mean()
                mean_t = t.mean()
                std_p = p.std()
                std_t = t.std()

                # 防止零标准差导致除零
                if std_t < 1e-10 or std_p < 1e-10:
                    continue

                # Pearson 相关系数
                r = float(np.corrcoef(p, t)[0, 1])
                if np.isnan(r):
                    continue

                beta = mean_p / (mean_t + 1e-10)
                gamma = std_p / (std_t + 1e-10)
                kge_matrix[b, m] = 1.0 - np.sqrt(
                    (r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
                )

        return kge_matrix

    def _reset_poor_members(
        self,
        kge_matrix: np.ndarray,
        threshold_ratio: float = 0.25,
        elite_ratio: float = 0.10,
    ) -> dict:
        """
        将每个流域中 KGE 排名靠后的差成员重置为精英成员附近的扰动点。

        Parameters
        ----------
        kge_matrix : np.ndarray
            shape=(n_basins, num_start)，由 _compute_member_kge 返回。
        threshold_ratio : float
            后 threshold_ratio 比例的成员被视为差成员，默认 0.25。
        elite_ratio : float
            前 elite_ratio 比例的成员被视为精英，默认 0.10。

        Returns
        -------
        dict
            重置统计信息字典。
        """
        n_basins, num_start = kge_matrix.shape

        # 获取 Calibrate 模块的 params
        model_name = self.config["delta_model"]["phy_model"]["model"]
        _model_name = model_name[0] if isinstance(model_name, list) else model_name
        _dpl_model = self.model.model_dict.get(_model_name)
        params = _dpl_model.nn_model.params  # 访问 Calibrate 的 logit 域参数

        # 噪声尺度：logit 域 std ≈ 1.34，取约 3% 作为扰动
        # params shape: (n_basins, ny, num_start)
        noise_scale = 0.05

        # 计算精英池和差成员的数量
        n_elite = max(1, int(num_start * elite_ratio))
        n_poor = max(1, int(num_start * threshold_ratio))

        n_reset_total = 0
        n_actually_reset = 0
        elite_kge_list = []
        poor_kge_list = []
        global_best_kge_list = []

        with torch.no_grad():
            for b in range(n_basins):
                kge_b = kge_matrix[b]  # [num_start]

                # NaN 视为最差：用 -inf 替换 NaN 后排序
                kge_b_filled = np.where(np.isnan(kge_b), -np.inf, kge_b)

                # 从好到差排序（降序），得到成员索引
                sorted_idx = np.argsort(kge_b_filled)[::-1]  # 降序

                elite_idx = sorted_idx[:n_elite]   # 前 n_elite：精英池
                poor_idx = sorted_idx[-n_poor:]    # 后 n_poor：差成员

                # 收集统计信息（使用原始 KGE，NaN 保留）
                elite_kge_vals = kge_b[elite_idx]
                poor_kge_vals = kge_b[poor_idx]
                global_best_kge_list.append(np.nanmedian(kge_b))

                valid_elite = elite_kge_vals[~np.isnan(elite_kge_vals)]
                if valid_elite.size > 0:
                    elite_kge_list.append(float(np.median(valid_elite)))

                valid_poor = poor_kge_vals[~np.isnan(poor_kge_vals)]
                if valid_poor.size > 0:
                    poor_kge_list.append(float(np.median(valid_poor)))

                # 对每个差成员：随机采样一个精英作为 donor，加噪声
                for poor_m in poor_idx:
                    # 随机选取一个精英 donor（避免所有差成员聚集到同一局部最优）
                    donor_m = int(np.random.choice(elite_idx))

                    # 在 logit 域直接操作，不经过 sigmoid
                    donor_params = params.data[b, :, donor_m].clone()  # [ny]
                    noise = torch.randn_like(donor_params) * noise_scale
                    params.data[b, :, poor_m] = donor_params + noise

                    # 清除该成员对应的 Adam 动量，防止旧动量把新参数拉回原位
                    if params in self.optimizer.state:
                        state = self.optimizer.state[params]
                        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                            if key in state:
                                state[key][b, :, poor_m] = 0.0

                    n_reset_total += 1
                    n_actually_reset += 1

        stats = {
            "n_reset_total": n_reset_total,
            "n_actually_reset": n_actually_reset,
            "elite_kge_median": float(np.median(elite_kge_list)) if elite_kge_list else float("nan"),
            "poor_kge_median": float(np.median(poor_kge_list)) if poor_kge_list else float("nan"),
            "global_best_kge": float(np.nanmedian(global_best_kge_list)) if global_best_kge_list else float("nan"),
        }
        return stats

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        """
        继承父类的完整训练逻辑，仅在末尾（保存模型之前）插入精英变异触发判断。
        """
        # 完全复用父类的 epoch 训练逻辑
        super().train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

        # ================================================================
        # 精英变异触发：在保存模型之后（父类已保存），周期性执行
        # ================================================================
        reset_interval = self.config["train"].get("elite_reset_interval", 20)
        reset_start    = self.config["train"].get("elite_reset_start", 30)
        reset_end      = self.config["train"].get("elite_reset_end", 90)

        if (epoch >= reset_start and
                epoch <= reset_end and
                (epoch - reset_start) % reset_interval == 0):
            threshold_ratio = self.config["train"].get("elite_threshold_ratio", 0.25)
            elite_ratio = self.config["train"].get("elite_ratio", 0.10)

            num_start = self._get_num_start()
            n_elite = max(1, int(num_start * elite_ratio))
            n_poor = max(1, int(num_start * threshold_ratio))
            self._emit_progress(
                f"[Epoch {epoch}] 精英变异开始 | "
                f"num_start={num_start} | n_elite={n_elite} | n_poor={n_poor}"
            )

            kge_matrix_before = self._compute_member_kge()
            global_kge_before = float(np.nanmedian(kge_matrix_before))
            stats = self._reset_poor_members(kge_matrix_before, threshold_ratio, elite_ratio)
            kge_matrix_after = self._compute_member_kge()
            global_kge_after = float(np.nanmedian(kge_matrix_after))

            # 变异后精英/差成员KGE统计
            n_basins = kge_matrix_after.shape[0]
            elite_after_list = []
            poor_after_list = []
            for b in range(n_basins):
                kge_b = kge_matrix_after[b]
                kge_b_filled = np.where(np.isnan(kge_b), -np.inf, kge_b)
                sorted_idx = np.argsort(kge_b_filled)[::-1]
                elite_vals = kge_b[sorted_idx[:n_elite]]
                poor_vals = kge_b[sorted_idx[-n_poor:]]
                valid_e = elite_vals[~np.isnan(elite_vals)]
                valid_p = poor_vals[~np.isnan(poor_vals)]
                if valid_e.size > 0:
                    elite_after_list.append(float(np.median(valid_e)))
                if valid_p.size > 0:
                    poor_after_list.append(float(np.median(valid_p)))
            elite_kge_after = float(np.median(elite_after_list)) if elite_after_list else float("nan")
            poor_kge_after = float(np.median(poor_after_list)) if poor_after_list else float("nan")

            self._emit_progress(
                f"[Epoch {epoch:>4}/{self.epochs}] 精英变异完成 | "
                f"重置参数组={stats['n_actually_reset']} | "
                f"全局KGE: {global_kge_before:.4f} -> {global_kge_after:.4f} | "
                f"精英KGE: {stats['elite_kge_median']:.4f} -> {elite_kge_after:.4f} | "
                f"差成员KGE: {stats['poor_kge_median']:.4f} -> {poor_kge_after:.4f}"
            )
