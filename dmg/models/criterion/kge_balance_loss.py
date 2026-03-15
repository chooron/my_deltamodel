from typing import Any, Optional
import torch
from dmg.models.criterion.base import BaseCriterion


class KgeBalanceLoss(BaseCriterion):
    """
    KGE + Balance Smooth Hybrid Loss: 结合标准KGE损失和公式平衡平滑损失。

    Combines:
    1. Standard KGE (Gupta et al., 2009): 评估径流模拟精度
    2. Balance Smooth Loss: 确保不同公式在滑动窗口内的水量平衡趋近于0

    Formula:
        Loss = w_kge * (1 - KGE) + w_balance * mean(normalized_diff^2)

    其中 normalized_diff = (flux_j - mean_flux) / scale
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: int,
    ) -> None:
        super().__init__(config, device)
        self.name = "KGE + Balance Smooth Loss"
        self.config = config
        self.device = device

        # 数值稳定项
        self.stability_eps = kwargs.get("eps", config.get("eps", 1e-5))

        # 权重配置: [KGE_Weight, Balance_Weight]
        # 默认为 [1.0, 0.01] - 降低balance权重避免过度约束
        weights = kwargs.get("hybrid_weights", config.get("hybrid_weights", [1.0, 0.01]))
        self.w_kge = weights[0]
        self.w_balance = weights[1]

        # 是否使用均值而不是求和（更稳定）
        self.use_mean = kwargs.get("use_mean", config.get("use_mean", True))

        # 过程名称和选项数量
        self.process_options = {
            "snow_outflow": 3,
            "infiltration": 3,
            "evaporation": 3,
            "quickflow": 3,
            "baseflow": 2,
        }

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        计算混合损失。

        Args:
            y_pred: [Time, Batch] 预测径流
            y_obs:  [Time, Batch] 观测径流
            **kwargs: 必须包含 'model_output' 键，值为模型的完整输出字典

        Returns:
            torch.Tensor: 标量损失值
        """
        # --- Part 1: Batch KGE Loss (与 KgeBatchLoss 一致的归一化方式) ---
        prediction, target = self._format(y_pred, y_obs)

        # 拉平所有 basin，只保留有效观测
        mask = ~torch.isnan(target)
        p_sub = prediction[mask]
        t_sub = target[mask]

        mean_p = torch.mean(p_sub)
        mean_t = torch.mean(t_sub)
        std_p = torch.std(p_sub)
        std_t = torch.std(t_sub)

        # 相关系数 r
        numerator = torch.sum((p_sub - mean_p) * (t_sub - mean_t))
        denominator = torch.sqrt(
            torch.sum((p_sub - mean_p) ** 2) * torch.sum((t_sub - mean_t) ** 2)
        )
        r = numerator / (denominator + self.stability_eps)

        # beta, gamma
        beta = mean_p / (mean_t + self.stability_eps)
        gamma = std_p / (std_t + self.stability_eps)

        kge_val = 1.0 - torch.sqrt(
            (r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
        )

        # KGE Loss (标量，约 0.2-0.3)
        loss_kge = 1.0 - kge_val

        # --- Part 2: Balance Smooth Loss ---
        if "model_output" not in kwargs:
            # 如果没有 model_output，只返回 KGE loss
            return self.w_kge * loss_kge

        model_output = kwargs["model_output"]
        loss_balance = torch.tensor(0.0, device=self.device)
        balance_count = 0

        # 遍历所有过程
        for proc_name, n_options in self.process_options.items():
            for j in range(n_options):
                key = f"balance_{proc_name}_{j}"
                if key in model_output:
                    normalized_diff = model_output[key]  # [T, B]
                    # 检查是否有 NaN 或 Inf
                    if torch.isnan(normalized_diff).any() or torch.isinf(normalized_diff).any():
                        # 跳过有问题的数据
                        continue
                    # 使用 Huber loss (smooth L1)，对大值梯度更温和，避免梯度爆炸
                    loss_component = torch.nn.functional.smooth_l1_loss(
                        normalized_diff,
                        torch.zeros_like(normalized_diff),
                        beta=1.0,
                        reduction="mean" if self.use_mean else "sum",
                    )
                    loss_balance = loss_balance + loss_component
                    balance_count += 1

        # 归一化 balance loss：取所有过程的平均
        if balance_count > 0:
            loss_balance = loss_balance / balance_count

        # --- Part 3: Combine ---
        total_loss = (self.w_kge * loss_kge) + (self.w_balance * loss_balance)

        return total_loss
