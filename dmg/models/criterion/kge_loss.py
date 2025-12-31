from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class KgeLoss(BaseCriterion):
    """Kling-Gupta efficiency (KGE) loss function.

    Standard KGE Implementation for Batch Processing.
    Calculates KGE for each sample in the batch independently and returns the sum.

    The KGE is calculated as:
        KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.
        - eps: Stability term to prevent division by zero. Default is 0.1 (or 1e-6).
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: int,
    ) -> None:
        super().__init__(config, device)
        self.name = "Batch KGE Loss"
        self.config = config
        self.device = device
        # 建议 eps 稍微设小一点，例如 1e-5 或 1e-6，以免过度影响 beta 和 gamma 的计算
        self.eps = kwargs.get("eps", config.get("eps", 1e-5))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_pred
            Tensor of predicted target data. Shape: [Batch, Time]
        y_obs
            Tensor of target observation data. Shape: [Batch, Time]

        Returns
        -------
        torch.Tensor
            The SUM of (1 - KGE) losses for the batch.
            Scalar tensor suitable for backward().
        """
        # _format 通常用于统一维度，假设返回 [Batch, Time]
        prediction, target = self._format(y_pred, y_obs)

        # 1. 创建 Mask 处理 NaNs (Target 中可能存在缺失值)
        # mask shape: [Batch, Time]
        mask = ~torch.isnan(target)

        # 将 NaNs 填充为 0，防止 sum() 出现 NaN
        # 注意：后续计算均值时会除以有效 count，所以填 0 不影响
        target_filled = torch.nan_to_num(target, nan=0.0)

        # 预测值只在观测值有效的地方参与计算
        pred_masked = prediction * mask
        target_masked = target_filled * mask

        # 2. 计算有效时间步数 (per sample)
        # shape: [Batch]
        count = mask.sum(dim=1)
        # 防止除以 0 (如果某一行全是 NaN)
        count = torch.clamp(count, min=1.0)

        # 3. 计算均值 (Mean) - 沿时间维度 dim=1
        mean_p = pred_masked.sum(dim=1) / count
        mean_t = target_masked.sum(dim=1) / count

        # 4. 计算标准差 (Std)
        # (x - mean) * mask 确保无效位置为 0，不影响平方和
        # unsqueeze(1) 是为了广播: [Batch, 1]
        dev_p = (prediction - mean_p.unsqueeze(1)) * mask
        dev_t = (target_filled - mean_t.unsqueeze(1)) * mask

        # 也就是 sum((x-mean)^2) / N
        var_p = (dev_p**2).sum(dim=1) / count
        var_t = (dev_t**2).sum(dim=1) / count

        std_p = torch.sqrt(var_p)
        std_t = torch.sqrt(var_t)

        # 5. 计算相关系数 (r) - Pearson Correlation
        # sum((p - mean_p) * (t - mean_t))
        numerator = (dev_p * dev_t).sum(dim=1)
        # sqrt(sum((p - mean_p)^2) * sum((t - mean_t)^2))
        denominator = torch.sqrt((dev_p**2).sum(dim=1)) * torch.sqrt(
            (dev_t**2).sum(dim=1)
        )

        r = numerator / (denominator + self.eps)

        # 6. 计算 KGE 组件
        # beta: 均值比 (Bias ratio)
        beta = mean_p / (mean_t + self.eps)

        # gamma: 变异系数比 (Variability ratio)
        gamma = std_p / (std_t + self.eps)

        # 7. 计算 KGE
        # shape: [Batch]
        kge = 1.0 - torch.sqrt(
            (r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
        )

        # 8. 返回 Loss
        # 目标是最大化 KGE，即最小化 (1 - KGE)
        # 使用 sum() 而不是 mean()，是因为在 Parameter 独立率定场景下，
        # 我们希望每个流域的梯度强度是独立的，不被 Batch Size 稀释。
        loss = (1.0 - kge).sum()

        return loss
