from typing import Any, Optional
import torch
import torch.nn.functional as F
from dmg.models.criterion.base import BaseCriterion

class KgeInverseLoss(BaseCriterion):
    """
    Inverse Kling-Gupta Efficiency (KGE) Loss for Low Flow Optimization.
    
    Ref: Pushpalatha et al. (2012). "A review of efficiency criteria suitable 
         for evaluating low-flow simulations." Journal of Hydrology.
    
    Calculates KGE on transformed flow: Q' = 1 / (Q + epsilon)
    where epsilon = 0.01 * mean(Q_obs) per catchment.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: int,
    ) -> None:
        super().__init__(config, device)
        self.name = "Batch Inverse KGE Loss"
        self.config = config
        self.device = device
        # 这里的 eps 是用于 KGE 分母(beta/gamma/r)的数值稳定项，
        # 不是用于 1/Q 变换的 epsilon (那个是动态计算的)
        self.stability_eps = kwargs.get("eps", config.get("eps", 1e-5))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute Inverse KGE loss.
        
        Args:
            y_pred: [Batch, Time] Simulated flow
            y_obs:  [Batch, Time] Observed flow
        """
        # 1. 基础维度对齐
        prediction, target = self._format(y_pred, y_obs)

        # 2. 处理 NaNs (Target 可能缺失)
        # mask shape: [Batch, Time]
        mask = ~torch.isnan(target)
        
        # 临时填充 NaN 以便进行 sum/mean 计算 (mask 会过滤掉它们)
        target_filled = torch.nan_to_num(target, nan=0.0)
        
        # ---------------------------------------------------------------------
        # Step A: 动态计算 Epsilon (针对每个 Batch 样本独立计算)
        # ---------------------------------------------------------------------
        # 逻辑：epsilon = 0.01 * mean(Q_obs)
        # 必须仅基于有效观测数据计算均值
        
        count = mask.sum(dim=1)
        count = torch.clamp(count, min=1.0) # 防止除零
        
        # [Batch]
        obs_mean_raw = (target_filled * mask).sum(dim=1) / count
        
        # [Batch, 1] 广播用于后续加法
        # 增加一个极小值 1e-3 防止完全干涸的河流导致 epsilon=0 -> 1/0 爆炸
        epsilon = (obs_mean_raw * 0.01).unsqueeze(1)
        epsilon = torch.clamp(epsilon, min=1e-3) 

        # ---------------------------------------------------------------------
        # Step B: 倒数变换 (Inverse Transformation)
        # ---------------------------------------------------------------------
        # Q_inv = 1 / (Q + eps)
        # 即使模型预测负值（虽然物理上不应发生），也要 clamp 到 0 保证安全性
        
        pred_safe = torch.clamp(prediction, min=0.0)
        target_safe = torch.clamp(target_filled, min=0.0)

        pred_inv = 1.0 / (pred_safe + epsilon)
        target_inv = 1.0 / (target_safe + epsilon)

        # 注意：变换后，原始的 mask 依然有效（因为只是做了数值映射）
        # 将变换后的数据应用 mask
        pred_masked = pred_inv * mask
        target_masked = target_inv * mask

        # ---------------------------------------------------------------------
        # Step C: 标准 KGE 计算流程 (使用变换后的数据)
        # ---------------------------------------------------------------------
        
        # 1. 计算均值 (Mean of 1/Q)
        mean_p = pred_masked.sum(dim=1) / count
        mean_t = target_masked.sum(dim=1) / count

        # 2. 计算方差/标准差 (Variance/Std of 1/Q)
        # [Batch, 1]
        mean_p_unsq = mean_p.unsqueeze(1)
        mean_t_unsq = mean_t.unsqueeze(1)
        
        # 偏差计算 (应用 mask 确保无效点不贡献误差)
        dev_p = (pred_inv - mean_p_unsq) * mask
        dev_t = (target_inv - mean_t_unsq) * mask

        var_p = (dev_p ** 2).sum(dim=1) / count
        var_t = (dev_t ** 2).sum(dim=1) / count

        std_p = torch.sqrt(var_p)
        std_t = torch.sqrt(var_t)

        # 3. 计算相关系数 (r) - Pearson
        numerator = (dev_p * dev_t).sum(dim=1)
        denominator = torch.sqrt((dev_p ** 2).sum(dim=1)) * torch.sqrt((dev_t ** 2).sum(dim=1))
        
        # 添加 stability_eps 防止分母为 0
        r = numerator / (denominator + self.stability_eps)

        # 4. 计算 KGE 组件 (alpha/beta)
        # 注意：有些文献中 alpha = std_p / std_t, beta = mean_p / mean_t
        # 这里沿用你代码中的命名习惯 (gamma=std_ratio)
        
        # Beta (Bias Ratio)
        beta = mean_p / (mean_t + self.stability_eps)
        
        # Gamma (Variability Ratio)
        gamma = std_p / (std_t + self.stability_eps)

        # 5. 组合 KGE
        # shape: [Batch]
        kge_value = 1.0 - torch.sqrt(
            (r - 1.0) ** 2 + 
            (beta - 1.0) ** 2 + 
            (gamma - 1.0) ** 2
        )

        # ---------------------------------------------------------------------
        # Step D: Loss 输出
        # ---------------------------------------------------------------------
        # 目标是最大化 KGE，Loss = Sum(1 - KGE)
        loss = (1.0 - kge_value).mean()

        return loss