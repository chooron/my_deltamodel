from typing import Any, Optional
import torch
from dmg.models.criterion.base import BaseCriterion

class KgeHybridLoss(BaseCriterion):
    """
    Hybrid KGE Loss: Weighted sum of Standard KGE and Inverse KGE.
    
    Combines:
    1. Standard KGE (Gupta et al., 2009): Focuses on high flows and general bias.
    2. Inverse KGE (Pushpalatha et al., 2012): Focuses on low flows via 1/(Q + eps) transform.
    
    Formula:
        Loss = w_std * Sum(1 - KGE_std) + w_inv * Sum(1 - KGE_inv)
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: int,
    ) -> None:
        super().__init__(config, device)
        self.name = "Hybrid KGE Loss (Standard + Inverse)"
        self.config = config
        self.device = device
        
        # 数值稳定项 (用于分母防止除零)
        self.stability_eps = kwargs.get("eps", config.get("eps", 1e-5))
        
        # 权重配置: [Standard_Weight, Inverse_Weight]
        # 默认为 [1.0, 1.0]
        weights = kwargs.get("hybrid_weights", config.get("hybrid_weights", [1.0, 1.0]))
        self.w_std = weights[0]
        self.w_inv = weights[1]

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute Hybrid Loss.
        Args:
            y_pred: [Batch, Time]
            y_obs:  [Batch, Time]
        """
        # --- 0. 预处理与掩码 ---
        prediction, target = self._format(y_pred, y_obs)
        
        # mask shape: [Batch, Time]
        mask = ~torch.isnan(target)
        target_filled = torch.nan_to_num(target, nan=0.0)
        
        # 预测值也要应用 mask (只在有观测值的地方计算)
        pred_masked = prediction * mask
        target_masked = target_filled * mask
        
        # 有效时间步统计 [Batch]
        count = mask.sum(dim=1)
        count = torch.clamp(count, min=1.0) # 防止除零

        # =====================================================================
        # Part 1: Standard KGE (针对原始流量 Q)
        # =====================================================================
        
        # 1.1 均值
        mean_p = pred_masked.sum(dim=1) / count
        mean_t = target_masked.sum(dim=1) / count
        
        # 1.2 方差/标准差
        # [Batch, 1] 广播
        mean_p_unsq = mean_p.unsqueeze(1)
        mean_t_unsq = mean_t.unsqueeze(1)
        
        dev_p = (prediction - mean_p_unsq) * mask
        dev_t = (target_filled - mean_t_unsq) * mask
        
        var_p = (dev_p ** 2).sum(dim=1) / count
        var_t = (dev_t ** 2).sum(dim=1) / count
        
        std_p = torch.sqrt(var_p)
        std_t = torch.sqrt(var_t)
        
        # 1.3 相关系数 r
        num = (dev_p * dev_t).sum(dim=1)
        den = torch.sqrt((dev_p ** 2).sum(dim=1)) * torch.sqrt((dev_t ** 2).sum(dim=1))
        r = num / (den + self.stability_eps)
        
        # 1.4 KGE 组件
        beta = mean_p / (mean_t + self.stability_eps)
        gamma = std_p / (std_t + self.stability_eps)
        
        kge_std_val = 1.0 - torch.sqrt(
            (r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
        )
        
        # 标准 KGE Loss (Sum over batch)
        loss_std = (1.0 - kge_std_val).sum()

        # =====================================================================
        # Part 2: Inverse KGE (针对变换流量 1/Q)
        # =====================================================================
        
        # 2.1 动态计算 Inverse 变换所需的 epsilon
        # epsilon = 0.01 * mean(Q_obs) per catchment
        obs_mean_raw = target_masked.sum(dim=1) / count
        epsilon_dyn = (obs_mean_raw * 0.01).unsqueeze(1)
        epsilon_dyn = torch.clamp(epsilon_dyn, min=1e-3) # 极小值保护
        
        # 2.2 倒数变换 Q' = 1 / (Q + eps)
        # 安全 clamp 防止负数导致的分母异常
        pred_safe = torch.clamp(prediction, min=0.0)
        target_safe = torch.clamp(target_filled, min=0.0)
        
        pred_inv = 1.0 / (pred_safe + epsilon_dyn)
        target_inv = 1.0 / (target_safe + epsilon_dyn)
        
        # 应用 mask 到变换后的数据
        pred_inv_masked = pred_inv * mask
        target_inv_masked = target_inv * mask
        
        # 2.3 Inverse 数据的均值
        mean_p_inv = pred_inv_masked.sum(dim=1) / count
        mean_t_inv = target_inv_masked.sum(dim=1) / count
        
        # 2.4 Inverse 数据的方差/标准差
        mean_p_inv_unsq = mean_p_inv.unsqueeze(1)
        mean_t_inv_unsq = mean_t_inv.unsqueeze(1)
        
        dev_p_inv = (pred_inv - mean_p_inv_unsq) * mask
        dev_t_inv = (target_inv - mean_t_inv_unsq) * mask
        
        var_p_inv = (dev_p_inv ** 2).sum(dim=1) / count
        var_t_inv = (dev_t_inv ** 2).sum(dim=1) / count
        
        std_p_inv = torch.sqrt(var_p_inv)
        std_t_inv = torch.sqrt(var_t_inv)
        
        # 2.5 Inverse 数据的相关系数 r
        num_inv = (dev_p_inv * dev_t_inv).sum(dim=1)
        den_inv = torch.sqrt((dev_p_inv ** 2).sum(dim=1)) * torch.sqrt((dev_t_inv ** 2).sum(dim=1))
        r_inv = num_inv / (den_inv + self.stability_eps)
        
        # 2.6 Inverse KGE 组件
        beta_inv = mean_p_inv / (mean_t_inv + self.stability_eps)
        gamma_inv = std_p_inv / (std_t_inv + self.stability_eps)
        
        kge_inv_val = 1.0 - torch.sqrt(
            (r_inv - 1.0) ** 2 + (beta_inv - 1.0) ** 2 + (gamma_inv - 1.0) ** 2
        )
        
        # Inverse KGE Loss (Sum over batch)
        loss_inv = (1.0 - kge_inv_val).sum()

        # =====================================================================
        # Part 3: Combine
        # =====================================================================
        
        total_loss = (self.w_std * loss_std) + (self.w_inv * loss_inv)
        
        return total_loss