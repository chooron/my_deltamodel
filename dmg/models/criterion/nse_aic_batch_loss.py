from typing import Any, Optional, Union, Dict

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class NseAicBatchLoss(BaseCriterion):
    """
    NSE Loss with Soft-AIC Regularization for Structural Sparsity.

    Minimizes:
        L = NSE_Loss + alpha * (Effective_Parameter_Count)

    where:
        NSE_Loss = mean((pred - obs)^2 / (std_obs + eps)^2)
        Effective_Parameter_Count = sum(weight_i * cost_i)

    This encourages the model to turn off unnecessary structural modules (weights -> 0)
    unless they significantly improve the NSE.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        - y_obs: Tensor [n_time, n_grid, 1]. Full observation data to calc stats. (Required)
        - eps: Stability term. Default 0.1.
        - aic_alpha: Hyperparameter for AIC penalty strength. Default 0.01.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.name = "Batch NSE Loss with Soft-AIC"
        self.config = config
        self.device = device

        # --- 1. NSE Initialization ---
        try:
            y_obs = kwargs["y_obs"]
            # Pre-calculate STD for each grid for NSE normalization
            # Shape: [n_grid]
            self.std = np.nanstd(y_obs[:, :, 0].cpu().detach().numpy(), axis=0)
        except KeyError as e:
            raise KeyError("'y_obs' is not provided in kwargs") from e

        self.eps = kwargs.get("eps", config.get("eps", 0.1))

        # --- 2. AIC Initialization ---
        # Alpha: Controls the trade-off between fitting (NSE) and sparsity (Complexity)
        # Suggest starting small (e.g., 1e-3) and potentially increasing during training.
        self.aic_alpha = kwargs.get("aic_alpha", config.get("aic_alpha", 0.005))

        # Define the parameter cost for each structural module.
        # This represents how many "free parameters" each module adds.
        self.param_costs = {
            "w_phen": 2.0,  # Adds: tmin, tmax
            "w_int": 2.0,  # Adds: alpha, is_time
            "w_snow": 2.0,  # Adds: ddf, tcrit (or tr)
            "w_sub": 1.0,  # Adds: structural path (complexifies Sb2 usage)
        }

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:

        # y_pred: [n_time, n_grid, 1]
        prediction, target = self._format(y_pred, y_obs)

        try:
            sample_ids = kwargs["sample_ids"]
            # support both torch.Tensor and numpy array/list
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.cpu().numpy().astype(int)
            else:
                sample_ids = np.asarray(sample_ids).astype(int)
        except KeyError as e:
            raise KeyError("'sample_ids' is not provided in kwargs") from e

        # ============================================================
        # 1. 计算 Fit Loss (NSE Loss) — 与 NseBatchLoss 一致
        # ============================================================
        if len(target) > 0:
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )

            # Mask: 只取 target 非 NaN 的位置（与 nse_batch_loss 一致）
            mask = ~torch.isnan(target)

            # 用布尔索引提取有效子集，避免 torch.where 的梯度穿透问题
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]

            # 计算归一化残差
            sq_res = (p_sub - t_sub) ** 2
            norm_res = sq_res / (std_sub + self.eps) ** 2

            loss_fit = torch.mean(norm_res)
        else:
            loss_fit = torch.tensor(0.0, device=self.device)

        # ============================================================
        # 2. AIC Penalty (预留，当前未启用)
        # ============================================================
        weights_dict = kwargs.get("weights", None)
        loss_complexity = torch.tensor(0.0, device=self.device)
        if weights_dict is not None:
            for name, cost in self.param_costs.items():
                if name in weights_dict:
                    w = weights_dict[name]
                    # w shape: [n_steps, n_grid, 1]
                    w = w[0, :, 0]
                    loss_complexity += torch.mean(w) * cost

        # ============================================================
        # 3. 组合 Loss
        # ============================================================
        final_loss = loss_fit + self.aic_alpha * loss_complexity

        return final_loss
