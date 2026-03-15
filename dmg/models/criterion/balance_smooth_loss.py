from typing import Any, Optional
import torch
from dmg.models.criterion.base import BaseCriterion


class BalanceSmoothLoss(BaseCriterion):
    """
    Balance Smooth Loss: 确保不同公式在滑动窗口内的水量平衡趋近于0。

    该损失函数读取模型输出中的 balance_{proc_name}_{j} 字段，
    这些字段表示每个过程的第j个公式与所有公式平均值的归一化差值。

    目标是最小化这些归一化差值，使得不同公式计算的水量在时间窗口内保持平滑一致。

    Formula:
        Loss = w_balance * mean(normalized_diff^2) for all processes and options

    其中 normalized_diff = (flux_j - mean_flux) / scale
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: int,
    ) -> None:
        super().__init__(config, device)
        self.name = "Balance Smooth Loss"
        self.config = config
        self.device = device

        # 权重配置
        self.w_balance = kwargs.get("balance_weight", config.get("balance_weight", 0.01))

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
        计算 Balance Smooth Loss。

        Args:
            y_pred: 模型输出字典（通过 kwargs 传入）
            y_obs: 观测值（本损失函数不使用）
            **kwargs: 必须包含 'model_output' 键，值为模型的完整输出字典

        Returns:
            torch.Tensor: 标量损失值
        """
        # 从 kwargs 获取模型完整输出
        if "model_output" not in kwargs:
            raise ValueError("BalanceSmoothLoss requires 'model_output' in kwargs")

        model_output = kwargs["model_output"]

        total_loss = torch.tensor(0.0, device=self.device)
        balance_count = 0

        # 遍历所有过程
        for proc_name, n_options in self.process_options.items():
            # 累积该过程所有选项的损失
            for j in range(n_options):
                key = f"balance_{proc_name}_{j}"
                if key in model_output:
                    normalized_diff = model_output[key]  # [T, B]
                    # 检查是否有 NaN 或 Inf
                    if torch.isnan(normalized_diff).any() or torch.isinf(normalized_diff).any():
                        # 跳过有问题的数据
                        continue
                    # 计算平方损失
                    if self.use_mean:
                        loss_component = (normalized_diff ** 2).mean()
                    else:
                        loss_component = (normalized_diff ** 2).sum()
                    total_loss = total_loss + loss_component
                    balance_count += 1

        # 如果使用求和模式，归一化balance loss
        if not self.use_mean and balance_count > 0:
            total_loss = total_loss / balance_count

        # 应用权重
        weighted_loss = self.w_balance * total_loss

        return weighted_loss
