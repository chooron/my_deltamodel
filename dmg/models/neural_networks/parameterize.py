import torch
import torch.nn as nn
from typing import Union


class Parameterize(torch.nn.Module):
    """
    Method B: 神经网络参数预测模块（对应 Calibrate 的 Method A）

    设计要点：
    1. 输入归一化流域静态属性 c_nn_norm (nx≈30~40维)，输出单组模型参数 (ny≈5~20维)
    2. 隐藏层加入 Dropout，支持推断阶段 MC-Dropout 采样
    3. 输出经 Sigmoid 映射到 [0, 1]，与 Method A 保持一致

    推断阶段用法（MC-Dropout）：
        model.train()  # 保持 Dropout 激活
        preds = [model(x) for _ in range(N)]  # N次前向传播
        params = torch.stack([p[1] for p in preds])  # (N, batch, ny)
        param_mean = params.mean(0)   # 点估计（用于KGE_B计算）
        param_p10  = params.quantile(0.1, dim=0)  # 先验区间下界
        param_p90  = params.quantile(0.9, dim=0)  # 先验区间上界
    """

    def __init__(
        self,
        *,
        nx: int,           # 流域静态属性维度（归一化后，约30~40）
        ny: int,           # 模型参数数量（约5~20）
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.15,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = "Parameterize"
        self.ny = ny
        self.device = device

        # 构建 MLP：输入层 → 隐藏层(含Dropout) → 输出层
        layers = []
        in_size = nx
        for i in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))   # 比 BatchNorm 对小批量更稳定
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_rate))  # MC-Dropout 的关键层
            in_size = hidden_size

        layers.append(nn.Linear(hidden_size, ny))      # 输出层：预测单组参数

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """He 初始化，适配 GELU 激活"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        return cls(
            nx=config["nx2"],
            ny=config["ny"],
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.2),
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        """
        参数
        ----
        x : dict，包含：
            'c_nn_norm' : Tensor, shape (batch, nx)  归一化流域静态属性

        返回
        ----
        (None, params)
            None    : 占位，与 Calibrate.forward 接口保持一致
            params  : Tensor, shape (batch, ny, 1)
                      最后一维保持与 Method A 的 num_start=1 对齐
        """
        attr = x['c_nn_norm']                      # (batch, nx)
        raw = self.mlp(attr)                       # (batch, ny)
        params = torch.sigmoid(raw).unsqueeze(-1)  # (batch, ny, 1)
        return None, params


# ──────────────────────────────────────────────
# MC-Dropout 推断工具函数
# ──────────────────────────────────────────────

def mc_dropout_inference(
    model: Parameterize,
    x: dict[str, torch.Tensor],
    n_samples: int = 100,
) -> dict[str, torch.Tensor]:
    """
    MC-Dropout 多次前向传播，返回参数统计量。

    用法
    ----
    model.train()   # 必须保持 train 模式使 Dropout 激活
    stats = mc_dropout_inference(model, x, n_samples=100)

    返回字典
    --------
    'mean'   : (batch, ny)  参数点估计，用于 KGE_B 计算
    'std'    : (batch, ny)  标准差，用于置信评级
    'p10'    : (batch, ny)  10th 分位数
    'p90'    : (batch, ny)  90th 分位数  ← 知识库先验区间
    'samples': (n_samples, batch, ny)  原始采样，供后续分析
    """
    model.train()  # 确保 Dropout 激活

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            _, params = model(x)           # (batch, ny, 1)
            samples.append(params.squeeze(-1))  # (batch, ny)

    samples = torch.stack(samples, dim=0)  # (n_samples, batch, ny)

    return {
        "mean"   : samples.mean(dim=0),
        "std"    : samples.std(dim=0),
        "p10"    : samples.quantile(0.1, dim=0),
        "p90"    : samples.quantile(0.9, dim=0),
        "samples": samples,
    }