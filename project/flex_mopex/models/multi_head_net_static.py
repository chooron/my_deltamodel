import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadNetStatic(nn.Module):
    """
    MOPEX 模型的静态参数网络（用于FlexMopexV3）

    与MultiHeadNetParam的区别：
    - 只输出静态MOPEX参数（12个参数），不输出权重
    - 不使用LSTM，所有参数都是基于静态属性预测的
    - 权重由FlexMopexV3内部的WeightMLPGate动态预测

    输出：
    1. static_params: 静态MOPEX参数 (12个: Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax) * nmul
    2. gamma_uh: 路由参数 (2个)
    """

    def __init__(
        self,
        input_dim: int = 35,         # 静态属性维度
        hidden_dim: int = 128,        # 共享层隐藏维度
        dropout: float = 0.0,
        nmul: int = 1,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.nmul = nmul

        # MOPEX参数名称（12个）
        self.param_names = [
            "Sb1", "tw", "tu", "Se", "tc", "ddf",
            "tcrit", "Sb2", "alpha", "is_time", "tmin", "tmax"
        ]

        # 输出头配置
        self.num_params_dict: dict = {
            "static_params": len(self.param_names) * nmul,  # 12 * nmul
            "gamma_uh": 2,                                   # 路由参数
        }

        # 1. 共享主干 (Shared Backbone) - 用于学习流域静态属性
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # 2. 独立参数头 (Independent Heads)
        self.heads = nn.ModuleDict()
        for head_name, n_params in self.num_params_dict.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, n_params)
            )

        # 3. 权重初始化
        self._initialize_weights()

        self.to(device)

    def _initialize_weights(self):
        """
        初始化网络权重，防止梯度爆炸和NaN
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # 特别处理输出层
        for head_name, head_net in self.heads.items():
            output_layer = head_net[-1]
            if isinstance(output_layer, nn.Linear):
                nn.init.normal_(output_layer.weight, mean=0.0, std=0.001)
                if output_layer.bias is not None:
                    nn.init.constant_(output_layer.bias, 0.0)

        print("[INFO] MultiHeadNetStatic weights initialized successfully")

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cuda:0"):
        """从配置构建网络"""
        return cls(
            input_dim=config["nx2"],
            nmul=config["nmul"],
            hidden_dim=config["hidden_size"],
            dropout=config["dr"],
            device=device,
        )

    def forward(self, x: dict[str, torch.Tensor]):
        """
        Args:
            x: 包含以下键的字典
                - "c_nn_norm": [Batch, input_dim] 静态属性

        Returns:
            out_dict: 包含以下键的字典
                - "static_params": [Batch, 12*nmul] 静态MOPEX参数
                - "gamma_uh": [Batch, 2] 路由参数
        """
        # 1. 提取静态属性
        x_attr = x["c_nn_norm"]

        # 检查输入
        if self.training and torch.isnan(x_attr).any():
            print(f"[ERROR] NaN in static input!")
            x_attr = torch.nan_to_num(x_attr, nan=0.0)

        shared_feat = self.backbone(x_attr)

        # 2. 各头独立输出
        out_dict = {}
        for head_name, head_net in self.heads.items():
            out_dict[head_name] = head_net(shared_feat)

        # 检查输出
        if self.training:
            for key, value in out_dict.items():
                if torch.isnan(value).any():
                    print(f"[ERROR] NaN in {key} output!")
                    raise ValueError(f"NaN detected in {key} output!")

        return out_dict
