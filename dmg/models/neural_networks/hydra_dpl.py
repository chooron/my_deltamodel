import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class ResidualBlock(nn.Module):
    """
    残差块：保持不变，用于构建深层的共享骨干网络。
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class HydraDpl(nn.Module):
    def __init__(
        self, 
        *,
        nx: int,          
        ny: int,          
        hidden_size: int, 
        num_heads: int = 16,
        dr: float = 0.5,
        device: str = "cuda:0"
    ) -> None:
        super().__init__()
        self.name = "HydraDpl"
        self.nx = nx
        self.ny = ny
        self.num_heads = num_heads
        self.device = torch.device(device)

        # 1. 共享骨干网络 (Shared Backbone)
        # 先降维或升维到 hidden_size
        self.input_layer = nn.Sequential(
            nn.Linear(nx, hidden_size),
            nn.ReLU()
        )
        
        # 堆叠残差块提取通用特征
        self.backbone = nn.Sequential(
            ResidualBlock(hidden_dim=hidden_size, dropout=dr),
            ResidualBlock(hidden_dim=hidden_size, dropout=dr)
        )

        # 2. 独立的多头 (Independent Heads)
        # N个独立的线性层，分别预测参数
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, ny) for _ in range(num_heads)
        ])
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cuda:0"):
        """
        根据配置字典构建模型
        """
        return cls(
            nx=config["nx2"],           
            ny=config["ny"],            
            hidden_size=config["hidden_size"],
            dr=config["dr"],
            num_heads=config.get("nmul", 16),
            device=device,
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        """
        Args:
            x: 字典，包含 key "c_nn_norm" -> Tensor [Batch, nx]
        Returns:
            Tuple(None, Output_Tensor)
            Output_Tensor Shape: [Batch, Num_Heads, ny]
            即 [num_basins, num_start, ny]
        """
        # 1. 获取输入 [Batch, nx]
        input_tensor = x["c_nn_norm"]
        
        # 2. Shared Backbone 提取特征
        x_feat = self.input_layer(input_tensor)
        features = self.backbone(x_feat) # [Batch, Hidden]

        # 3. 噪声注入 (Noise Injection) - 仅在训练时开启
        # 这是多头不趋同的关键
        if self.training:
            # 加上极小值防止 std=0
            std = features.std(dim=0, keepdim=True).detach() + 1e-6
            noise = torch.randn_like(features) * 0.05 * std
            features = features + noise

        # 4. 多头独立预测
        outputs = []
        for head in self.heads:
            # 每个 head 输出 [Batch, ny]
            out = head(features) 
            outputs.append(out)
        
        # 5. 堆叠: [Batch, Num_Heads, ny]
        # 维度对应: [self.num_basins, self.num_start, self.ny]
        raw_output = torch.stack(outputs, dim=1)

        # 6. 返回 Sigmoid (0-1)
        # 物理映射将在模型外部进行
        return None, torch.sigmoid(raw_output)