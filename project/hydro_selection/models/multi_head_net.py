import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 27,        # 静态属性维度
        hidden_dim: int = 128,       # 共享层隐藏维度
        dropout: float = 0.0,
        nmul: int = 1,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.num_params_dict: dict = {
            "HBV": 14 * nmul, 
            "SHM": 7 * nmul, 
            "EXPHYDRO": 6 * nmul, 
            "HYMOD": 5 * nmul,
            "GAMMA_UH": 2 * 4, # 4个模型
        }
        
        # 1. 共享主干 (Shared Backbone)
        # 作用：提取流域的通用“嵌入表示” (Embedding)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh 在物理参数映射中通常比 ReLU 更稳定
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        # 2. 独立参数头 (Independent Heads)
        # 作用：将通用特征映射到各模型的特定参数空间
        self.heads = nn.ModuleDict()
        for model_name, n_params in self.num_params_dict.items():
            self.heads[model_name] = nn.Sequential(
                # 可以加一层私有隐藏层，进一步隔离干扰
                nn.Linear(hidden_dim, hidden_dim // 2), 
                nn.Tanh(),
                # 输出层：输出未归一化的参数 (Raw params)
                nn.Linear(hidden_dim // 2, n_params)
            )
        self.to(device)
            
    @classmethod
    def build_by_config(cls, config: dict, device: str = "cuda:0"):
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
            x_attr: [Batch, Input_Dim] (静态属性)
        Returns:
            params_dict: {
                "HBV": [Batch, 14],
                "SHM": [Batch, 7],
                ...
            }
        """
        # 1. 提取共享特征
        # shared_feat: [Batch, Hidden_Dim]
        x_attr = x["c_nn_norm"]
        shared_feat = self.backbone(x_attr)
        
        # 2. 各头独立输出
        out_dict = {}
        for model_name, head_net in self.heads.items():
            # 得到原始参数 (Raw)，后续会在 BlendHydro 里通过 Sigmoid + Scale 映射到物理范围
            out_dict[model_name] = head_net(shared_feat)
            
        return out_dict