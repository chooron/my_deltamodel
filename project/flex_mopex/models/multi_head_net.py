import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadNet(nn.Module):
    """
    MOPEX 模型的多头参数网络
    
    输出三个独立的head：
    1. params: MOPEX物理参数 (12个 * nmul)
    2. weights: 结构权重 (4个过程 * 2个状态 * nmul)
    3. gamma_uh: 路由参数 (2个)
    """
    
    def __init__(
        self,
        input_dim: int = 27,        # 静态属性维度
        hidden_dim: int = 128,       # 共享层隐藏维度
        dropout: float = 0.0,
        nmul: int = 1,
        device: str = "cuda:0",
    ):
        super().__init__()
        
        # MOPEX 模型的三个输出头
        self.num_params_dict: dict = {
            "params": 12 * nmul,      # MOPEX物理参数 (Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax)
            "weights": 4 * 2 * nmul,  # 结构权重 (w_phen, w_int, w_snow, w_sub, 每个2个logits: Off/On)
            "gamma_uh": 2,            # 路由参数 (rout_a, rout_b)
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
        # 作用：将通用特征映射到各类参数的特定空间
        self.heads = nn.ModuleDict()
        for head_name, n_params in self.num_params_dict.items():
            self.heads[head_name] = nn.Sequential(
                # 私有隐藏层，进一步隔离不同类型参数的特征空间
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
            x: 包含 "c_nn_norm" 的字典 - [Batch, Input_Dim] (静态属性)
            
        Returns:
            params_dict: {
                "params": [Batch, 12*nmul],      # MOPEX物理参数
                "weights": [Batch, 8*nmul],      # 结构权重(4过程*2状态*nmul)
                "gamma_uh": [Batch, 2],          # 路由参数
            }
        """
        # 1. 提取共享特征
        # shared_feat: [Batch, Hidden_Dim]
        x_attr = x["c_nn_norm"]
        shared_feat = self.backbone(x_attr)
        
        # 2. 各头独立输出
        out_dict = {}
        for head_name, head_net in self.heads.items():
            # 得到原始参数 (Raw)
            # params: 通过 Sigmoid + Scale 映射到物理范围
            # weights: 通过 Gumbel-Softmax 做独立二分类
            # gamma_uh: 通过 Sigmoid + Scale 映射到路由参数范围
            out_dict[head_name] = head_net(shared_feat)
            
        return out_dict