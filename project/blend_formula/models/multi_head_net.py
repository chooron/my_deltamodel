import torch
import torch.nn as nn

PARAM_BOUNDS = {
    # 入渗
    "x1_hmets_runoff_coeff": [0.0, 1.0],
    "x2_b_exp": [0.1, 3.0],
    "x3_hbv_beta": [0.5, 3.0],
    # 快速流
    "x4_log_k_quick": [-5.0, -1.0],
    "x5_q_max": [0.0, 100.0],
    "x6_n_quick": [0.5, 2.0],
    "x7_topmodel_lambda": [5.0, 10.0],
    # 蒸发
    "x8_pet_correction": [0.0, 3.0],
    "x9_sat_wilt": [0.0, 0.05],
    "x10_delta_fc": [0.0, 0.45],
    # 基流
    "x11_log_k_base": [-5.0, -2.0],
    "x12_n_base": [0.5, 2.0],
    # 雪平衡
    "x13_swi_min": [0.0, 0.1],
    "x14_delta_swi_max": [0.01, 0.3],
    "x15_swi_reduct": [0.005, 0.1],
    "x16_refreeze_temp": [-5.0, 2.0],
    "x17_refreeze_exp": [0.0, 1.0],
    "x18_refreeze_factor": [0.0, 5.0],
    "x19_snow_swi_hbv": [0.0, 0.4],
    # 汇流
    "x20_gamma_shape_surf": [0.3, 20.0],
    "x21_gamma_scale_surf": [0.01, 5.0],
    "x22_gamma_shape_delay": [0.5, 13.0],
    "x23_gamma_scale_delay": [0.15, 1.5],
    # 潜在融雪
    "x24_min_melt_factor": [1.5, 3.0],
    "x25_delta_melt_factor": [0.0, 5.0],
    "x26_dd_melt_temp": [-1.0, 1.0],
    "x27_dd_aggradation": [0.01, 0.2],
    # 渗漏与土壤
    "x28_perc_coeff_top": [0.00001, 0.02],
    "x29_thickness_top": [0.0, 0.5],
    "x30_thickness_phreatic": [0.0, 2.0],
    # 气象
    "x31_rainsnow_temp": [-3.0, 3.0],
    "x32_rainsnow_delta": [0.5, 4.0],
    "x33_rain_correction": [0.8, 1.2],
    "x34_snow_correction": [0.8, 1.2],
    "x35_perc_coeff_phreatic": [0.0, 0.02],
    "x36_soilevap_vic_gamma": [0.1, 3.0],
}

ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

PROCESS_OPTIONS = {
    "rainsnow": 3,    # HBV, Dingman, Threshold
    "snowbal": 3,     # Simple, HBV, HMETS
    "infiltration": 3,# HMETS, VIC_ARNO, HBV
    "evaporation": 3, # ALL, LINEAR, VIC
    "quickflow": 3,   # LINEAR_ANALYTIC, VIC, TOPMODEL
    "baseflow": 2,    # LINEAR_ANALYTIC, POWER_LAW
}
TOTAL_WEIGHT_LOGITS = sum(PROCESS_OPTIONS.values())  # 17

class MultiHeadNet(nn.Module):
    """
    DiffBlendV1 的多头参数网络。

    输出通过三个独立 head 预测，然后拼接成 DiffBlendV1.forward
    所需的 parameters 结构：(None, raw_tensor)。

    raw_tensor 的拼接顺序必须与 DiffBlendV1.unpack_parameters 一致：
    1. 物理参数 logits: 35 * nmul
    2. 路由参数 logits: 2
    3. 过程权重 logits: 17
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

        # 与 DiffBlendV1 保持一致的参数维度
        self.n_phy_params = len(PARAM_BOUNDS)
        self.n_routing_params = len(ROUTING_BOUNDS)
        self.n_weight_logits = TOTAL_WEIGHT_LOGITS

        # 三个输出头: 物理参数、路由参数、过程权重 logits
        self.num_params_dict: dict = {
            "phy_params": self.n_phy_params * nmul,
            "routing_params": self.n_routing_params,
            "process_weight_logits": self.n_weight_logits,
        }
        self.learnable_param_count = sum(self.num_params_dict.values())
        
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

        # 3. 权重初始化 (Critical for numerical stability!)
        self._initialize_weights()

        self.to(device)

    def _initialize_weights(self):
        """
        初始化网络权重，防止梯度爆炸和NaN

        策略：
        1. 使用Xavier初始化隐藏层
        2. 输出层使用小方差初始化，确保初始输出接近0
        3. 偏置初始化为0
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初始化权重
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # 特别处理输出层：使用更小的初始化，确保初始输出在合理范围
        for head_name, head_net in self.heads.items():
            # 获取输出层（Sequential的最后一层）
            output_layer = head_net[-1]
            if isinstance(output_layer, nn.Linear):
                # 使用更小的方差初始化，使初始输出接近0
                # 经过sigmoid后会在0.5附近，这是一个安全的起点
                nn.init.normal_(output_layer.weight, mean=0.0, std=0.001)  # 从0.01改为0.001
                if output_layer.bias is not None:
                    nn.init.constant_(output_layer.bias, 0.0)

        print("[INFO] MultiHeadNet weights initialized successfully")
            
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
            parameters: (None, raw_tensor)
                raw_tensor shape = [Batch, learnable_param_count]
                拼接顺序 = [phy_params, routing_params, process_weight_logits]

            该返回值可直接作为 DiffBlendV1.forward 的 `parameters` 参数。
        """
        # 1. 提取共享特征
        x_attr = x["c_nn_norm"]

        # 检查输入是否有NaN（简化版）
        if self.training and torch.isnan(x_attr).any():
            print(f"[ERROR] NaN in MultiHeadNet input!")
            x_attr = torch.nan_to_num(x_attr, nan=0.0)
            print(f"[WARNING] Replaced NaN with 0.0 in input")

        shared_feat = self.backbone(x_attr)

        # 检查backbone输出
        if self.training and torch.isnan(shared_feat).any():
            print(f"[ERROR] NaN in backbone output!")
            raise ValueError("NaN detected in backbone output!")

        # 2. 各头独立输出
        out_dict = {}
        for head_name, head_net in self.heads.items():
            out_dict[head_name] = head_net(shared_feat)

            # 检查每个head的输出
            if self.training and torch.isnan(out_dict[head_name]).any():
                print(f"[ERROR] NaN in head '{head_name}' output!")
                raise ValueError(f"NaN detected in head '{head_name}' output!")

        # 3. 按 DiffBlendV1.unpack_parameters 的顺序拼接
        raw_tensor = torch.cat(
            [
                out_dict["phy_params"],
                out_dict["routing_params"],
                out_dict["process_weight_logits"],
            ],
            dim=-1,
        )

        if raw_tensor.shape[-1] != self.learnable_param_count:
            raise ValueError(
                f"Unexpected raw_tensor dim: {raw_tensor.shape[-1]}, "
                f"expected {self.learnable_param_count}"
            )

        if self.training and torch.isnan(raw_tensor).any():
            raise ValueError("NaN detected in concatenated raw_tensor output!")

        return None, raw_tensor

