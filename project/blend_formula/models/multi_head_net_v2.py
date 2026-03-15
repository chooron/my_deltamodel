"""
MultiHeadNetV2 - DiffBlendV2 的参数预测网络

输出三个头:
  phy_params            → [B, 36*nmul]  物理参数 logits
  routing_params        → [B, 2]        路由参数 logits
  process_weight_logits → [B, 17]       公式权重 logits（流域特异性）
"""

import torch
import torch.nn as nn

from project.blend_formula.models.diff_blend_v1 import TOTAL_WEIGHT_LOGITS

PARAM_BOUNDS = {
    "x1_hmets_runoff_coeff": [0.0, 1.0],
    "x2_b_exp": [0.1, 3.0],
    "x3_hbv_beta": [0.5, 3.0],
    "x4_log_k_quick": [-5.0, -1.0],
    "x5_q_max": [0.0, 100.0],
    "x6_n_quick": [0.5, 2.0],
    "x7_topmodel_lambda": [5.0, 10.0],
    "x8_pet_correction": [0.0, 3.0],
    "x9_sat_wilt": [0.0, 0.05],
    "x10_delta_fc": [0.0, 0.45],
    "x11_log_k_base": [-5.0, -2.0],
    "x12_n_base": [0.5, 2.0],
    "x13_swi_min": [0.0, 0.1],
    "x14_delta_swi_max": [0.01, 0.3],
    "x15_swi_reduct": [0.005, 0.1],
    "x16_refreeze_temp": [-5.0, 2.0],
    "x17_refreeze_exp": [0.0, 1.0],
    "x18_refreeze_factor": [0.0, 5.0],
    "x19_snow_swi_hbv": [0.0, 0.4],
    "x20_gamma_shape_surf": [0.3, 20.0],
    "x21_gamma_scale_surf": [0.01, 5.0],
    "x22_gamma_shape_delay": [0.5, 13.0],
    "x23_gamma_scale_delay": [0.15, 1.5],
    "x24_min_melt_factor": [1.5, 3.0],
    "x25_delta_melt_factor": [0.0, 5.0],
    "x26_dd_melt_temp": [-1.0, 1.0],
    "x27_dd_aggradation": [0.01, 0.2],
    "x28_perc_coeff_top": [0.00001, 0.02],
    "x29_thickness_top": [0.0, 0.5],
    "x30_thickness_phreatic": [0.0, 2.0],
    "x31_rainsnow_temp": [-3.0, 3.0],
    "x32_rainsnow_delta": [0.5, 4.0],
    "x33_rain_correction": [0.8, 1.2],
    "x34_snow_correction": [0.8, 1.2],
    "x35_perc_coeff_phreatic": [0.0, 0.02],
    "x36_soilevap_vic_gamma": [0.1, 3.0],
}

ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

class MultiHeadNetV2(nn.Module):
    """DiffBlendV2 的参数预测网络（含权重 logits 头）。

    输出结构: (None, raw_tensor)
    raw_tensor 拼接顺序与 DiffBlendV2.unpack_parameters 一致:
      1. 物理参数 logits:       36 * nmul
      2. 路由参数 logits:       2
      3. process_weight_logits: 17  ← 流域特异性权重
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        nmul: int = 1,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.n_phy_params = len(PARAM_BOUNDS)
        self.n_routing_params = len(ROUTING_BOUNDS)

        # 三个输出头: 物理参数、路由参数、公式权重 logits
        self.num_params_dict: dict = {
            "phy_params": self.n_phy_params * nmul,
            "routing_params": self.n_routing_params,
            "process_weight_logits": TOTAL_WEIGHT_LOGITS,
        }
        self.learnable_param_count = sum(self.num_params_dict.values())

        # 共享主干
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # 独立参数头
        self.heads = nn.ModuleDict()
        for head_name, n_params in self.num_params_dict.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, n_params),
            )

        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        """Xavier 初始化隐藏层，小方差初始化输出层。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        for head_name, head_net in self.heads.items():
            output_layer = head_net[-1]
            if isinstance(output_layer, nn.Linear):
                nn.init.normal_(output_layer.weight, mean=0.0, std=0.001)
                if output_layer.bias is not None:
                    nn.init.constant_(output_layer.bias, 0.0)

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
        """前向传播: 输出 (None, raw_tensor)，包含权重 logits。"""
        x_attr = x["c_nn_norm"]

        if self.training and torch.isnan(x_attr).any():
            x_attr = torch.nan_to_num(x_attr, nan=0.0)

        shared_feat = self.backbone(x_attr)

        if self.training and torch.isnan(shared_feat).any():
            raise ValueError("NaN detected in backbone output!")

        out_dict = {}
        for head_name, head_net in self.heads.items():
            out_dict[head_name] = head_net(shared_feat)
            if self.training and torch.isnan(out_dict[head_name]).any():
                raise ValueError(f"NaN detected in head '{head_name}' output!")

        # 拼接: [phy_params, routing_params, process_weight_logits]
        raw_tensor = torch.cat(
            [
                out_dict["phy_params"],
                out_dict["routing_params"],
                out_dict["process_weight_logits"],
            ],
            dim=-1,
        )

        if self.training and torch.isnan(raw_tensor).any():
            raise ValueError("NaN detected in concatenated raw_tensor output!")

        return None, raw_tensor
