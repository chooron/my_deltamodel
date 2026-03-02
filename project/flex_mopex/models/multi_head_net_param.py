import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadNetParam(nn.Module):
    """
    MOPEX 模型的动态参数网络

    与MultiHeadNet的区别：
    - 不输出静态params，而是通过LSTM基于xc_nn_norm动态预测部分参数
    - 需要额外接收xc_nn_norm (时序气象数据+静态属性) 作为输入

    输出：
    1. static_params: 静态MOPEX参数 (8个: Sb1, tw, tu, tc, tcrit, is_time, tmin, tmax) * nmul
    2. dynamic_params: 动态参数 [n_steps, n_grid, 4*nmul] (ddf, alpha, Sb2, Se)
    3. gamma_uh: 路由参数 (2个)
    """

    def __init__(
        self,
        input_dim: int = 27,         # 静态属性维度
        hidden_dim: int = 128,        # 共享层隐藏维度
        dropout: float = 0.0,
        nmul: int = 1,
        lstm_input_size: int = None,  # LSTM输入维度（xc_nn_norm的维度）
        lstm_hidden_size: int = 128,  # LSTM隐藏层大小
        lstm_dropout: float = 0.0,    # LSTM dropout
        device: str = "cuda:0",
    ):
        super().__init__()

        self.nmul = nmul
        self.lstm_hidden_size = lstm_hidden_size

        # 如果未指定LSTM输入维度，使用默认值
        if lstm_input_size is None:
            lstm_input_size = 3 + input_dim

        self.lstm_input_size = lstm_input_size

        # 静态参数：Sb1, tw, tu, tc, tcrit, is_time, tmin, tmax
        self.static_param_names = ["Sb1", "tw", "tu", "tc", "tcrit", "is_time", "tmin", "tmax"]
        # 动态参数：ddf, alpha, Sb2, Se
        self.dynamic_param_names = ["ddf", "alpha", "Sb2", "Se"]

        # 输出头配置
        self.num_params_dict: dict = {
            "static_params": len(self.static_param_names) * nmul,  # 8 * nmul
            "gamma_uh": 2,                                          # 路由参数
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

        # 2. 独立参数头 (Independent Heads) - 仅输出static_params和gamma_uh
        self.heads = nn.ModuleDict()
        for head_name, n_params in self.num_params_dict.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, n_params)
            )

        # 3. LSTM动态参数预测器 - 参考 MultiHeadNetDyn 的写法
        # 使用 Sequential 包装 Linear + ReLU + LSTM
        self.param_lstm = nn.Sequential(
            nn.Linear(lstm_input_size, lstm_hidden_size),
            nn.ReLU(),
            nn.LSTM(lstm_hidden_size, lstm_hidden_size, dropout=lstm_dropout, batch_first=False),
        )

        # LSTM输出层：预测动态参数
        self.param_fc = nn.Linear(lstm_hidden_size, len(self.dynamic_param_names) * nmul)

        # 4. 权重初始化
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

        # 初始化LSTM参数输出层
        nn.init.normal_(self.param_fc.weight, mean=0.0, std=0.001)
        if self.param_fc.bias is not None:
            nn.init.constant_(self.param_fc.bias, 0.0)

        print("[INFO] MultiHeadNetParam weights initialized successfully")

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cuda:0"):
        # 参考 MultiHeadNetDyn，直接使用 config["nx"] 作为 LSTM 输入维度
        # config["nx"] 应该是 xc_nn_norm 的维度
        return cls(
            input_dim=config["nx2"],
            nmul=config["nmul"],
            hidden_dim=config["hidden_size"],
            dropout=config["dr"],
            lstm_input_size=config["nx"],  # 直接使用 xc_nn_norm 的维度
            lstm_hidden_size=config.get("lstm_hidden_size", 128),
            lstm_dropout=config.get("lstm_dropout", 0.0),
            device=device,
        )

    def forward(self, x: dict[str, torch.Tensor]):
        """
        Args:
            x: 包含以下键的字典
                - "c_nn_norm": [Batch, Input_Dim] 静态属性
                - "xc_nn_norm": [n_steps, Batch, Input_Dim + Num_Forcings] 时序气象数据+静态属性

        Returns:
            params_dict: {
                "static_params": [Batch, 8*nmul],           # 静态MOPEX参数
                "dynamic_params": [n_steps, Batch, 4*nmul], # 动态参数
                "gamma_uh": [Batch, 2],                     # 路由参数
            }
        """
        # 1. 提取共享特征（静态属性）
        x_attr = x["c_nn_norm"]

        # 检查输入
        if self.training and torch.isnan(x_attr).any():
            print(f"[ERROR] NaN in static input!")
            x_attr = torch.nan_to_num(x_attr, nan=0.0)

        shared_feat = self.backbone(x_attr)

        # 2. 各头独立输出（static_params和gamma_uh）
        out_dict = {}
        for head_name, head_net in self.heads.items():
            out_dict[head_name] = head_net(shared_feat)

        # 3. LSTM预测动态参数 - 参考 MultiHeadNetDyn 的写法
        z1 = x["xc_nn_norm"]  # [n_steps, batch, lstm_input_size]

        # 检查输入
        if self.training and torch.isnan(z1).any():
            print(f"[ERROR] NaN in xc_nn_norm input!")
            z1 = torch.nan_to_num(z1, nan=0.0)

        # LSTM前向传播（Sequential会自动处理）
        lstm_out, _ = self.param_lstm(z1)  # [n_steps, batch, lstm_hidden_size]

        # 通过全连接层预测动态参数
        dynamic_params = self.param_fc(lstm_out)  # [n_steps, batch, 4*nmul]

        out_dict["dynamic_params"] = dynamic_params

        # 检查输出
        if self.training:
            for key, value in out_dict.items():
                if torch.isnan(value).any():
                    print(f"[ERROR] NaN in {key} output!")
                    raise ValueError(f"NaN detected in {key} output!")

        return out_dict
