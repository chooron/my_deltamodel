import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadNetDyn(nn.Module):
    """
    MOPEX 模型的多头参数网络 + 动态权重LSTM

    与MultiHeadNet的区别：
    - 不输出静态weights，而是通过LSTM基于xc_nn_norm动态预测weights
    - 需要额外接收xc_nn_norm (时序气象数据+静态属性) 作为输入

    输出：
    1. params: MOPEX物理参数 (12个 * nmul)
    2. weights: 动态权重 [n_steps, n_grid, 4, 2] (通过LSTM预测)
    3. gamma_uh: 路由参数 (2个)
    """

    def __init__(
        self,
        input_dim: int = 27,        # 静态属性维度
        hidden_dim: int = 128,       # 共享层隐藏维度
        dropout: float = 0.0,
        nmul: int = 1,
        lstm_input_size: int = None,  # LSTM输入维度（xc_nn_norm的维度）
        lstm_hidden_size: int = 64,  # LSTM隐藏层大小
        lstm_dropout: float = 0.0,   # LSTM dropout
        device: str = "cuda:0",
    ):
        super().__init__()

        self.nmul = nmul
        self.lstm_hidden_size = lstm_hidden_size

        # 如果未指定LSTM输入维度，使用默认值
        if lstm_input_size is None:
            lstm_input_size = 3 + input_dim

        self.lstm_input_size = lstm_input_size

        # MOPEX 模型的输出头（不包含weights，因为weights由LSTM动态生成）
        self.num_params_dict: dict = {
            "params": 12 * nmul,      # MOPEX物理参数
            "gamma_uh": 2,            # 路由参数
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

        # 2. 独立参数头 (Independent Heads) - 仅输出params和gamma_uh
        self.heads = nn.ModuleDict()
        for head_name, n_params in self.num_params_dict.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, n_params)
            )

        # 3. LSTM权重预测器 - 参考 LstmMlpModel 的写法
        # 使用 Sequential 包装 Linear + ReLU + LSTM
        self.weight_lstm = nn.Sequential(
            nn.Linear(lstm_input_size, lstm_hidden_size),
            nn.ReLU(),
            nn.LSTM(lstm_hidden_size, lstm_hidden_size, dropout=lstm_dropout, batch_first=False),
        )

        # LSTM输出层：预测每个权重的2个logits (Off/On)
        # self.weight_fc = nn.Linear(lstm_hidden_size, 4 * 2)  # 4个过程，每个2个状态 for softmax
        self.weight_fc = nn.Linear(lstm_hidden_size, 4)  # 4个过程，每个2个状态 for sigmoid

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

        # 初始化LSTM权重输出层
        nn.init.normal_(self.weight_fc.weight, mean=0.0, std=0.001)
        if self.weight_fc.bias is not None:
            nn.init.constant_(self.weight_fc.bias, 0.0)

        print("[INFO] MultiHeadNetDyn weights initialized successfully")

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cuda:0"):
        # 参考 LstmMlpModel，直接使用 config["nx"] 作为 LSTM 输入维度
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
                "params": [Batch, 12*nmul],      # MOPEX物理参数
                "weights": [n_steps, Batch, 4, 2],  # 动态权重logits
                "gamma_uh": [Batch, 2],          # 路由参数
            }
        """
        # 1. 提取共享特征（静态属性）
        x_attr = x["c_nn_norm"]
        shared_feat = self.backbone(x_attr)

        # 2. 各头独立输出（params和gamma_uh）
        out_dict = {}
        for head_name, head_net in self.heads.items():
            out_dict[head_name] = head_net(shared_feat)

        # 3. LSTM预测动态权重 - 参考 LstmMlpModel 的写法
        z1 = x["xc_nn_norm"]  # [n_steps, batch, lstm_input_size]

        # LSTM前向传播（Sequential会自动处理）
        lstm_out, _ = self.weight_lstm(z1)  # [n_steps, batch, hidden_size]

        # 通过全连接层预测权重logits
        weights_logits = self.weight_fc(lstm_out)  # [n_steps, batch, 8]
        weights_logits = weights_logits.view(
            weights_logits.shape[0], weights_logits.shape[1], 4, 2
        )  # [n_steps, batch, 4, 2]

        out_dict["weights"] = weights_logits

        return out_dict