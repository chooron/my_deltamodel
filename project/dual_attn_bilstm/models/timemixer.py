import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.TimeMixer_Layers import PastDecomposableMixing

# 复用原本的 PDM 模块 (不需要改动)


class TimeMixer(nn.Module):
    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        # === 1. 维度定义 ===
        # 动态特征数 (P, T, PET等)
        self.dynamic_dim = configs.dynamic_input_dim
        # 静态特征数 (Area, Soil, Slope等)
        self.static_dim = configs.static_input_dim
        # 融合后的总输入特征数
        self.total_input_dim = self.dynamic_dim + self.static_dim

        # dPL 参数设置
        self.nmul = configs.nmul  # 总专家数 (e.g., 16)
        self.num_params = configs.num_params  # 物理模型参数量 (e.g., HBV=9)
        self.down_sampling_layers = configs.down_sampling_layers  # 下采样层数

        # 计算每个尺度分配多少个专家
        # Scale 0 (原始), Scale 1 (1/2), Scale 2 (1/4)...
        self.num_scales = self.down_sampling_layers + 1
        self.experts_per_scale = self.nmul // self.num_scales
        # 把除不尽的余数分给 Scale 0 (高频专家)
        self.experts_s0 = self.nmul - (
            self.experts_per_scale * (self.num_scales - 1)
        )

        # === 2. 核心骨干 (PDM Blocks) ===
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )

        self.preprocess = series_decomp(configs.moving_avg)

        # Embedding 层
        # 注意: 输入维度变成了 total_input_dim
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                self.total_input_dim,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )

        self.normalize_layers = torch.nn.ModuleList(
            [
                # 注意这里使用的是 total_input_dim
                Normalize(
                    self.total_input_dim,
                    affine=True,
                    non_norm=True if configs.use_norm == 0 else False,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # === 3. 参数生成头 (Projection Heads) ===
        # 为每个尺度定义一个映射头
        self.param_heads = torch.nn.ModuleList()

        for i in range(self.num_scales):
            n_exp = self.experts_s0 if i == 0 else self.experts_per_scale

            # 输入: d_model (特征维度)
            # 输出: n_exp * num_params (该尺度下所有专家的参数展平)
            self.param_heads.append(
                nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_ff),
                    nn.GELU(),
                    nn.Linear(configs.d_ff, n_exp * self.num_params),
                )
            )

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # 这是一个辅助函数，用于生成多尺度的输入
        # 逻辑与原版 TimeMixer 保持一致
        if self.configs.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(
                self.configs.down_sampling_window, return_indices=False
            )
        elif self.configs.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            # 注意: 这里 in_channels 需要是 total_input_dim
            down_pool = nn.Conv1d(
                in_channels=self.total_input_dim,
                out_channels=self.total_input_dim,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc

        x_enc = x_enc.permute(0, 2, 1)  # [B, C, T]
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []

        # Scale 0
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        # Generate Scale 1, 2, ...
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[
                        :, :: self.configs.down_sampling_window, :
                    ]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[
                    :, :: self.configs.down_sampling_window, :
                ]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc  # 如果没有 mask 就不处理

        return x_enc, x_mark_enc

    def forward(self, x_dict, x_mark_enc=None, mask=None):
        """
        x_dict: {
            'dynamic': [Batch, Time, Dynamic_Feats],  # 时变数据 (730, D)
            'static':  [Batch, Static_Feats]          # 静态数据 (S)
        }
        """
        # === Step 1: 数据解析与融合 (Early Fusion) ===
        x_dynamic = x_dict["dynamic"]  # [B, T, D]
        x_static = x_dict["static"]  # [B, S]

        B, T, _ = x_dynamic.shape

        # 将静态数据在时间维度复制
        # [B, S] -> [B, 1, S] -> [B, T, S]
        x_static_expanded = x_static.unsqueeze(1).repeat(1, T, 1)

        # 拼接: [B, T, D+S]
        # 这里的顺序很重要，通常把静态放在后面
        x_combined = torch.cat([x_dynamic, x_static_expanded], dim=2)

        # === Step 2: 多尺度分解 ===
        # 将融合后的数据分解为 Scale 0, Scale 1, Scale 2...
        # x_enc_list 是一个列表，包含不同长度的 Tensor
        x_enc_list, x_mark_list = self.__multi_scale_process_inputs(
            x_combined, x_mark_enc
        )

        # === Step 3: Embedding & Norm ===
        enc_out_list = []
        for i, x, x_mark in zip(
            range(len(x_enc_list)), x_enc_list, x_mark_list
        ):
            # 归一化
            x = self.normalize_layers[i](x, "norm")

            # Embedding
            if x_mark is None:
                enc_out = self.enc_embedding(x, None)
            else:
                enc_out = self.enc_embedding(x, x_mark)
            enc_out_list.append(enc_out)

        # === Step 4: TimeMixer Encoding (特征提取) ===
        # PDM Blocks 会混合时间和通道信息
        for i in range(self.configs.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # === Step 5: 参数生成与上采样 (关键步骤) ===
        all_scale_params = []

        for i, enc_out in enumerate(enc_out_list):
            # enc_out: [Batch, Time_Scaled, D_model]
            # Time_Scaled: Scale 0 是 730, Scale 1 是 365, Scale 2 是 182...

            # A. 映射到参数空间
            # 输出维度: [Batch, Time_Scaled, n_exp * num_params]
            params_raw = self.param_heads[i](enc_out)

            # B. 维度变换准备插值 [Batch, Channels, Time_Scaled]
            params_permuted = params_raw.permute(0, 2, 1)

            # C. 强制上采样 (Upsampling) 回原始长度 730
            if i > 0:
                # 使用线性插值。这天然保证了低频专家的参数曲线是平滑的
                params_interpolated = F.interpolate(
                    params_permuted,
                    size=self.seq_len,
                    mode="linear",
                    align_corners=False,  # 保持相位对其
                )
            else:
                # Scale 0 不需要插值
                params_interpolated = params_permuted

            # D. 变换回 [Batch, Time, Channels]
            # [Batch, 730, n_exp * num_params]
            params_reshaped = params_interpolated.permute(0, 2, 1)

            all_scale_params.append(params_reshaped)

        # === Step 6: 拼接输出 ===
        # 我们不求平均，而是把所有专家的参数拼接在一起
        # final_params: [Batch, 730, Total_Experts * Num_Params]
        # 例如: 16个专家 * 9个参数 = 最后一维是 144
        final_params = torch.cat(all_scale_params, dim=2)

        # 如果需要 reshape 成 [Batch, Time, nmul, num_params] 方便后续处理
        # 可以在这里做，或者在外部做
        final_params_structured = final_params.view(
            B, T, self.nmul, self.num_params
        )

        return final_params_structured
