"""
EMTSF (Extraordinary Mixture of SOTA Models for Time Series Forecasting) 架构实现

本模块实现了基于 Transformer 的门控网络 (Gating Network)，用于动态加权多个专家模型的预测输出。

核心思想：
-----------
EMTSF 使用 Transformer 作为门控网络，而非传统 MoE 中的简单线性网络。
这使得模型能够在每个时间步上动态调整每个专家的贡献权重。

权重计算公式：
-----------
    g_i(x) = Softmax(MA_k(G(cat(TSFModel_i(x)))))_i    ... (1)
    
    其中:
    - G: 基于 Transformer 的门控网络
    - MA_k: 移动平均，用于平滑权重
    - Softmax: 归一化为概率分布

最终输出：
-----------
    output = Σ g_i(x) * TSFModel_i(x)    ... (2)

参考文献：
-----------
EMTSF: Extraordinary Mixture of SOTA Models for Time Series Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    相比 LayerNorm，RMSNorm 省略了均值中心化步骤，计算更高效。
    
    公式: x_norm = x / sqrt(mean(x^2) + eps) * gamma
    
    Parameters
    ----------
    dim : int
        输入特征维度
    eps : float
        防止除零的小常数，默认 1e-8
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (..., dim)
            
        Returns
        -------
        torch.Tensor
            归一化后的张量，形状与输入相同
        """
        # 计算 RMS (Root Mean Square)
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.gamma


class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码 (Sinusoidal Positional Encoding)
    
    为 Transformer 提供序列位置信息，使用正弦和余弦函数生成固定的位置编码。
    
    公式:
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    Parameters
    ----------
    embedding_dim : int
        嵌入维度
    max_seq_length : int
        最大序列长度，默认 512
    dropout : float
        Dropout 概率，默认 0.1
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                if i + 1 < embedding_dim:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * i + 1) / embedding_dim))
                    )
        # 添加 batch 维度: (1, max_seq_length, embedding_dim)
        pe = pe.unsqueeze(0)
        # 注册为 buffer（不参与梯度更新，但会随模型保存）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (batch, seq_len, embedding_dim)
            
        Returns
        -------
        torch.Tensor
            添加位置编码后的张量
        """
        # 缩放输入（标准 Transformer 做法）
        x = x * math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        # 截取对应长度的位置编码
        pe = self.pe[:, :seq_length].to(x.device)
        x = x + pe
        return self.dropout(x)


class MHSelfAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    
    实现标准的缩放点积注意力，支持因果掩码（causal masking）。
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Parameters
    ----------
    dim : int
        输入/输出维度
    heads : int
        注意力头数，默认 8
    dim_head : int, optional
        每个头的维度，默认为 dim // heads
    causal : bool
        是否使用因果掩码（防止看到未来信息），默认 True
    """
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = None, causal: bool = True):
        super().__init__()
        self.dim_head = (dim // heads) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.causal = causal
        
        # QKV 投影（合并为单个线性层提高效率）
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False)
        # 输出投影
        self.W_out = nn.Linear(_dim, dim, bias=False)
        # 缩放因子: 1/sqrt(d_k)
        self.scale_factor = self.dim_head**-0.5

    def set_causal(self, causal: bool):
        """动态设置是否使用因果掩码"""
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (batch, seq_len, dim)
            
        Returns
        -------
        torch.Tensor
            注意力输出，形状为 (batch, seq_len, dim)
        """
        # 计算 Q, K, V
        qkv = self.to_qkv(x)
        # 重排为 (3, batch, heads, seq_len, dim_head)
        q, k, v = tuple(
            rearrange(qkv, "b n (d k h) -> k b h n d", k=3, h=self.heads)
        )
        
        # 计算缩放点积注意力分数
        # einsum: (b h i d) @ (b h j d) -> (b h i j)
        scaled_dot_prod = (
            torch.einsum("b h i d, b h j d -> b h i j", q, k)
            * self.scale_factor
        )
        
        i, j = scaled_dot_prod.shape[2], scaled_dot_prod.shape[3]
        
        # 应用因果掩码（上三角矩阵设为 -inf）
        if self.causal:
            causal_mask = (
                torch.ones(i, j, device=x.device).triu_(j - i + 1).bool()
            )
            scaled_dot_prod = scaled_dot_prod.masked_fill(
                causal_mask, float("-inf")
            )
        
        # Softmax 归一化
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        
        # 加权求和
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        
        # 合并多头
        out = rearrange(out, "b h n d -> b n (h d)")
        
        return self.W_out(out)


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    
    包含多头自注意力层和前馈网络层，使用 Pre-Norm 结构。
    
    结构: x -> MHSA -> Add & Norm -> FFN -> Add & Norm
    
    Parameters
    ----------
    dim : int
        输入/输出维度
    heads : int
        注意力头数，默认 8
    dim_head : int, optional
        每个头的维度
    causal : bool
        是否使用因果掩码，默认 False
    dim_linear_block : int
        前馈网络隐藏层维度，默认 512
    dropout : float
        Dropout 概率，默认 0.1
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = None,
        causal: bool = False,
        dim_linear_block: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 多头自注意力层
        self.mhsa = MHSelfAttention(dim, heads, dim_head, causal)
        self.drop = nn.Dropout(dropout)
        # RMSNorm 归一化层
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        # 前馈网络 (FFN)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (batch, seq_len, dim)
            
        Returns
        -------
        torch.Tensor
            输出张量，形状与输入相同
        """
        # 自注意力 + 残差连接 + 归一化
        y = self.norm1(x + self.drop(self.mhsa(x)))
        # 前馈网络 + 残差连接 + 归一化
        return self.norm2(y + self.linear(y))


class SimpleTransformer(nn.Module):
    """
    门控网络的核心 Transformer 模块
    
    这是 EMTSF 架构中的关键组件，用于学习专家模型输出之间的时序依赖关系，
    并生成每个时间步的门控系数。
    
    输入处理流程:
        1. 将所有专家的输出拼接
        2. 通过线性层投影到嵌入空间
        3. 添加位置编码
        4. 经过多层 Transformer 块处理
        5. 投影回原始空间
    
    Parameters
    ----------
    dim : int
        Transformer 隐藏维度
    enc_in : int
        每个专家输出的特征数
    num_experts : int
        专家模型数量
    num_layers : int
        Transformer 层数，默认 3
    heads : int
        注意力头数，默认 4
    dim_head : int, optional
        每个头的维度
    max_seq_len : int
        最大序列长度，默认 512
    causal : bool
        是否使用因果注意力，默认 True
    """
    
    def __init__(
        self,
        dim: int,
        enc_in: int,
        num_experts: int,
        num_layers: int = 3,
        heads: int = 4,
        dim_head: int = None,
        max_seq_len: int = 512,
        causal: bool = True,
    ):
        super().__init__()
        # 位置编码
        self.pos_enc = PositionalEncoding(dim, max_seq_len)
        # 多层 Transformer 块
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, causal)
                for _ in range(num_layers)
            ]
        )
        # 输入投影: (enc_in * num_experts) -> dim
        # 将所有专家的输出拼接后投影到 Transformer 的隐藏维度
        self.linear_in = nn.Linear(enc_in * num_experts, dim)
        # 输出投影: dim -> (enc_in * num_experts)
        # 为每个专家的每个特征生成门控分数
        self.linear_out = nn.Linear(dim, enc_in * num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            专家输出张量，形状为 (Batch, Feature, NumExperts, Seq)
            
        Returns
        -------
        torch.Tensor
            门控 logits，形状为 (Batch, Feature, NumExperts, Seq)
        """
        # x: (B, F, E, S) - Batch, Feature, Experts, Sequence
        B, Fe, E, S = x.shape
        
        # Step 1: 重排并拼接专家输出
        # (B, F, E, S) -> (B, S, F, E) -> (B, S, F*E)
        x = x.permute(0, 3, 1, 2).reshape(B, S, Fe * E)
        
        # Step 2: 投影到 Transformer 隐藏空间
        x = self.linear_in(x)  # (B, S, dim)
        
        # Step 3: 添加位置编码
        x = self.pos_enc(x)
        
        # Step 4: 通过 Transformer 层
        for layer in self.layers:
            x = layer(x)
        
        # Step 5: 投影回原始空间
        x = self.linear_out(x)  # (B, S, F*E)
        
        # 重排回原始形状
        return x.reshape(B, S, Fe, E).permute(0, 2, 3, 1)  # (B, F, E, S)


class MovingAvg(nn.Module):
    """
    移动平均层 (Moving Average)
    
    对门控系数进行时序平滑，减少权重的剧烈波动，使专家混合更加平稳。
    
    这对应 EMTSF 公式中的 MA_k 操作:
        g_i(x) = Softmax(MA_k(G(...)))
    
    Parameters
    ----------
    kernel_size : int
        移动平均窗口大小，应满足 k << T（远小于序列长度）
    stride : int
        步长，默认 1
    """
    
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        # 使用 1D 平均池化实现移动平均
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (Batch, Feature, NumExperts, Seq)
            
        Returns
        -------
        torch.Tensor
            平滑后的张量，形状与输入相同
        """
        B, Fe, E, S = x.shape
        # 确保张量连续，然后重排为 AvgPool1d 需要的形状: (N, C, L)
        # 使用 reshape 替代 view，因为输入可能不是连续的
        x = x.contiguous().reshape(B * Fe * E, 1, S)
        x = self.avg(x)
        # 处理边界情况：池化可能改变序列长度
        if x.shape[-1] != S:
            x = x[:, :, :S]
        return x.reshape(B, Fe, E, S)


class MoeLayer(nn.Module):
    """
    EMTSF 混合专家层 (Mixture of Experts Layer)
    
    这是 EMTSF 架构的核心模块，实现基于 Transformer 的动态门控机制。
    
    核心思想：
    ---------
    传统 MoE 使用简单线性网络生成所有时间步的固定权重，
    而 EMTSF 使用 Transformer 门控网络，能够：
    1. 捕捉专家输出之间的时序依赖关系
    2. 在每个时间步动态调整专家权重
    3. 根据数据特征自适应选择最优专家组合
    
    计算流程：
    ---------
    1. 接收所有专家模型的预测输出: {TSFModel_1(x), ..., TSFModel_n(x)}
    2. 拼接专家输出并送入 Transformer 门控网络 G
    3. 对门控输出应用移动平均 MA_k 进行平滑
    4. 通过 Softmax 归一化得到权重分布 g_i(x)
    
    公式：
    -----
        g_i(x) = Softmax(MA_k(G(cat(TSFModel_i(x)))))_i
    
    Parameters
    ----------
    enc_in : int
        每个专家输出的特征维度
    num_experts : int
        专家模型数量
    target_points : int
        预测序列长度（时间步数）
    embed_dim : int
        Transformer 嵌入维度，默认 128
    smoothing_k : int
        移动平均窗口大小，默认 5。应满足 k << target_points
    num_layers : int
        Transformer 层数，默认 2
    num_heads : int
        注意力头数，默认 4
    causal : bool
        是否使用因果注意力（防止看到未来），默认 True
    
    Attributes
    ----------
    transformer_gating : SimpleTransformer
        基于 Transformer 的门控网络
    moving_avg : MovingAvg
        移动平均层，用于平滑门控系数
    
    Examples
    --------
    >>> # 创建 MoE 层：4 个专家，每个输出 1 个特征，序列长度 730
    >>> moe = MoeLayer(enc_in=1, num_experts=4, target_points=730)
    >>> 
    >>> # 专家输出: (Seq=730, Batch=32, NumExperts=4, Feature=1)
    >>> experts_outputs = torch.randn(730, 32, 4, 1)
    >>> 
    >>> # 获取门控权重
    >>> weights = moe(experts_outputs)  # (730, 32, 4, 1)
    >>> 
    >>> # 加权求和得到最终输出
    >>> final_output = (weights * experts_outputs).sum(dim=2)  # (730, 32, 1)
    """

    def __init__(
        self,
        enc_in: int,
        num_experts: int,
        target_points: int,
        embed_dim: int = 128,
        smoothing_k: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        causal: bool = True,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.target_points = target_points

        # 基于 Transformer 的门控网络
        # 这是 EMTSF 的核心创新：用 Transformer 替代传统的简单线性门控
        self.transformer_gating = SimpleTransformer(
            dim=embed_dim,
            enc_in=enc_in,
            num_experts=num_experts,
            num_layers=num_layers,
            heads=num_heads,
            max_seq_len=target_points,
            causal=causal,
        )
        
        # 移动平均层，用于平滑门控系数
        # 使权重在相邻时间步之间过渡更加平滑
        self.moving_avg = MovingAvg(kernel_size=smoothing_k)

    def forward(self, experts_outputs: torch.Tensor) -> torch.Tensor:
        """
        计算每个时间步上各专家的动态权重。
        
        Parameters
        ----------
        experts_outputs : torch.Tensor
            所有专家模型的预测输出
            形状: (Seq, Batch, NumExperts, Feature)
            
        Returns
        -------
        torch.Tensor
            归一化的门控权重，可直接与专家输出相乘
            形状: (Seq, Batch, NumExperts, 1)
            
        Notes
        -----
        返回的权重在 NumExperts 维度上和为 1（通过 Softmax 归一化）。
        最后一个维度为 1，便于与专家输出进行广播乘法。
        
        使用示例:
            final_output = (weights * experts_outputs).sum(dim=2)
        """
        S, B, E, Fe = experts_outputs.shape

        # Step 1: 转换维度以适配 Transformer 输入
        # (S, B, E, F) -> (B, E, S, F) -> (B, F, E, S)
        x = experts_outputs.permute(1, 2, 0, 3)  # (B, E, S, F)
        x = x.permute(0, 3, 1, 2)  # (B, F, E, S)

        # Step 2: 通过 Transformer 门控网络计算原始门控 logits
        # G(cat(TSFModel_i(x))) in formula (1)
        gating_logits = self.transformer_gating(x)  # (B, F, E, S)
        
        # Step 3: 移动平均平滑
        # MA_k(...) in formula (1)
        gating_smooth = self.moving_avg(gating_logits)  # (B, F, E, S)
        
        # Step 4: Softmax 归一化得到概率分布
        # Softmax(...) in formula (1)
        # 在专家维度 (dim=2) 上进行 softmax，确保权重和为 1
        gating_weights = F.softmax(gating_smooth, dim=2)  # (B, F, E, S)

        # Step 5: 重排维度并在特征维度上求平均
        # 得到每个专家在每个时间步的综合权重
        # (B, F, E, S) -> (S, B, E, F) -> (S, B, E, 1)
        gating_weights_for_sum = gating_weights.permute(3, 0, 2, 1).mean(
            dim=-1, keepdim=True
        )

        return gating_weights_for_sum

    def forward_with_details(self, experts_outputs: torch.Tensor) -> tuple:
        """
        带详细信息的前向传播，用于分析和可视化。
        
        Parameters
        ----------
        experts_outputs : torch.Tensor
            形状: (Seq, Batch, NumExperts, Feature)
            
        Returns
        -------
        tuple
            (gating_weights, gating_logits, gating_smooth)
            - gating_weights: 最终门控权重 (S, B, E, 1)
            - gating_logits: 原始门控 logits (B, F, E, S)
            - gating_smooth: 平滑后的 logits (B, F, E, S)
        """
        S, B, E, Fe = experts_outputs.shape
        
        x = experts_outputs.permute(1, 2, 0, 3).permute(0, 3, 1, 2)
        gating_logits = self.transformer_gating(x)
        gating_smooth = self.moving_avg(gating_logits)
        gating_weights = F.softmax(gating_smooth, dim=2)
        gating_weights_for_sum = gating_weights.permute(3, 0, 2, 1).mean(
            dim=-1, keepdim=True
        )
        
        return gating_weights_for_sum, gating_logits, gating_smooth


class EMTSFMoE(nn.Module):
    """
    完整的 EMTSF 混合专家模型
    
    这是一个封装类，包含 MoE 门控层和最终输出计算。
    可以直接接收专家输出并返回加权混合后的最终预测。
    
    公式：
        output = Σ g_i(x) * TSFModel_i(x)
    
    Parameters
    ----------
    enc_in : int
        每个专家输出的特征维度
    num_experts : int
        专家模型数量
    target_points : int
        预测序列长度
    embed_dim : int
        Transformer 嵌入维度，默认 128
    smoothing_k : int
        移动平均窗口大小，默认 5
    
    Examples
    --------
    >>> emtsf = EMTSFMoE(enc_in=1, num_experts=4, target_points=730)
    >>> experts_outputs = torch.randn(730, 32, 4, 1)
    >>> final_output, weights = emtsf(experts_outputs)
    >>> print(final_output.shape)  # (730, 32, 1)
    """
    
    def __init__(
        self,
        enc_in: int,
        num_experts: int,
        target_points: int,
        embed_dim: int = 128,
        smoothing_k: int = 5,
    ):
        super().__init__()
        self.moe_layer = MoeLayer(
            enc_in=enc_in,
            num_experts=num_experts,
            target_points=target_points,
            embed_dim=embed_dim,
            smoothing_k=smoothing_k,
        )
    
    def forward(
        self, 
        experts_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        experts_outputs : torch.Tensor
            形状: (Seq, Batch, NumExperts, Feature)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - output: 加权混合后的最终输出 (Seq, Batch, Feature)
            - weights: 门控权重 (Seq, Batch, NumExperts, 1)
        """
        # 获取门控权重
        weights = self.moe_layer(experts_outputs)  # (S, B, E, 1)
        
        # 加权求和: Σ g_i(x) * TSFModel_i(x)
        # (S, B, E, F) * (S, B, E, 1) -> (S, B, E, F) -> sum over E -> (S, B, F)
        output = (weights * experts_outputs).sum(dim=2)
        
        return output, weights


if __name__ == "__main__":
    print("=" * 60)
    print("EMTSF (Mixture of Experts) 模块测试")
    print("=" * 60)
    
    # 测试参数
    S, B, E, Fe = 730, 32, 16, 1  # seq_len, batch, experts, feature
    print(f"\n测试配置:")
    print(f"  - 序列长度 (S): {S}")
    print(f"  - 批次大小 (B): {B}")
    print(f"  - 专家数量 (E): {E}")
    print(f"  - 特征维度 (F): {Fe}")
    
    # 模拟专家模型输出
    x = torch.randn(S, B, E, Fe)
    print(f"\n专家输出形状: {x.shape}")
    
    # 测试 MoeLayer
    print("\n" + "-" * 40)
    print("测试 MoeLayer")
    moe = MoeLayer(
        enc_in=Fe, 
        num_experts=E, 
        target_points=S, 
        smoothing_k=11
    )
    weights = moe(x)
    print(f"  门控权重形状: {weights.shape}")
    print(f"  权重和 (应接近 1.0): {weights[:, 0, :, 0].sum(dim=1).mean().item():.4f}")
    
    # 测试 EMTSFMoE
    print("\n" + "-" * 40)
    print("测试 EMTSFMoE (完整模型)")
    emtsf = EMTSFMoE(
        enc_in=Fe, 
        num_experts=E, 
        target_points=S
    )
    output, weights = emtsf(x)
    print(f"  最终输出形状: {output.shape}")
    print(f"  门控权重形状: {weights.shape}")
    
    # 验证权重归一化
    print("\n" + "-" * 40)
    print("验证权重归一化")
    weight_sums = weights[:, :, :, 0].sum(dim=2)  # 在专家维度求和
    print(f"  权重和均值: {weight_sums.mean().item():.6f}")
    print(f"  权重和标准差: {weight_sums.std().item():.6f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
