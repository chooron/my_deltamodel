"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

if tuple(map(int, torch.__version__.split(".")[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = (
                X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            )
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(
        self,
        d_model,
        cfr,
        cfi,
        N=64,
        dt_min=0.0001,
        dt_max=0.1,
        lr=None,
        lr_dt=None,
        wd=None,
    ):
        super().__init__()
        # Generate dt

        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, 0, lr=lr)

        # print("S4D kernel: N = ", N, cfi, cfr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2)) * cfr  # config v1
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H) * cfi
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = (
            A * dt.unsqueeze(-1)
        )  # (H N) discretizing the continuous-time dynamics to generate a discrete-time convolution kernel
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

        return K

    def register(self, name, tensor, wd, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": wd}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        dropout=0.0,
        add_noise=0,
        mult_noise=0,
        cfr=1,
        cfi=1,
        transposed=True,
        use_gated=False,
        out_activation="tanh",
        **kernel_args,
    ):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.add_noise = add_noise
        self.mult_noise = mult_noise
        self.use_gated = use_gated

        # print("s4d.py self.h, self.n: ", self.h, self.n)

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, cfr, cfi, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()  # config v1
        # self.activation = nn.Tanh()  # config v2
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        if self.use_gated:
            self.gate = nn.Conv1d(self.h, self.h, kernel_size=1)
            nn.init.constant_(self.gate.bias, 1.0)  # type: ignore
            nn.init.xavier_uniform_(self.gate.weight)

        # position-wise output transform to mix features
        if out_activation == "glu":
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        elif out_activation == "tanh":
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, self.h, kernel_size=1),
                nn.Tanh(),
            )

    def forward(
        self, u, **kwargs
    ):  # absorbs return_output and transformer src mask
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        ybar = u_f * k_f
        y = torch.fft.irfft(ybar, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if self.use_gated:
            g = torch.sigmoid(self.gate(y))  # (B, H, L)
            y = g * y + (1 - g) * u  # gated residual fusion

        if not self.transposed:
            y = y.transpose(-1, -2)
        return (
            y,
            None,
        )  # Return a dummy state to satisfy this repo's interface, but this can be modified


class Hope(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        n_layers=2,  # 4 -> 3
        dropout=0.1,
        cfg=None,
        output_size=None,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder
        self.encoder = nn.Linear(input_size, hidden_size)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if cfg is None:
            # config v1
            # cfg = {"lr_min": 0.001, "lr": 0.01, "lr_dt": 0.0, "min_dt": 0.001,
            #        "max_dt": 1, "wd": 0.0, "d_state": 64, "cfr": 1.0, "cfi": 1.0}
            # config v2
            cfg = {
                "lr_min": 0.001,
                "lr": 0.01,
                "lr_dt": 0.0,
                "min_dt": 0.001,
                "max_dt": 0.01,  # change to 0.01
                "wd": 0.01,  # change to 0.01
                "d_state": 64,
                "cfr": 1.0,
                "cfi": 1.0,
                "use_gated": True,
                "out_activation": "tanh",
            }

        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    hidden_size,
                    dropout=dropout,
                    transposed=True,
                    lr=min(cfg["lr_min"], cfg["lr"]),
                    d_state=cfg["d_state"],
                    dt_min=cfg["min_dt"],
                    dt_max=cfg["max_dt"],
                    lr_dt=cfg["lr_dt"],
                    cfr=cfg["cfr"],
                    cfi=cfg["cfi"],
                    wd=cfg["wd"],
                    out_activation=cfg["out_activation"],
                    use_gated=cfg["use_gated"],
                )
            )
            # self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        if output_size is None:
            self.decoder = nn.Identity()
        else:
            self.decoder = nn.Linear(hidden_size, output_size)
        # self.lnorm = torch.nn.LayerNorm(365)
        # self.att = SelfAttentionLayer(365) #20240626 remove attention

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # x = torch.cat((x, torch.zeros((x.shape[0],1,x.shape[2])).to('cuda')), 1 )
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        # x = torch.cat((x,torch.flip(x,dims=[-1])),dim=-1) # bi-directional

        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts
        ):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x

            # z, _ = self.att(z)

            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)
        x = x.transpose(-1, -2)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


class HopeV2(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        n_layers=2,
        dropout=0.1,
        cfg=None,
        output_size=None,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # 1. Linear encoder
        self.encoder = nn.Linear(input_size, hidden_size)

        # 2. Stack S4 layers
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # 定义 Dropout 函数
        dropout_fn = lambda d: nn.Dropout(d)

        if cfg is None:
            cfg = {
                "lr_min": 0.001,
                "lr": 0.01,
                "lr_dt": 0.0,
                "min_dt": 0.001,
                "max_dt": 0.01,
                "wd": 0.01,
                "d_state": 64,
                "cfr": 1.0,
                "cfi": 1.0,
                "out_activation": "tanh",
            }

        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    hidden_size,
                    dropout=dropout,
                    transposed=True,  # 意味着 S4D 期望 (B, Hidden, Length)
                    lr=min(cfg["lr_min"], cfg["lr"]),
                    d_state=cfg["d_state"],
                    dt_min=cfg["min_dt"],
                    dt_max=cfg["max_dt"],
                    lr_dt=cfg["lr_dt"],
                    cfr=cfg["cfr"],
                    cfi=cfg["cfi"],
                    wd=cfg["wd"],
                    out_activation=cfg["out_activation"],
                )
            )
            # 【关键修改 1】使用 LayerNorm 替代 BatchNorm1d
            # LayerNorm 更加适合长序列回归，避免破坏时间动态性
            self.norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(dropout_fn(dropout))

        # 【关键修改 2】平滑层 (Conv1d)
        self.smoother = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=5,  # 如果仍然震荡，可以尝试增大到 15 或 25
            padding=2,  # (kernel_size - 1) // 2
            groups=hidden_size,  # 通道独立
            bias=False,
        )

        # 【关键修改 3】强制初始化为均值平滑
        with torch.no_grad():
            self.smoother.weight.fill_(1.0 / 7)

        # Linear decoder
        if output_size is None:
            self.decoder = nn.Identity()
        else:
            self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        # 1. Encoder: (B, L, Input) -> (B, L, Hidden)
        x = self.encoder(x)

        # 2. Transpose: (B, L, Hidden) -> (B, Hidden, L)
        # S4D 和 Conv1d 都需要 Hidden 在中间
        x = x.transpose(-1, -2)

        # 3. Main Loop
        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts
        ):
            z = x  # z shape: (B, Hidden, L)

            # --- PreNorm 逻辑 ---
            if self.prenorm:
                # LayerNorm 需要 (B, L, Hidden)，所以必须转置
                z = z.transpose(-1, -2)  # (B, L, Hidden)
                z = norm(z)
                z = z.transpose(-1, -2)  # 转回 (B, Hidden, L) 给 S4

            # Apply S4 block (Expects B, Hidden, L)
            z, _ = layer(z)

            # Dropout
            z = dropout(z)

            # Residual connection
            x = z + x

            # --- PostNorm 逻辑 (通常默认走这里) ---
            if not self.prenorm:
                # LayerNorm 需要 (B, L, Hidden)，必须转置
                x = x.transpose(-1, -2)  # (B, L, Hidden)
                x = norm(x)
                x = x.transpose(-1, -2)  # 转回 (B, Hidden, L) 保持循环一致性

        # 4. Apply Smoother (Conv1d)
        # 此时 x 是 (B, Hidden, L)，正是 Conv1d 想要的
        x = self.smoother(x)

        # 5. Transpose back for Decoder
        # Decoder (Linear) 需要 (B, L, Hidden)
        x = x.transpose(-1, -2)

        # 6. Decoder
        x = self.decoder(x)  # (B, L, Output)

        return x


class HopeV3(nn.Module):
    def __init__(self, input_size, hidden_size=256, n_layers=2, dropout=0.1, cfg=None, output_size=None, prenorm=False):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(input_size, hidden_size)

        # 核心层列表
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 【新加】层内卷积列表
        self.conv_layers = nn.ModuleList() 

        dropout_fn = lambda d: nn.Dropout(d)
        
        for _ in range(n_layers):
            # 1. 局部特征提取 (Local Context)
            # 使用 depthwise convolution 节省参数，且不改变通道数
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size),
                nn.SiLU() # 或者 GELU
            ))
            
            
            if cfg is None:
                cfg = {
                    "lr_min": 0.001,
                    "lr": 0.01,
                    "lr_dt": 0.0,
                    "min_dt": 0.001,
                    "max_dt": 0.01,
                    "wd": 0.01,
                    "d_state": 64,
                    "cfr": 1.0,
                    "cfi": 1.0,
                    "out_activation": "tanh",
                }
            
            # 2. 全局特征提取 (Global Context)
            self.s4_layers.append(
                S4D(
                    hidden_size,
                    dropout=dropout,
                    transposed=True,  # 意味着 S4D 期望 (B, Hidden, Length)
                    lr=min(cfg["lr_min"], cfg["lr"]),
                    d_state=cfg["d_state"],
                    dt_min=cfg["min_dt"],
                    dt_max=cfg["max_dt"],
                    lr_dt=cfg["lr_dt"],
                    cfr=cfg["cfr"],
                    cfi=cfg["cfi"],
                    wd=cfg["wd"],
                    out_activation=cfg["out_activation"],
                )
            )
            
            self.norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(dropout_fn(dropout))

        if output_size is None:
            self.decoder = nn.Identity()
        else:
            self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, L, Input)
        x = self.encoder(x)
        x = x.transpose(-1, -2) # (B, Hidden, L)

        for conv, layer, norm, dropout in zip(self.conv_layers, self.s4_layers, self.norms, self.dropouts):
            residual = x
            z = x
            
            # --- PreNorm (推荐) ---
            if self.prenorm:
                z = z.transpose(-1, -2)
                z = norm(z)
                z = z.transpose(-1, -2)

            # --- 1. 先做局部卷积 (Local Conv) ---
            # 这替代了你原本最后那个巨大的 Smoother
            # 让网络自己在每一层学习是否需要平滑，而不是强制平滑
            z = conv(z) 

            # --- 2. 再做 S4D (Global SSM) ---
            z, _ = layer(z)
            
            # Dropout
            z = dropout(z)
            
            # Residual
            x = z + residual
            
            # --- PostNorm ---
            if not self.prenorm:
                x = x.transpose(-1, -2)
                x = norm(x)
                x = x.transpose(-1, -2)

        x = x.transpose(-1, -2) # Back to (B, L, Hidden)
        x = self.decoder(x)
        return x