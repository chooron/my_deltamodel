import torch
import torch.nn.functional as F

def recharge_1(p1, S, Smax, flux, nearzero=1e-6):
    """
    Recharge as scaled fraction of incoming flux
    Formula: out = p1 * S / Smax * flux
    """
    Smax_safe = torch.clamp(Smax, min=1.0)
    ratio = torch.clamp(S / Smax_safe, max=1.0)
    return p1 * ratio * flux

def recharge_2(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Recharge as non-linear scaling of incoming flux
    Formula: out = flux * (max(S,0) / Smax)^p1
    """
    ratio = torch.clamp(F.relu(S) / (Smax + nearzero), max=1.5)
    return flux * (ratio + nearzero).pow(p1)


def recharge_3(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Linear recharge
    Formula: out = p1 * S
    """
    return p1 * S


def recharge_4(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Constant recharge
    Formula: out = min(p1, S)
    Note: dt is assumed to be 1.0
    """
    return torch.minimum(p1, S)


def recharge_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    [Safe Gradient Version] Recharge using Softplus.
    
    Formula concept: Flux ~ p1 * S1 * max(0, 1 - S2/p2)
    But using Softplus to keep gradients alive when S2 > p2.
    """
    # 1. 依然保留分母锁，防止 NaN (这是底线)
    p2_safe = torch.clamp(p2, min=1.0)
    
    # 2. 计算"赤字" (Deficit ratio)
    # 我们希望: 当 S2 < p2 时，有流量；当 S2 > p2 时，流量归零。
    # 定义 diff = 1.0 - S2 / p2
    diff = 1.0 - S2 / p2_safe
    
    # 3. 使用 Softplus 代替 Hard Clamp/ReLU
    # Softplus(x) = log(1 + e^x)
    # 当 diff > 0 (未满)时: Softplus 接近线性，梯度 ~ 1
    # 当 diff < 0 (溢出)时: Softplus 接近 0，但梯度 > 0 (不会完全死掉)
    # beta=10.0 控制平滑度，越大越像 ReLU，越小越平滑
    term = F.softplus(diff, beta=10.0)
    
    # 注意：Softplus 的值域是 (0, inf)，而我们原本逻辑是 (0, 1)。
    # 当 diff 很大(极空)时，softplus 会超过 1。
    # 为了物理严谨性，我们可以再加一个 tanh 或者 clamp，
    # 但实际上让补给稍微快一点点通常没问题。为了保险，加个 min(1.0)
    # 但这个 min 要放在最后，且只针对数值，不截断梯度流
    
    # 这里直接用 term 即可，dPL 模型会自动调整 p1 来适应 scale。
    return p1 * S1 * term


def recharge_6(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Recharge to fulfil evaporation demand if the receiving store is below a threshold
    Formula: out = min(max(S,0), p1 * max(S,0)^p2)
    Note: dt is assumed to be 1.0
    """
    s_pos = F.relu(S)
    potential = p1 * (s_pos + nearzero).pow(p2)
    return torch.minimum(s_pos, potential)


def recharge_7(
    p1: torch.Tensor, incoming_flux: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Constant recharge limited by incoming flux
    Formula: out = min(p1, fin)
    """
    return torch.minimum(p1, incoming_flux)


def recharge_8(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    p2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Recharge as non-linear scaling of incoming flux
    Formula: out = min(p2 * (max(S,0)/Smax)^p1, max(S,0))
    Note: dt is assumed to be 1.0
    """
    ratio = F.relu(S) / (Smax + nearzero)
    potential = p2 * (ratio + nearzero).pow(p1)
    return torch.minimum(potential, F.relu(S))
