import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def interflow_1(p1, S, Smax, flux, nearzero=1e-6):
    """
    """
    Smax_safe = torch.clamp(Smax, min=1.0)
    ratio = torch.clamp(S / Smax_safe, max=1.0)
    return p1 * ratio * flux


def interflow_2(
    p1: torch.Tensor, S: torch.Tensor, p2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Non-linear interflow. f <= S
    """
    out = p1 * (S + nearzero).pow(1.0 + p2)
    return torch.minimum(out, S)


def interflow_3(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Non-linear interflow (variant). f <= S
    """
    out = p1 * (S + nearzero).pow(p2)
    return torch.minimum(out, S)


def interflow_4(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Combined linear and scaled quadratic interflow. f <= S
    """
    out = p1 * S + p2 * S.pow(2)
    return torch.minimum(S, out)


def interflow_5(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Linear interflow.
    """
    return p1 * S


def interflow_6(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Scaled linear interflow if a storage in the receiving store exceeds a threshold.
    """
    s2_rel = S2 / (S2max + nearzero)
    term1 = (torch.minimum(torch.ones_like(s2_rel), s2_rel) - p2) / (
        1.0 - p2 + nearzero
    )
    term2 = smooth_threshold_storage_logistic(s2_rel, p2, nearzero=nearzero)
    return p1 * S1 * term1 * term2


def interflow_7(
    S: torch.Tensor,
    Smax: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Non-linear interflow if storage exceeds a threshold.
    """
    excess = F.relu(S - p1 * Smax)
    out = (excess / (p2 + nearzero) + nearzero).pow(1.0 / (p3 + nearzero))
    return torch.minimum(excess, out)


def interflow_8(
    S: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Linear interflow if storage exceeds a threshold.
    """
    return F.relu(p1 * (S - p2))


def interflow_9(
    S: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Non-linear interflow if storage exceeds a threshold.
    """
    excess = F.relu(S - p2)
    out = (p1 * excess + nearzero).pow(p3)
    return torch.minimum(excess, out)


def interflow_10(
    S: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Scaled linear interflow if storage exceeds a threshold.
    """
    return p1 * F.relu(S - p2) / (p3 + nearzero)


def interflow_11(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Constant interflow if storage exceeds a threshold.
    """
    excess = F.relu(S - p2)
    out = torch.minimum(p1, excess)
    return out * smooth_threshold_storage_logistic(S, p2, nearzero=nearzero)


def interflow_12(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Non-linear interflow (variant) when current storage is over
    a threshold (FC) and zero otherwise.
    """
    fc = p2 * Smax
    excess = F.relu(S - fc)
    out = torch.minimum(p1 * (excess + nearzero).pow(p3), S)
    return out * smooth_threshold_storage_logistic(S, fc, nearzero=nearzero)
