import torch
import torch.nn.functional as F


def exchange_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    S: torch.Tensor,
    fmax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Two-way channel exchange: linear and exponential.
    Formula: out = (p1 * |S| + p2 * (1 - exp(-p3 * |S|))) * sign(S)
    Constraint: out >= -fmax
    Note: dt is assumed to be 1.0.
    """
    s_abs = torch.abs(S)
    flow = (p1 * s_abs + p2 * (1.0 - torch.exp(-p3 * s_abs))) * torch.sign(S)
    return torch.maximum(flow, -fmax)


def exchange_2(
    p1: torch.Tensor,
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Water exchange based on relative storages.
    Formula: out = p1 * (S1/S1max - S2/S2max)
    """
    ratio1 = S1 / (S1max + nearzero)
    ratio2 = S2 / (S2max + nearzero)
    return p1 * (ratio1 - ratio2)


def exchange_3(
    p1: torch.Tensor, S: torch.Tensor, p2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Water exchange with infinite size store based on threshold.
    Formula: out = p1 * (S - p2)
    """
    return p1 * (S - p2)
