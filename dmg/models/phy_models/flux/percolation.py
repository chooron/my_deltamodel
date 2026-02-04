import torch
import torch.nn.functional as F


def percolation_1(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Percolation at a constant rate.
    Formula: out = min(p1, S)
    """
    return torch.minimum(p1, S)


def percolation_2(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Percolation scaled by current relative storage.
    Formula: out = min(S, p1 * S / Smax)
    """
    return torch.minimum(S, p1 * S / (Smax + nearzero))


def percolation_3(
    S: torch.Tensor, Smax: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Non-linear percolation (empirical).
    Formula: out = Smax^(-4) / 4 * (4/9)^4 * S^5
    """
    # Formula components
    # Smax^(-4) / 4
    term1 = (Smax + nearzero).pow(-4.0) / 4.0
    # (4/9)^4
    term2 = (4.0 / 9.0) ** 4.0
    return term1 * term2 * S.pow(5)


def percolation_4(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    p5: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Demand-based percolation scaled by available moisture.
    Formula: out = max(0, min(S, max(0, S/Smax) * (p1 * (1 + p2 * (p4/p5)^(1+p3)))))
    """
    ratio_s = F.relu(S) / (Smax + nearzero)
    ratio_def = F.relu(p4) / (p5 + nearzero)

    demand = ratio_s * (p1 * (1.0 + p2 * (ratio_def + nearzero).pow(1.0 + p3)))

    return F.relu(torch.minimum(S, demand))


def percolation_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Non-linear percolation.
    Formula: out = min(S, p1 * (S/Smax)^p2)
    """
    # Assuming S >= 0 from global constraints
    ratio = F.relu(S) / (Smax + nearzero)
    potential_flow = p1 * (ratio + nearzero).pow(p2)
    return torch.minimum(S, potential_flow)


def percolation_6(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Threshold-based percolation from a store that can reach negative values.
    Formula: out = min(S, p1 * min(1, max(0, S)/p2))
    """
    ratio = F.relu(S) / (p2 + nearzero)
    modifier = torch.minimum(torch.ones_like(S), ratio)
    return torch.minimum(S, p1 * modifier)
