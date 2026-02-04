import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def baseflow_1(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 1: Outflow from a linear reservoir
    Formula: out = p1 * S
    """
    return p1 * S


def baseflow_2(
    S: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 2: Non-linear outflow from a reservoir
    Constraint: f <= S
    Formula: out = (S / p1)^(1 / p2)
    """
    term_flow = (S / (p1 + nearzero)).pow(1.0 / (p2 + nearzero))
    return torch.minimum(term_flow, S)


def baseflow_3(
    S: torch.Tensor, Smax: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 3: Empirical non-linear outflow
    Formula: out = Smax^(-4) / 4 * S^5
    """
    return ((Smax + nearzero).pow(-4.0) / 4.0) * S.pow(5)


def baseflow_4(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 4: Exponential outflow from deficit store
    Formula: out = p1 * exp(-p2 * S)
    """
    return p1 * torch.exp(-p2 * S)


def baseflow_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Baseflow 5: Non-linear scaled outflow
    Constraint: f <= S
    ratio: (S / Smax)^p2
    """
    ratio = S / (Smax + nearzero)
    term_flow = p1 * (ratio + nearzero).pow(p2)
    return torch.minimum(S, term_flow)


def baseflow_6(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 6: Quadratic outflow if storage threshold is exceeded
    """
    q_quadratic = torch.minimum(S, p1 * S.pow(2))

    # sf returns ~1 when S > p2
    sf = smooth_threshold_storage_logistic(S, p2, nearzero=nearzero)
    return q_quadratic * (1.0 - sf)


def baseflow_7(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 7: Non-linear outflow
    Formula: out = min(S, p1 * S^p2)
    """
    term_flow = p1 * (S + nearzero).pow(p2)
    return torch.minimum(S, term_flow)


def baseflow_8(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Baseflow 8: Exponential scaled outflow from deficit store
    Formula: out = p1 * (exp(p2 * min(1, max(S,0)/Smax)) - 1)
    """
    ratio = S / (Smax + nearzero)
    ratio_clamped = torch.clamp(ratio, max=1.0)
    return p1 * (torch.exp(p2 * ratio_clamped) - 1.0)


def baseflow_9(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 9: Linear flow above a threshold
    Formula: out = p1 * max(0, S - p2)
    """
    # Using Softplus for smooth transition
    excess_storage = F.softplus(S - p2, beta=50.0)
    return p1 * excess_storage
