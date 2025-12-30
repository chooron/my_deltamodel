import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def recharge_1(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Recharge as scaled fraction of incoming flux
    Formula: out = p1 * S / Smax * flux
    """
    return p1 * S / (Smax + nearzero) * flux


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
    ratio = F.relu(S) / (Smax + nearzero)
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
    Recharge to fulfil evaporation demand if the receiving store is below a threshold
    Formula: out = p1 * S1 * (1 - min(1, S2 / p2))
    Smoothed for dPL using smooth_threshold_storage_logistic.
    """
    # sf returns ~1 when S2 > p2, so (1-sf) is the gate that closes as S2 approaches p2
    sf = smooth_threshold_storage_logistic(S2, p2, nearzero=nearzero)
    return p1 * S1 * (1.0 - sf)


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
