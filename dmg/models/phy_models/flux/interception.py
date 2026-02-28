import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def interception_1(
    incoming_flux: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Interception excess when maximum capacity is reached.
    Formula: out = In * (1 - smooth_threshold_storage_logistic(S, Smax))
    """
    # sf returns ~1 when S > Smax
    sf = smooth_threshold_storage_logistic(S, Smax, nearzero=nearzero)
    return incoming_flux * (1.0 - sf)


def interception_2(
    incoming_flux: torch.Tensor, p1: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Interception excess after a constant amount is intercepted.
    Formula: out = max(In - p1, 0)
    """
    return F.relu(incoming_flux - p1)


def interception_3(
    p1: torch.Tensor, incoming_flux: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Interception excess after a fraction is intercepted.
    Formula: out = p1 * In
    """
    return p1 * incoming_flux


def interception_4(
    p1: torch.Tensor,
    p2: torch.Tensor,
    t: torch.Tensor,
    tmax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Interception excess after a time-varying fraction is intercepted.
    Formula: out = max(0, p1 + (1-p1) * cos(2*pi*(t - p2) / tmax)) * In
    Note: dt is assumed to be 1.0.
    """
    # cos argument: 2*pi*(t - p2) / tmax
    angle = 2.0 * torch.pi * (t - p2) / (tmax + nearzero)
    fraction = p1 + (1.0 - p1) * torch.cos(angle)
    return F.relu(fraction) * incoming_flux


def interception_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Interception excess after a combined absolute amount and fraction are intercepted.
    Formula: out = max(p1 * In - p2, 0)
    """
    return F.relu(p1 * incoming_flux - p2)
