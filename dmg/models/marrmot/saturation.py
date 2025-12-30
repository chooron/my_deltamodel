import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def saturation_1(
    incoming_flux: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess from a store that has reached maximum capacity.
    """
    return incoming_flux * smooth_threshold_storage_logistic(
        S, Smax, nearzero=nearzero
    )


def saturation_2(
    S: torch.Tensor,
    Smax: torch.Tensor,
    p1: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess from a store with different degrees of saturation.
    """
    s_rel = S / (Smax + nearzero)
    term = torch.clamp(1.0 - s_rel, min=0.0, max=1.0)
    out_frac = 1.0 - (term + nearzero).pow(p1)
    return out_frac * incoming_flux


def saturation_3(
    S: torch.Tensor,
    Smax: torch.Tensor,
    p1: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess from a store with different degrees of saturation (exponential variant).
    """
    ratio = S / (Smax + nearzero)
    out_frac = 1.0 - (1.0 / (1.0 + torch.exp((ratio + 0.5) / (p1 + nearzero))))
    return out_frac * incoming_flux


def saturation_4(
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess from a store with different degrees of saturation (quadratic variant).
    """
    out_frac = F.relu(1.0 - (S / (Smax + nearzero)).pow(2))
    return out_frac * incoming_flux


def saturation_5(
    S: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Deficit store: exponential saturation excess based on current storage and a threshold parameter.
    """
    ratio = F.relu(S) / (p1 + nearzero)
    out_frac = 1.0 - torch.minimum(
        torch.ones_like(S), (ratio + nearzero).pow(p2)
    )
    return out_frac * incoming_flux


def saturation_6(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess from a store with different degrees of saturation (linear variant).
    """
    return p1 * S / (Smax + nearzero) * incoming_flux


def saturation_7(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    p5: torch.Tensor,
    S: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess from a store with different degrees of saturation (gamma function variant).
    """
    x_low = p5 * F.relu(S) + p4
    y = F.relu(x_low - p3)
    return (
        torch.special.gammaincc(p2 + nearzero, y / (p1 + nearzero))
        * incoming_flux
    )


def saturation_8(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess flow from a store with different degrees of saturation (min-max linear variant).
    """
    return (p1 + (p2 - p1) * S / (Smax + nearzero)) * incoming_flux


def saturation_9(
    incoming_flux: torch.Tensor,
    S: torch.Tensor,
    St: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Deficit store: Saturation excess from a store that has reached maximum capacity.
    """
    return incoming_flux * (
        1.0 - smooth_threshold_storage_logistic(S, St, nearzero=nearzero)
    )


def saturation_10(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    S: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess flow from a store with different degrees of saturation (min-max exponential variant).
    """
    return torch.minimum(p1, p2 + p2 * torch.exp(p3 * S)) * incoming_flux


def saturation_11(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smin: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess flow from a store with different degrees of saturation (min exponential variant).
    """
    ratio = F.relu(S - Smin) / (Smax - Smin + nearzero)
    term = torch.minimum(torch.ones_like(S), p1 * (ratio + nearzero).pow(p2))
    return (
        incoming_flux
        * term
        * smooth_threshold_storage_logistic(S, Smin, nearzero=nearzero)
    )


def saturation_12(
    p1: torch.Tensor,
    p2: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess flow from a store with different degrees of saturation (min-max linear variant).
    """
    return F.relu((p1 - p2) / (1.0 - p2 + nearzero)) * incoming_flux


def saturation_13(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess flow from a store with different degrees of saturation (normal distribution variant).
    """
    inner = torch.log10(F.relu(S) / (p1 + nearzero) + nearzero) / torch.log10(
        p1 / (p2 + nearzero) + nearzero
    )
    return incoming_flux * 0.5 * (1.0 + torch.erf(inner / 1.41421356))


def saturation_14(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Saturation excess flow from a store with different degrees of saturation (two-part exponential variant).
    """
    s_rel = S / (Smax + nearzero)
    threshold = 0.5 - p1
    val1 = (0.5 - p1 + nearzero).pow(1.0 - p2) * F.relu(s_rel + nearzero).pow(
        p2
    )
    val2 = 1.0 - (0.5 + p1 + nearzero).pow(1.0 - p2) * F.relu(
        1.0 - s_rel + nearzero
    ).pow(p2)
    return torch.where(s_rel <= threshold, val1, val2) * incoming_flux
