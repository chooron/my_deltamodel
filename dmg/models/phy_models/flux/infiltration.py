import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def infiltration_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    fin: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Infiltration as exponentially declining based on relative storage.
    Formula: out = min(p1 * exp(-p2 * S / Smax), fin)
    """
    # Using nearzero for division safety
    rate = p1 * torch.exp(-p2 * S / (Smax + nearzero))
    return torch.minimum(rate, fin)


def infiltration_2(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S1: torch.Tensor,
    S1max: torch.Tensor,
    flux: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Infiltration as exponentially declining based on relative storage.
    Formula: out = max(min(p1 * exp(-p2 * S1 / S1max) - flux, S2), 0)
    Note: dt is assumed to be 1.0.
    """
    potential_inf = p1 * torch.exp(-p2 * S1 / (S1max + nearzero))
    net_inf = potential_inf - flux
    return F.relu(torch.minimum(net_inf, S2))


def infiltration_3(
    incoming_flux: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Infiltration to soil moisture of liquid water stored in snow pack.
    Formula: out = In * (1 - smooth_threshold_storage_logistic(S, Smax))
    """
    # Passing nearzero to smooth_threshold_storage_logistic
    # sf returns ~1 when S > Smax
    sf = smooth_threshold_storage_logistic(S, Smax, nearzero=nearzero)
    return incoming_flux * (1.0 - sf)


def infiltration_4(
    incoming_flux: torch.Tensor, p1: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Constant infiltration rate.
    Formula: out = min(fin, p1)
    """
    return torch.minimum(incoming_flux, p1)


def infiltration_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Maximum infiltration rate non-linearly based on relative deficit and storage.
    Formula: out = max(0, min(10^9, p1 * (1 - S1/S1max) * (S2/S2max)^(-p2)))
    """
    ratio1 = 1.0 - S1 / (S1max + nearzero)
    ratio2 = F.relu(S2) / (S2max + nearzero)
    # Protection for negative power
    inf_potential = p1 * ratio1 * (ratio2 + nearzero).pow(-p2)
    return F.relu(
        torch.minimum(torch.tensor(1e9, device=p1.device), inf_potential)
    )


def infiltration_6(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    fin: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Infiltration rate non-linearly scaled by relative storage.
    Formula: out = min(fin, p1 * (S/Smax)^p2 * fin)
    """
    ratio = F.relu(S) / (Smax + nearzero)
    inf_rate = p1 * (ratio + nearzero).pow(p2) * fin
    return torch.minimum(fin, inf_rate)


def infiltration_7(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Infiltration exponentially declining and scaled by storage threshold.
    Formula: out = infiltration_1 * (1 - smooth_threshold_storage_logistic(S, Smax))
    """
    inf_1 = infiltration_1(p1, p2, S, Smax, incoming_flux, nearzero)
    sf = smooth_threshold_storage_logistic(S, Smax, nearzero=nearzero)
    return inf_1 * (1.0 - sf)


def infiltration_8(
    S: torch.Tensor,
    Smax: torch.Tensor,
    fin: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Simple capacity-based infiltration.
    Formula: out = (S < Smax) * fin
    Continuous version: (1 - smooth_threshold_storage_logistic) * fin
    """
    sf = smooth_threshold_storage_logistic(S, Smax, nearzero=nearzero)
    return (1.0 - sf) * fin
