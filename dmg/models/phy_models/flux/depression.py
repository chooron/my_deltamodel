import torch
import torch.nn.functional as F


def depression_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Exponential inflow to surface depression store.
    Formula: out = min(p1 * exp(-p2 * S / max(Smax - S, 0)) * flux, max(Smax - S, 0))
    """
    capacity = F.relu(Smax - S)
    # Protection for exp argument and division
    potential_inflow = (
        p1 * torch.exp(-p2 * S / (capacity + nearzero)) * incoming_flux
    )
    return torch.minimum(potential_inflow, capacity)
