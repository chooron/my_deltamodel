import torch
import torch.nn.functional as F


def refreeze_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    T: torch.Tensor,
    S: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Refreezing of stored melted snow.
    Formula: out = max(min(p1 * p2 * (p3 - T), S), 0)
    Note: dt is assumed to be 1.0.
    """
    refreeze_potential = p1 * p2 * (p3 - T)
    return F.relu(torch.minimum(refreeze_potential, S))
