import torch
import torch.nn.functional as F


def excess_1(
    So: torch.Tensor, Smax: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Storage excess when store size changes.
    Formula: out = max(So - Smax, 0)
    Note: dt is assumed to be 1.0.
    """
    return F.relu(So - Smax)
