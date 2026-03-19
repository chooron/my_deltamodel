import torch
import torch.nn.functional as F


def effective_1(
    In1: torch.Tensor, In2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    General effective flow.
    Formula: out = max(In1 - In2, 0)
    """
    return F.relu(In1 - In2)
