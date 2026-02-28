import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def area_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smin: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Auxiliary function that calculates a variable contributing area.

    Formula:
    out = min(1, p1 * (max(0, S - Smin) / (Smax - Smin))^p2)

    Constraints:
    A <= 1

    Args:
        p1 (torch.Tensor): linear scaling parameter [-]
        p2 (torch.Tensor): exponential scaling parameter [-]
        S (torch.Tensor): current storage [mm]
        Smin (torch.Tensor): minimum contributing storage [mm]
        Smax (torch.Tensor): maximum contributing storage [mm]
        nearzero (float): numerical stability constant. Defaults to 1e-6.

    Returns:
        torch.Tensor: contributing area fraction [0, 1]
    """
    # Calculate storage excess above minimum threshold using ReLU
    storage_excess = F.relu(S - Smin)

    # Calculate the range of contributing storage [mm]
    storage_range = Smax - Smin

    # Normalized storage term: (max(0, S-Smin) / (Smax-Smin))
    # Protected against division by zero using nearzero
    norm_storage = storage_excess / (storage_range + nearzero)

    # Area calculation: p1 * norm_storage^p2
    # Protected against power-function derivative explosion at norm_storage=0
    area_term = p1 * (norm_storage + nearzero).pow(p2)

    # Apply sigmoid-based smoothing to ensure differentiability at S = Smin
    # Note: MATLAB's (1 - smoothThreshold) logic is replaced by direct product
    # since our smooth_threshold_storage_logistic returns ~1 when S > Smin.
    smoother = smooth_threshold_storage_logistic(S, Smin, nearzero=nearzero)

    # Physical constraint: Area fraction cannot exceed 1.0
    # Use torch.minimum for simple clamping as per .cursorrules
    out = torch.minimum(torch.ones_like(area_term), area_term)

    # Apply smoothing and return
    return out * smoother
