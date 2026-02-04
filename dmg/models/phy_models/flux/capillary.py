import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def capillary_1(
    p1: torch.Tensor,
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Capillary rise: based on deficit in higher reservoir.
    f = min(p1 * (1 - S1/S1max), S2)

    Args:
        p1: Maximum capillary rise rate [mm/d]
        S1: Current storage in receiving store [mm]
        S1max: Maximum storage in receiving store [mm]
        S2: Current storage in providing store [mm]
        nearzero: Numerical stability constant
    """
    # Deficit-based flux, active when S1 < S1max
    flux = p1 * (1.0 - S1 / (S1max + nearzero))
    # Constraints: f >= 0, f <= S2
    return torch.minimum(F.relu(flux), S2)


def capillary_2(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Capillary rise at constant rate.
    f = min(p1, S)

    Args:
        p1: Base capillary rise rate [mm/d]
        S: Current storage in providing store [mm]
        nearzero: Numerical stability constant
    """
    # Constraints: f <= S
    return torch.minimum(p1, S)


def capillary_3(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Capillary rise scaled by receiving store's deficit up to a storage threshold.
    f = min(S2, p1 * (1 - S1/p2) * (1 - smooth_threshold_storage_logistic(S1, p2)))

    Args:
        p1: Base capillary rise rate [mm/d]
        p2: Threshold above which no capillary flow occurs [mm]
        S1: Current storage in receiving store [mm]
        S2: Current storage in supplying store [mm]
        nearzero: Numerical stability constant
    """
    # In MARRMoT MATLAB, smoothThreshold_storage_logistic(S, p) returns 1 when S < p.
    # Our smooth_threshold_storage_logistic returns 1 when S > p.
    # Thus we use (1 - sigmoid).
    smooth_gate = 1.0 - smooth_threshold_storage_logistic(
        S1, p2, nearzero=nearzero
    )
    flux = p1 * (1.0 - S1 / (p2 + nearzero)) * smooth_gate
    # Constraints: f >= 0, f <= S2
    return torch.minimum(S2, F.relu(flux))
