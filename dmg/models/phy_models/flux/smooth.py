import torch


def smooth_threshold_storage_logistic(
    S: torch.Tensor,
    threshold: torch.Tensor,
    k: float = 10.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Sigmoid-based smooth threshold function for storage.
    Returns value close to 1 when S > threshold, and close to 0 otherwise.

    Args:
        S: Current storage [mm]
        threshold: Threshold value [mm]
        k: Smoothing steepness
        nearzero: Numerical stability constant
    """
    thresh_abs = torch.abs(threshold) + nearzero
    scale = torch.clamp(k / thresh_abs, max=50.0)
    return torch.sigmoid(scale * (S - threshold))


def smooth_threshold_temperature_logistic(
    T: torch.Tensor,
    threshold: torch.Tensor,
    k: float = 5.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Sigmoid-based smooth threshold function for temperature.
    Returns value close to 1 when T < threshold (Snow condition),
    and close to 0 when T > threshold (Rain condition).

    Args:
        T: Current temperature [oC]
        threshold: Temperature threshold [oC]
        k: Smoothing steepness
        nearzero: Numerical stability constant
    """
    return torch.sigmoid(k * (threshold - T))
