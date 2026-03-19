import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_temperature_logistic


def rainfall_1(
    incoming_flux: torch.Tensor,
    T: torch.Tensor,
    p1: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Rainfall based on temperature threshold.
    Formula: out = In * (1 - smooth_threshold_temperature_logistic(T, p1))
    """
    # sf returns ~1 when T < p1 (Snow condition)
    sf = smooth_threshold_temperature_logistic(T, p1, nearzero=nearzero)
    return incoming_flux * (1.0 - sf)


def rainfall_2(
    incoming_flux: torch.Tensor,
    T: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Rainfall based on a temperature threshold interval.
    Interval: [p1 - 0.5*p2, p1 + 0.5*p2]
    Formula: out = In * clamp((T - (p1 - 0.5*p2)) / p2, 0, 1)
    """
    t_min = p1 - 0.5 * p2
    # Calculate rain fraction: linear ramp from 0 at t_min to 1 at t_min + p2
    rain_frac = torch.clamp((T - t_min) / (p2 + nearzero), min=0.0, max=1.0)
    return incoming_flux * rain_frac
