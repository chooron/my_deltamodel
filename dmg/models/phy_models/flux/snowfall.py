import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_temperature_logistic


def snowfall_1(
    incoming_flux: torch.Tensor,
    T: torch.Tensor,
    p1: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Snowfall based on temperature threshold.
    Formula: out = In * smooth_threshold_temperature_logistic(T, p1)
    """
    # sf returns ~1 when T < p1 (Snow condition)
    sf = smooth_threshold_temperature_logistic(T, p1, nearzero=nearzero)
    return incoming_flux * sf


def snowfall_2(
    incoming_flux: torch.Tensor,
    T: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Snowfall based on a temperature threshold interval.
    Interval: [p1 - 0.5*p2, p1 + 0.5*p2]
    Formula: out = In * clamp((p1 + 0.5*p2 - T) / p2, 0, 1)
    """
    t_max = p1 + 0.5 * p2
    # Calculate snow fraction: linear ramp from 1 at t_max - p2 to 0 at t_max
    snow_frac = torch.clamp((t_max - T) / (p2 + nearzero), min=0.0, max=1.0)
    return incoming_flux * snow_frac
