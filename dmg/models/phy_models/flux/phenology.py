import torch
import torch.nn.functional as F


def phenology_1(
    T: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Corrects Ep for phenology effects.
    out = min(1, max(0, (T - p1) / (p2 - p1))) * Ep

    Inputs:
        T: current temperature [oC]
        p1: temperature threshold where evaporation stops [oC]
        p2: temperature threshold above which corrected Ep = Ep [oC]
        Ep: current potential evapotranspiration [mm/d]
    """
    # Use F.relu for max(0, x) and torch.minimum for min(1, x)
    # Protect division by (p2 - p1)
    scale = torch.minimum(
        torch.ones_like(T), F.relu((T - p1) / (p2 - p1 + nearzero))
    )
    return scale * Ep


def phenology_2(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: torch.Tensor,
    tmax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Phenology-based maximum interception capacity.
    out = p1 * (1 + p2 * sin(2 * pi * (t - p3) / tmax))

    Inputs:
        p1: mean interception capacity [mm]
        p2: seasonal change as fraction of the mean [-]
        p3: time of maximum store size [d]
        t: current time step [-]
        tmax: seasonal length [d]
        (dt is removed from original formula as per rules)
    """
    # Protect division by tmax
    angle = 2.0 * torch.pi * (t - p3) / (tmax + nearzero)
    return p1 * (1.0 + p2 * torch.sin(angle))
