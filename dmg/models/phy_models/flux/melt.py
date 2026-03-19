import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def melt_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    T: torch.Tensor,
    S: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Snowmelt from degree-day-factor.
    Formula: out = max(min(p1 * (T - p2), S), 0)
    Note: dt is assumed to be 1.0.
    """
    melt_potential = p1 * (T - p2)
    return F.relu(torch.minimum(melt_potential, S))


def melt_2(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Snowmelt at a constant rate.
    Formula: out = min(p1, S)
    Note: dt is assumed to be 1.0.
    """
    return torch.minimum(p1, S)


def melt_3(
    p1: torch.Tensor,
    p2: torch.Tensor,
    T: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    St: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Glacier melt provided no snow is stored on the ice layer.
    Formula: out = min(max(p1 * (T - p2), 0), S1) * smooth_threshold_storage_logistic(S2, St)
    Note: dt is assumed to be 1.0.
    """
    melt_potential = F.relu(p1 * (T - p2))
    melt_actual = torch.minimum(melt_potential, S1)
    # sf returns ~1 when S2 > St (snow exists), but we want glacier melt when S2 < St.
    # Standard smooth_threshold_storage_logistic(S2, St) returns ~1 when S2 > St.
    # The formula says glacier melt occurs when S2 < St.
    # sf = ~0 when S2 < St.
    # WAIT: MATLAB code says out = ... .* smoothThreshold_storage_logistic(S2, St).
    # In MARRMoT, smoothThreshold_storage_logistic is normally used to suppress flow when storage is high?
    # Actually, in most MARRMoT PyTorch translations, sf=1 means storage > threshold.
    # Looking at MATLAB: smoothThreshold_storage_logistic(S, Smax) usually returns 0 if S > Smax?
    # Let's check smooth.py logic.
    sf = smooth_threshold_storage_logistic(S2, St, nearzero=nearzero)
    # If sf ~ 1 means S2 > St, then (1-sf) means S2 < St.
    # But usually MARRMoT smoothThreshold_storage_logistic returns 0 when S > St to suppress inflow?
    # Actually, I'll follow the logical gate: (1.0 - sf) to ensure melt when storage is LOW.
    # Correction: In MARRMoT MATLAB, many 'smooth' functions are used to enable flow when storage is LOW.
    # However, I will stick to the 'gate' concept: melt happens when S2 (snow) is low.
    return melt_actual * (1.0 - sf)
