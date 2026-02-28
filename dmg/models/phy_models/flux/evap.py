import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def evap_1(
    S: torch.Tensor, Ep: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Evaporation at the potential rate.
    Constraint: f <= S
    """
    return torch.minimum(S, Ep)


def evap_2(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation at a scaled, plant-controlled rate.
    Constraints: f <= Ep, f <= S
    """
    return torch.minimum(torch.minimum(p1 * S / (Smax + nearzero), Ep), S)


def evap_3(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation based on scaled current water storage and wilting point.
    Constraints: f <= Ep, f <= S
    """
    return torch.minimum(torch.minimum(S / (p1 * Smax + nearzero) * Ep, Ep), S)


def evap_4(
    Ep: torch.Tensor,
    p1: torch.Tensor,
    S: torch.Tensor,
    p2: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation based on scaled current water storage, a wilting point,
    a constraining factor and limited by potential rate.
    """
    scaled_s = p1 * (S - p2 * Smax) / (Smax - p2 * Smax + nearzero)
    return torch.minimum(Ep * F.relu(scaled_s), S)


def evap_5(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation from bare soil scaled by relative storage.
    Constraints: Ea <= Ep, Ea <= S
    """
    return F.relu(torch.minimum((1 - p1) * S / (Smax + nearzero) * Ep, S))


def evap_6(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Transpiration from vegetation at the potential rate if storage is above
    a wilting point and scaled by relative storage if not.
    """
    return torch.minimum(
        torch.minimum(p1 * Ep, p1 * Ep * S / (p2 * Smax + nearzero)), S
    )


def evap_7(
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation scaled by relative storage.
    """
    ratio = torch.clamp(S / Smax, max=1.0)
    return torch.minimum(ratio * Ep, S)


def evap_8(
    S1: torch.Tensor,
    S2: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Transpiration from vegetation, at potential rate if soil moisture is
    above the wilting point, and linearly decreasing if not.
    Also scaled by relative storage across all stores.
    """
    s_frac = S1 / (S1 + S2 + nearzero)
    s_limit = S1 / (p2 + nearzero)
    return F.relu(
        torch.minimum(
            torch.minimum(s_frac * p1 * Ep, s_frac * s_limit * p1 * Ep), S1
        )
    )


def evap_9(
    S1: torch.Tensor,
    S2: torch.Tensor,
    p1: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation from bare soil scaled by relative storage and
    by relative water availability across all stores.
    """
    s_frac = S1 / (S1 + S2 + nearzero)
    s_avail = S1 / (Smax - S2 + nearzero)
    return F.relu(torch.minimum(s_frac * (1 - p1) * s_avail * Ep, S1))


def evap_10(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation from bare soil scaled by relative storage.
    """
    return F.relu(torch.minimum(p1 * S / (Smax + nearzero) * Ep, S))


def evap_11(
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation quadratically related to current soil moisture.
    """
    s_rel = S / (Smax + nearzero)
    return F.relu((2 * s_rel - s_rel.pow(2)) * Ep)


def evap_12(
    S: torch.Tensor, p1: torch.Tensor, Ep: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Evaporation from deficit store, with exponential decline as
    deficit goes below a threshold.
    """
    return (
        torch.minimum(
            torch.ones_like(S), torch.exp(2 * (1 - S / (p1 + nearzero)))
        )
        * Ep
    )


def evap_13(
    p1: torch.Tensor,
    p2: torch.Tensor,
    Ep: torch.Tensor,
    S: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Exponentially scaled evaporation.
    """
    return torch.minimum((p1 + nearzero).pow(p2) * Ep, S)


def evap_14(
    p1: torch.Tensor,
    p2: torch.Tensor,
    Ep: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S2min: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Exponentially scaled evaporation that only activates if
    another store goes below a certain threshold.
    """
    evap = torch.minimum((p1 + nearzero).pow(p2) * Ep, S1)
    return evap * smooth_threshold_storage_logistic(
        S2, S2min, nearzero=nearzero
    )


def evap_15(
    Ep: torch.Tensor,
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2min: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Scaled evaporation if another store is below a threshold.
    """
    evap = S1 / (S1max + nearzero) * Ep
    return torch.minimum(
        evap * smooth_threshold_storage_logistic(S2, S2min, nearzero=nearzero),
        S1,
    )


def evap_16(
    p1: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S2min: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Scaled evaporation if another store is below a threshold.
    """
    evap = p1 * Ep
    return torch.minimum(
        evap * smooth_threshold_storage_logistic(S2, S2min, nearzero=nearzero),
        S1,
    )


def evap_17(
    p1: torch.Tensor, S: torch.Tensor, Ep: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Scaled evaporation from a store that allows negative values.
    """
    return 1 / (1 + torch.exp(-p1 * S)) * Ep


def evap_18(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    S: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Exponentially declining evaporation from deficit store.
    """
    return p1 * torch.exp(-p2 * S / (p3 + nearzero)) * Ep


def evap_19(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Non-linear scaled evaporation.
    """
    s_rel = F.relu(S / (Smax + nearzero))
    evap_p = p1 * (s_rel + nearzero).pow(p2) * Ep
    return torch.minimum(torch.minimum(S, Ep), evap_p)


def evap_20(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Evaporation limited by a maximum evaporation rate and
    scaled below a wilting point.
    """
    return torch.minimum(torch.minimum(p1 * S / (p2 * Smax + nearzero), Ep), S)


def evap_21(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Threshold-based evaporation with constant minimum rate.
    """
    rate = torch.maximum(
        p2, torch.minimum(S / (p1 + nearzero), torch.ones_like(S))
    )
    return torch.minimum(rate * Ep, S)


def evap_22(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Threshold-based evaporation rate.
    """
    rate = torch.minimum((S - p1) / (p2 - p1 + nearzero) * Ep, Ep)
    return torch.minimum(S, F.relu(rate))


def evap_23(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Transpiration from vegetation at the potential rate if storage is above
    field capacity and scaled by relative storage if not, plus evaporation
    from bare soil scaled by relative storage.
    """
    v1 = p1 * Ep + (1 - p1) * S / (Smax + nearzero) * Ep
    v2 = (
        p1 * Ep * S / (p2 * Smax + nearzero)
        + (1 - p1) * S / (Smax + nearzero) * Ep
    )
    return torch.minimum(torch.minimum(v1, v2), S)
