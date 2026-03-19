import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def soilmoisture_1(
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Water rebalance to equal relative storage (2 stores).
    Formula: out = ((S2 * S1max - S1 * S2max) / (S1max + S2max)) * smooth_threshold(S1/S1max, S2/S2max)
    """
    ratio1 = S1 / (S1max + nearzero)
    ratio2 = S2 / (S2max + nearzero)

    # sf returns ~1 when ratio1 > ratio2 (meaning store 1 is relatively fuller)
    sf = smooth_threshold_storage_logistic(ratio1, ratio2, nearzero=nearzero)

    rebalance = (S2 * S1max - S1 * S2max) / (S1max + S2max + nearzero)
    return rebalance * sf


def soilmoisture_2(
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    S3: torch.Tensor,
    S3max: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Water rebalance to equal relative storage (3 stores).
    Formula: out = (S2 * (S1 * (S2max + S3max) + S1max * (S2 + S3)) / ((S2max + S3max) * (S1max + S2max + S3max))) * smooth_threshold
    """
    ratio1 = S1 / (S1max + nearzero)
    combined_S23 = S2 + S3
    combined_max23 = S2max + S3max
    ratio23 = combined_S23 / (combined_max23 + nearzero)

    # sf returns ~1 when ratio1 > ratio23
    sf = smooth_threshold_storage_logistic(ratio1, ratio23, nearzero=nearzero)

    numerator = S2 * (S1 * combined_max23 + S1max * combined_S23)
    denominator = combined_max23 * (S1max + S2max + S3max + nearzero)

    return (numerator / (denominator + nearzero)) * sf
