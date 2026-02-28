import torch


def split_1(
    p1: torch.Tensor, incoming_flux: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Split flow (returns flux [mm/d])
    Description: p1 fraction of flux is diverted
    """
    return p1 * incoming_flux


def split_2(
    p1: torch.Tensor, incoming_flux: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Split flow (returns flux [mm/d]), counterpart to split_1
    Description: (1-p1) fraction of flux is diverted
    """
    return (1.0 - p1) * incoming_flux
