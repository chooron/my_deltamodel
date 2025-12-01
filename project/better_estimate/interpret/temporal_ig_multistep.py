import torch
from typing import Tuple, Union, List, cast
from torch import Tensor
from tint.attr import TemporalIntegratedGradients


class MultiStepTemporalIntegratedGradients(TemporalIntegratedGradients):
    """
    Multi-step Temporal Integrated Gradients.

    Unlike the standard TemporalIntegratedGradients which only perturbs the
    last time step (calculating the contribution of new information),
    this version perturbs the entire history up to the current time step.
    This allows calculating the contribution of all previous inputs to the
    current output in a causal manner.
    """

    def __init__(self, forward_func, multiply_by_inputs: bool = True):
        super().__init__(forward_func, multiply_by_inputs)

    @staticmethod
    def scale_features(
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        alphas: List[float],
    ) -> Tuple[Tensor, ...]:
        """
        Overridden to scale ALL features up to the current time step,
        instead of just the last one.
        """
        # Get baselines in correct shape (broadcast if scalar)
        # Note: In the original TIG, baselines might be sliced.
        # Here we assume inputs and baselines are already sliced to the current time window [0...t]

        # We need to handle the case where baseline is a scalar or tensor
        formatted_baselines = []
        for input, baseline in zip(inputs, baselines):
            if isinstance(baseline, (int, float)):
                # Create a tensor baseline with same shape as input filled with scalar
                b = torch.full_like(input, baseline)
                formatted_baselines.append(b)
            elif isinstance(baseline, Tensor):
                # If baseline is a tensor, it should match input shape or be broadcastable
                # If it was sliced by TIG, it should match.
                formatted_baselines.append(baseline)
            else:
                raise TypeError(f"Unsupported baseline type: {type(baseline)}")

        baselines = tuple(formatted_baselines)

        # Scale features: (baseline + alpha * (input - baseline))
        # We do this for the WHOLE sequence (dim 1 is time)

        scaled_features_tpl = tuple(
            torch.cat(
                [(baseline + alpha * (input - baseline)) for alpha in alphas],
                dim=0,
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        return cast(Tuple[Tensor, ...], scaled_features_tpl)
