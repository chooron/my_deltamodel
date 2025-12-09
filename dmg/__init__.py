"""
DMG - Differentiable Model for Generic hydrological modeling

A deep learning-physics hybrid framework for hydrological modeling,
combining neural networks with conceptual hydrological models.

Acknowledgement
---------------
This package is modified based on the work of MHPI (Multiscale Hydrology 
and Process Innovation) team at Penn State University. The original 
differentiable parameter learning (dPL) framework was developed by:

- Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable,
  regionalized process-based models with multiphysical outputs can approach 
  state-of-the-art hydrologic prediction accuracy. Water Resources Research.

- Song, Y., Bindas, T., et al. (2024). Improving hydrological process 
  understanding through global sensitivity analysis with deep learning. 
  Water Resources Research.

We gratefully acknowledge the MHPI team for their pioneering work in 
differentiable hydrological modeling. This modified version extends 
their framework with additional features and models.

Original repository: https://github.com/mhpi/hydroDL2

Author: chooron
License: MIT
"""

__version__ = "0.1.0"
__author__ = "chooron"
__credits__ = ["MHPI Team (Penn State University)", "chooron"]

# from dmg._version import __version__
from dmg.core import calc, data, post, utils
from dmg.core.data import loaders, samplers
from dmg.models import criterion, delta_models, neural_networks, phy_models
from dmg.models.model_handler import ModelHandler

# In case setuptools scm says version is 0.0.0
# assert not __version__.startswith('0.0.0')

__all__ = [
    "__version__",
    "__author__",
    "__credits__",
    "calc",
    "data",
    "post",
    "utils",
    "loaders",
    "samplers",
    "criterion",
    "delta_models",
    "neural_networks",
    "phy_models",
    "ModelHandler",
]