"""
Unified Hydrological Model (Unify)

A unified model wrapper that supports various hydrological models (e.g., HBV, Xinanjiang, Collie3)
by utilizing their standardized step functions and parameter bounds.

Author: chooron / Antigravity
"""

from typing import Any, Optional, Union, Dict, List, Tuple
import torch
import torch.nn as nn
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma
from dmg.models.hydromodel import PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO


class Unify(nn.Module):
    """
    Unified Hydrological Model

    Supports dynamic selection of hydrological models based on configuration.

    Parameters
    ----------
    config : dict, optional
        Model configuration dictionary. Key 'model_name' specifies the hydro model.
    device : torch.device, optional
        Running device.
    backend : str, optional
        Computing backend: "compile" (default), "autograd", or "jit".
    """

    ROUTING_BOUNDS = {"rout_a": [0, 2.9], "rout_b": [0, 6.5]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        super().__init__()

        # Default configuration
        self.config = config or {}
        self.model_name = self.config.get("model_name", "hbv96")
        self.name = f"Unify_{self.model_name}"
        
        # Load model info from registry
        if self.model_name not in PARAM_INFO:
            raise ValueError(f"Unknown model_name: {self.model_name}. Available: {list(PARAM_INFO.keys())}")
            
        self.parameter_bounds = PARAM_INFO[self.model_name]
        self.raw_step_fn = STFN_INFO[self.model_name]
        self.init_fn = INIT_INFO[self.model_name]
        self.n_states = STATE_INFO[self.model_name]
        
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.nmul = 1
        
        # Backend selection
        self.backend = self.config.get("backend", backend)
        
        # Routing bounds
        self.routing_parameter_bounds = self.ROUTING_BOUNDS

        # Device configuration
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Setup step function based on backend
        if self.backend == "compile" and hasattr(torch, "compile"):
            self.step_fn = torch.compile(self.raw_step_fn)
        elif self.backend == "jit":
            self.step_fn = torch.jit.script(self.raw_step_fn)
        else:
            self.step_fn = self.raw_step_fn

        # Load attributes from config
        self._load_config(self.config)
        self._set_parameters()

    def _load_config(self, config: Dict) -> None:
        """Load parameters from config dictionary"""
        simple_attrs = [
            "warm_up",
            "warm_up_states",
            "variables",
            "nearzero",
            "nmul",
        ]
        for attr in simple_attrs:
            if attr in config:
                setattr(self, attr, config[attr])

        if "dynamic_params" in config:
            # Look for dynamic params for this specific model or general
            self.dynamic_params = config["dynamic_params"].get(
                self.model_name, config["dynamic_params"].get("General", [])
            )

    def _set_parameters(self) -> None:
        """Set up parameter names and counts"""
        self.phy_param_names = list(self.parameter_bounds.keys())
        self.routing_param_names = list(self.routing_parameter_bounds.keys())

        static_count = len(self.phy_param_names) - len(self.dynamic_params)
        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul
        self.learnable_param_count2 = static_count * self.nmul + len(
            self.routing_param_names
        )
        self.learnable_param_count = (
            self.learnable_param_count1 + self.learnable_param_count2
        )

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        """Initialize state tensors using the model's init function"""
        return self.init_fn(n_grid, self.nmul, self.device, self.nearzero)

    def _descale_params(
        self, params: torch.Tensor, names: List[str], bounds: Dict[str, List[float]]
    ) -> Dict[str, torch.Tensor]:
        """General parameter descaling"""
        return {
            name: change_param_range(params[:, i, :], bounds[name])
            for i, name in enumerate(names)
        }

    def unpack_parameters(
        self, parameters: Tuple[Optional[torch.Tensor], torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Unpack raw parameters from NN output"""
        dy_count = len(self.dynamic_params)
        static_count = len(self.phy_param_names) - dy_count
        raw_phy_dy, raw_phy_static = parameters

        # Dynamic params: (T, B, dy_count, nmul)
        if raw_phy_dy is not None and dy_count > 0:
            phy_dy = raw_phy_dy.view(
                raw_phy_dy.shape[0], raw_phy_dy.shape[1], dy_count, self.nmul
            )
        else:
            phy_dy = None

        # Static params: (B, static_count, nmul)
        phy_static = raw_phy_static[:, : static_count * self.nmul].view(
            raw_phy_static.shape[0], static_count, self.nmul
        )

        return phy_dy, phy_static

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Forward pass"""
        x = x_dict["x_phy"]

        if not self.warm_up_states:
            self.pred_cutoff = self.warm_up

        # Unpack and descale parameters
        phy_dy, phy_static = self.unpack_parameters(parameters)

        n_grid = x.size(1)

        # Initialize states
        states = self._init_states(n_grid)

        # Descale physical parameters
        static_names = [
            p for p in self.phy_param_names if p not in self.dynamic_params
        ]
        
        phy_dy_dict = {}
        phy_static_dict = self._descale_params(
            phy_static, static_names, self.parameter_bounds
        )

        return self._run_model(x, states, phy_dy_dict, phy_static_dict)

    def _run_model(
        self,
        forcing: torch.Tensor,
        states: Tuple[torch.Tensor, ...],
        dy_params: Dict[str, torch.Tensor],
        static_params: Dict[str, torch.Tensor],
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Optimized unified model execution loop"""
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul
        device = self.device
        nearzero = self.nearzero

        # 1. Pre-extract and expand forcing data (T, B, E)
        # Using expand instead of repeat to create views (zero-copy)
        P_seq = forcing[:, :, 0:1].expand(-1, -1, nmul).unbind(0)
        T_seq = forcing[:, :, 1:2].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[:, :, 2:3].expand(-1, -1, nmul).unbind(0)

        # 2. Pre-process parameters for faster iteration
        # This avoids dictionary lookups and dimensionality checks inside the loop
        fast_params = []
        for name in self.phy_param_names:
            p = dy_params.get(name, static_params.get(name))
            if p.dim() == 3: # Dynamic (T, B, E)
                fast_params.append(p.unbind(0))
            else: # Static (B, E)
                # Use a single-element list and index logic to avoid repetitive list creation
                fast_params.append([p] * n_steps)
        
        # Transpose to (Time, Params)
        params_by_time = list(zip(*fast_params))

        # 3. Buffer for output (avoids list.append and final torch.stack)
        Qsim_out = torch.empty((n_steps, n_grid, nmul), device=device, dtype=torch.float32)
        
        curr_states = states

        # 4. Core Loop
        # The step_fn is typically torch.compiled for maximum execution speed
        for t in range(n_steps):
            outputs = self.step_fn(
                P_seq[t], T_seq[t], PET_seq[t],
                *params_by_time[t],
                *curr_states,
                nearzero
            )
            # Convention: outputs[0] is Qsim, [1] is Ea, [2:] are new states
            Qsim_out[t] = outputs[0]
            curr_states = outputs[2:]

        if self.initialize:
            return curr_states

        return self._finalize_output(Qsim_out, n_steps, n_grid)

    def _apply_averaging(self, Qsimmu: torch.Tensor) -> torch.Tensor:
        """Ensemble averaging over nmul dimension"""
        return Qsimmu.mean(-1)

    def _apply_routing(
        self, Qsim: torch.Tensor, n_steps: int, n_grid: int
    ) -> torch.Tensor:
        """Gamma-unit hydrograph routing"""
        # (B, 1) -> (T, B, 1) for routing weights
        rout_a = self.routing_param_dict["rout_a"].repeat(n_steps, 1).unsqueeze(-1)
        rout_b = self.routing_param_dict["rout_b"].repeat(n_steps, 1).unsqueeze(-1)
        
        UH = uh_gamma(rout_a, rout_b, lenF=15).permute([1, 2, 0])
        rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
        Qsrout = uh_conv(rf, UH).permute([2, 0, 1])
        return Qsrout

    def _finalize_output(
        self,
        Qsim_out: torch.Tensor,
        n_steps: int,
        n_grid: int,
    ) -> Dict[str, torch.Tensor]:
        """Finalize and package model outputs"""
        # 1. Ensemble average
        Qsimavg = self._apply_averaging(Qsim_out)

        # 2. Routing
        Qs = self._apply_routing(Qsimavg, n_steps, n_grid)

        # 3. Build result dict
        result = {
            "streamflow": Qs,
            "Qsimmu": Qsim_out, # Individual ensemble outputs
        }

        # 4. Warm-up cutoff
        if not self.warm_up_states:
            for key in result:
                if result[key] is not None:
                    result[key] = result[key][self.pred_cutoff :]

        return result
