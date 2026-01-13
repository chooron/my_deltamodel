from typing import Any, Optional, Union

import torch
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma


class Exphydro(torch.nn.Module):
    """
    HMETS 模型

    这是一个功能完整的、可微分的 PyTorch HMETS 模型，支持状态预热和动态参数。

    Parameters
    ----------
    config : dict, optional
        配置字典，用于覆盖默认设置。
    device : torch.device, optional
        模型运行的设备（'cpu' 或 'cuda'）。
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = "exphydro"
        self.config = config
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- 从配置或默认值设置模型属性 ---
        self.warm_up = 0
        self.warm_up_states = True
        self.pred_cutoff = 0
        self.dynamic_params = []
        self.dy_drop = 0.0
        # HMETS 需要的输入变量
        self.variables = ["Prcp", "Tmean", "Pet"]
        self.routing = True  # 在 HMETS 中，路由是不可分割的一部分
        self.initialize = False
        self.nearzero = 1e-5
        self.nmul = 1

        # HMETS 的参数边界
        self.parameter_bounds = {
            # Snow Model (10)
            "ddf": [0.0, 40.0],  # D
            "Tbm": [-2.0, 3.0],
            "wrf": [0.0, 0.5],  # D, water_retention_fraction
            "Tbf": [-5.0, 2.0],
            "Kf": [0.0, 5.0],
            "exp_fe": [0.0, 1.0],
            # ET (1)
            "ET_efficiency": [0.0, 1.0],
            # Subsurface (6)
            "coef_runoff": [0.0, 1.0],
            "coef_vadose2phreatic": [0.00001, 0.02],
            "coef_vadose": [0.0, 0.1],
            "coef_phreatic": [0.00001, 0.01],
            "vadose_max_level": [0.001, 500.0],
        }

        # HMETS 没有像 HBV 那样独立的路由参数，它们是主参数集的一部分
        self.routing_parameter_bounds = {
            # Unit Hydrograph (4)
            "alpha1": [0.3, 20.0],
            "beta1": [0.01, 5.0],
            "alpha2": [0.5, 13.0],
            "beta2": [0.15, 1.5],
        }

        if config is not None:
            self.warm_up = config.get("warm_up", self.warm_up)
            self.warm_up_states = config.get(
                "warm_up_states", self.warm_up_states
            )
            self.dy_drop = config.get("dy_drop", self.dy_drop)
            self.dynamic_params = config.get("dynamic_params", {}).get(
                self.__class__.__name__, self.dynamic_params
            )
            self.variables = config.get("variables", self.variables)
            self.nearzero = config.get("nearzero", self.nearzero)
            self.nmul = config.get("nmul", self.nmul)

        self.set_parameters()

    def set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul
        self.learnable_param_count2 = (len(self.phy_param_names) - len(self.dynamic_params)) * self.nmul \
                                      + len(self.routing_param_names)
        self.learnable_param_count = self.learnable_param_count1 + self.learnable_param_count2

    def unpack_parameters(
        self,
        parameters: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.

        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        dy_param_count = len(self.dynamic_params)
        dif_count = phy_param_count - dy_param_count

        # Physical dynamic parameters
        phy_dy_params = parameters[0].view(
            parameters[0].shape[0],
            parameters[0].shape[1],
            dy_param_count,
            self.nmul,
        )

        # Physical static parameters
        phy_static_params = parameters[1][:, : dif_count * self.nmul].view(
            parameters[1].shape[0],
            dif_count,
            self.nmul,
        )

        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = parameters[1][:, dif_count * self.nmul :]

        return phy_dy_params, phy_static_params, routing_params

    def descale_phy_dy_parameters(
        self,
        phy_dy_params: torch.Tensor,
        dy_list: list,
    ) -> dict:
        """Descale physical parameters.

        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        dy_list
            List of dynamic parameters.

        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        n_steps = phy_dy_params.size(0)
        n_grid = phy_dy_params.size(1)

        # TODO: Fix; if dynamic parameters are not entered in config as they are
        # in HBV params list, then descaling misamtch will occur.
        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(dy_list):
            staPar = (
                phy_dy_params[-1, :, i, :].unsqueeze(0).repeat([n_steps, 1, 1])
            )

            dynPar = phy_dy_params[:, :, i, :]
            drmask = torch.bernoulli(pmat).detach_().to(self.device)

            comPar = dynPar * (1 - drmask) + staPar * drmask
            param_dict[name] = change_param_range(
                param=comPar,
                bounds=self.parameter_bounds[name],
            )
        return param_dict

    def descale_phy_stat_parameters(
        self,
        phy_stat_params: torch.Tensor,
        stat_list: list,
    ) -> dict:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(stat_list):
            param = phy_stat_params[:, i, :]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.parameter_bounds[name],
            )
        return parameter_dict

    def descale_rout_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> dict:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            param = routing_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """HMETS 的前向传播。"""
        # 解包输入数据
        x = x_dict["x_phy"]  # shape: [time, grids, vars]
        self.muwts = x_dict.get("muwts", None)
        # watershed_area = x_dict['area'] # shape: [grids]

        # 解包参数
        phy_dy_params, phy_static_params, routing_params = self.unpack_parameters(parameters)

        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(
                routing_params
            )

        # --- 状态预热 ---
        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            self.pred_cutoff = self.warm_up
            warm_up = 0

        n_grid = x.size(1)

        # 初始化模型状态
        SNOW_ON_GROUND = (
            torch.zeros(
                [n_grid, self.nmul], dtype=torch.float32, device=self.device
            )
            + self.nearzero
        )
        WATER_IN_SNOWPACK = (
            torch.zeros(
                [n_grid, self.nmul], dtype=torch.float32, device=self.device
            )
            + self.nearzero
        )

        VADOSE_LEVEL = (
            torch.zeros(
                [n_grid, self.nmul], dtype=torch.float32, device=self.device
            )
            + self.nearzero
        )
        PHREATIC_LEVEL = (
            torch.zeros(
                [n_grid, self.nmul], dtype=torch.float32, device=self.device
            )
            + self.nearzero
        )

        phy_dy_params_dict = self.descale_phy_dy_parameters(
            phy_dy_params,
            dy_list=self.dynamic_params,
        )

        phy_static_params_dict = self.descale_phy_stat_parameters(
            phy_static_params,
            stat_list=[param for param in self.phy_param_names if param not in self.dynamic_params],
        )

        # Run the model for the remainder of simulation period.
        return self.PBM(
            x,
            (SNOW_ON_GROUND, WATER_IN_SNOWPACK, VADOSE_LEVEL, PHREATIC_LEVEL),
            phy_dy_params_dict,
            phy_static_params_dict,
        )

    def PBM(
        self,
        forcing: torch.Tensor,
        initial_states: tuple,
        phy_dy_params_dict: dict,
        phy_static_params_dict: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        (
            SNOW_ON_GROUND,
            WATER_IN_SNOWPACK,
            VADOSE_LEVEL,
            PHREATIC_LEVEL,
        ) = initial_states

        Prcp = forcing[:, :, self.variables.index("Prcp")]
        Tmean = forcing[:, :, self.variables.index("Tmean")]
        Pet = forcing[:, :, self.variables.index("Pet")]
        n_steps, n_grid = Prcp.size()

        Pm = Prcp.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = Tmean.unsqueeze(2).repeat(1, 1, self.nmul)
        Petm = Pet.unsqueeze(2).repeat(1, 1, self.nmul)

        horizontal_transfert_mu = torch.zeros(
            n_steps, n_grid, self.nmul, 4, device=self.device, dtype=Pm.dtype
        )
        RET_sim_mu = torch.zeros(
            n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype
        )
        vadose_level_sim_mu = torch.zeros(
            n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype
        )
        phreatic_level_sim_mu = torch.zeros(
            n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype
        )

        param_dict = {}

        for t in range(n_steps):
            for key in phy_dy_params_dict.keys():
                param_dict[key] = phy_dy_params_dict[key][t, :, :]
            for key in phy_static_params_dict.keys():
                param_dict[key] = phy_static_params_dict[key][:, :]
            # -- 雨雪划分 --
            PRECIP = Pm[t]
            RAIN = torch.mul(PRECIP, (Tm[t] >= 0.0).float())
            SNOW = torch.mul(PRECIP, (Tm[t] < 0.0).float())

            # -- 融雪模块 --
            Tdiurnal = Tm[t]
            potential_freezing = (
                param_dict["Kf"]
                * torch.clamp(param_dict["Tbf"] - Tdiurnal, min=self.nearzero)
                ** param_dict["exp_fe"]
            )
            overnight_freezing = torch.min(
                potential_freezing, WATER_IN_SNOWPACK
            )

            WATER_IN_SNOWPACK = WATER_IN_SNOWPACK - overnight_freezing
            SNOW_ON_GROUND = SNOW_ON_GROUND + overnight_freezing

            snowmelt_potential = torch.clamp(
                param_dict["ddf"] * (Tm[t] - param_dict["Tbm"]), min=0
            )

            snowmelt = torch.min(snowmelt_potential, SNOW_ON_GROUND + SNOW)
            SNOW_ON_GROUND = SNOW_ON_GROUND + SNOW - snowmelt

            water_retention = param_dict["wrf"] * SNOW_ON_GROUND

            water_in_snowpack_temp = WATER_IN_SNOWPACK + snowmelt + RAIN
            water_available = torch.clamp(
                water_in_snowpack_temp - water_retention, min=0
            )
            WATER_IN_SNOWPACK = torch.where(
                water_available > 0, water_retention, water_in_snowpack_temp
            )

            # -- 土壤含水模块 --
            RET = param_dict["ET_efficiency"] * Petm[t]

            horizontal_transfert_mu[t, :, :, 0] = (
                param_dict["coef_runoff"]
                * (VADOSE_LEVEL / param_dict["vadose_max_level"])
                * water_available
            )
            infiltration = (
                water_available - horizontal_transfert_mu[t, :, :, 0] - RET
            )

            delayed_runoff_frac = (
                param_dict["coef_runoff"]
                * (VADOSE_LEVEL / param_dict["vadose_max_level"]) ** 2
            )
            horizontal_transfert_mu[t, :, :, 2] = (
                param_dict["coef_vadose"] * VADOSE_LEVEL
            )
            vadose2phreatic = param_dict["coef_vadose2phreatic"] * VADOSE_LEVEL

            horizontal_transfert_mu[t, :, :, 1] = torch.where(
                infiltration > 0,
                delayed_runoff_frac * infiltration,
                torch.tensor(0.0, device=self.device),
            )
            infiltration = torch.clamp(infiltration, min=0)

            VADOSE_LEVEL = (
                VADOSE_LEVEL
                + infiltration
                - RET
                - horizontal_transfert_mu[t, :, :, 1]
                - horizontal_transfert_mu[t, :, :, 2]
                - vadose2phreatic
            )

            vadose_neg_mask = VADOSE_LEVEL < 0
            RET[vadose_neg_mask] = (
                RET[vadose_neg_mask] + VADOSE_LEVEL[vadose_neg_mask]
            )
            RET = torch.clamp(RET, min=0)
            VADOSE_LEVEL = torch.clamp(VADOSE_LEVEL, min=0)

            vadose_pos_mask = VADOSE_LEVEL > param_dict["vadose_max_level"]
            vadose_excess = VADOSE_LEVEL - param_dict["vadose_max_level"]
            horizontal_transfert_mu[t, :, :, 0][vadose_pos_mask] = (
                horizontal_transfert_mu[t, :, :, 0][vadose_pos_mask]
                + vadose_excess[vadose_pos_mask]
            )
            VADOSE_LEVEL = torch.clamp(
                VADOSE_LEVEL, max=param_dict["vadose_max_level"]
            )

            horizontal_transfert_mu[t, :, :, 3] = (
                param_dict["coef_phreatic"] * PHREATIC_LEVEL
            )
            PHREATIC_LEVEL = (
                PHREATIC_LEVEL
                + vadose2phreatic
                - horizontal_transfert_mu[t, :, :, 3]
            )

            # 记录每个mu的模拟变量
            RET_sim_mu[t], vadose_level_sim_mu[t], phreatic_level_sim_mu[t] = (
                RET,
                VADOSE_LEVEL,
                PHREATIC_LEVEL,
            )

        if self.muwts is None:
            horizontal_transfert = horizontal_transfert_mu.mean(dim=2)
        else:
            horizontal_transfert = (horizontal_transfert_mu * self.muwts).sum(
                -1
            )

        # --- 流量演算 (Routing) ---
        if self.routing:
            UH1 = uh_gamma(
                self.routing_param_dict["alpha1"]
                .repeat(n_steps, 1)
                .unsqueeze(-1),  # Shape: [time, n_grid]
                self.routing_param_dict["beta1"]
                .repeat(n_steps, 1)
                .unsqueeze(-1),
                lenF=50,
            ).to(horizontal_transfert.dtype)
            UH2 = uh_gamma(
                self.routing_param_dict["alpha2"]
                .repeat(n_steps, 1)
                .unsqueeze(-1),
                self.routing_param_dict["beta2"]
                .repeat(n_steps, 1)
                .unsqueeze(-1),
                lenF=50,
            ).to(horizontal_transfert.dtype)

            rf_delay = (
                horizontal_transfert[:, :, 1].permute(1, 0).unsqueeze(1)
            )  # Shape: [n_grid, 1, time]
            rf_base = (
                horizontal_transfert[:, :, 2].permute(1, 0).unsqueeze(1)
            )  # Shape: [n_grid, 1, time]

            UH1_permuted = UH1.permute(1, 2, 0)  # Shape: [n_grid, 1, time]
            UH2_permuted = UH2.permute(1, 2, 0)  # Shape: [n_grid, 1, time]

            rf_delay_rout = uh_conv(rf_delay, UH1_permuted).permute(2, 0, 1)
            rf_base_rout = uh_conv(rf_base, UH2_permuted).permute(2, 0, 1)

            rf_ruf = horizontal_transfert[:, :, 0].unsqueeze(-1)
            rf_gwd = horizontal_transfert[:, :, 3].unsqueeze(-1)

            Qsim = rf_delay_rout + rf_base_rout + rf_ruf + rf_gwd

        else:
            Qsim = horizontal_transfert.sum(dim=2).unsqueeze(-1)
            rf_ruf = horizontal_transfert[:, :, 0].unsqueeze(-1)
            rf_delay_rout = horizontal_transfert[:, :, 1].unsqueeze(-1)
            rf_base_rout = horizontal_transfert[:, :, 2].unsqueeze(-1)
            rf_gwd = horizontal_transfert[:, :, 3].unsqueeze(-1)

        out_dict = {
            "streamflow": Qsim,
            "srflow": rf_ruf,
            "interflow": rf_delay_rout,
            "ssflow": rf_base_rout,
            "gwflow": rf_gwd,
            "AET_hydro": RET_sim_mu.mean(dim=2).unsqueeze(-1),
            "vadose_storage": vadose_level_sim_mu.mean(dim=2).unsqueeze(-1),
            "phreatic_storage": phreatic_level_sim_mu.mean(dim=2).unsqueeze(-1),
        }
        if not self.warm_up_states:
            for key in out_dict.keys():
                if key != 'BFI':
                    out_dict[key] = out_dict[key][self.pred_cutoff:, :, :]
        return out_dict
