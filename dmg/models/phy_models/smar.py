import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List

from dmg.models.phy_models.unify_v2 import UnifyV2

# 引入通量计算函数
from dmg.models.hydromodel.flux.effective import effective_1
from dmg.models.hydromodel.flux.saturation import saturation_1, saturation_6
from dmg.models.hydromodel.flux.infiltration import infiltration_4
from dmg.models.hydromodel.flux.evap import evap_13, evap_14
from dmg.models.hydromodel.flux.split import split_1
from dmg.models.hydromodel.flux.baseflow import baseflow_1
from dmg.models.hydromodel.unithydro.base import DplUHBase


# 引入 Gamma 单位线 (对应 MARRMoT uh_6_gamma)
class DplGamma6(DplUHBase):
    """
    Gamma Unit Hydrograph (Nash Cascade)
    基于用户提供的 lgamma PDF 逻辑实现，适配 Grouped Conv1d。
    """

    def get_weights(self, params):
        """
        params: (Batch, 2) -> [n, k]
        n (alpha): Shape parameter
        k (theta): Scale parameter
        """
        # 1. 提取参数 (Batch, 1)
        # 确保维度是 (Batch, 1, 1) 以便与时间轴 (1, 1, L) 广播
        n = params[:, 0:1].unsqueeze(-1)  # shape: (Batch, 1, 1)
        k = params[:, 1:2].unsqueeze(-1)  # shape: (Batch, 1, 1)

        # 2. 参数安全处理 (参考你的 uh_gamma 逻辑)
        # alpha对应 n, theta对应 k
        # 你的逻辑: F.relu(a) + 0.1
        alpha = F.relu(n) + 0.1
        theta = F.relu(k) + 0.5

        # 3. 时间轴 t
        # self.t_idx 是基类注册的 buffer, shape (1, 1, MaxLag)
        # 你的逻辑用的是 t = arange(0.5, lenF*1.0)，我们也用这个中心点逻辑
        t = self.t_idx.to(
            alpha.device
        )  # t_idx 是 1, 2, 3... -> 0.5, 1.5, 2.5...

        t = t - torch.ones_like(t) * 0.5
        # 4. 计算 Gamma PDF 权重 (无需循环，直接广播)
        # 公式: (1 / (Gamma(alpha) * theta^alpha)) * t^(alpha-1) * exp(-t/theta)

        # log空间计算避免溢出: denom = lgamma(alpha) + alpha * log(theta)
        log_denom = torch.lgamma(alpha) + alpha * torch.log(theta)

        # log_num = (alpha - 1) * log(t) - t / theta
        log_num = (alpha - 1) * torch.log(t) - (t / theta)

        # log_w = log_num - log_denom
        log_w = log_num - log_denom
        w = torch.exp(log_w)

        # 此时 w 的形状已经是 (Batch, 1, MaxLag)
        # 满足 F.conv1d(groups=Batch) 对权重的要求 [Groups, In_channels/Groups, Kernel]
        # 即 [Batch, 1, Length]

        # 5. 归一化
        # 基类 DplUHBase 的 forward 会再做一次归一化，
        # 但在这里先做一次可以保证数值稳定性
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        return w


# ==============================================================================
# 1. Parameter Definitions
# ==============================================================================
SMAR_PARAMS_BOUNDS = {
    "h_runoff": [0.0, 1.0],  # Max fraction of direct runoff [-]
    "y_inf": [0.0, 200.0],  # Infiltration rate [mm/d]
    "smax": [1.0, 2000.0],  # Max soil moisture storage [mm]
    "c_evap": [0.0, 1.0],  # Evap reduction coeff [-]
    "g_rech": [0.0, 1.0],  # Groundwater recharge coeff [-]
    "kg": [0.0, 1.0],  # Groundwater time parameter [d-1]
    "n_res": [1.0, 10.0],  # Number of Nash reservoirs [-]
    "nk_delay": [1.0, 120.0],  # Total routing delay [d]
}

SMAR_PARAMS_DESC = {
    "h_runoff": "Maximum fraction of direct runoff [-]",
    "y_inf": "Infiltration rate [mm/d]",
    "smax": "Maximum soil moisture storage [mm]",
    "c_evap": "Evaporation reduction coefficient [-]",
    "g_rech": "Groundwater recharge coefficient [-]",
    "kg": "Groundwater time parameter [d-1]",
    "n_res": "Number of Nash cascade reservoirs [-]",
    "nk_delay": "Routing delay [d] (n*k)",
}


# ==============================================================================
# 2. Static Step Function (Compiled)
# ==============================================================================


def _smar_production_step_impl(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    S6: torch.Tensor,
    h_runoff: torch.Tensor,
    y_inf: torch.Tensor,
    smax: torch.Tensor,
    c_evap: torch.Tensor,
    g_rech: torch.Tensor,
    kg: torch.Tensor,
    nearzero: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Phase 1: Production Step (Strict Match with MATLAB model_fun)
    """

    # --- Fluxes Functions (Correspondence to MATLAB) ---

    # flux_pstar = effective_1(P,Ep);
    flux_pstar = effective_1(P, PET, nearzero=nearzero)

    # flux_estar = effective_1(Ep,P);
    flux_estar = effective_1(PET, P, nearzero=nearzero)

    # flux_evap = min(Ep,P); (Base evap from intercepted/surface)
    flux_evap_base = torch.minimum(PET, P)

    # flux_r1 = saturation_6(h,(S1+S2+S3+S4+S5),smax,flux_pstar);
    S_tot = S1 + S2 + S3 + S4 + S5
    flux_r1 = saturation_6(h_runoff, S_tot, smax, flux_pstar, nearzero=nearzero)

    # flux_i = infiltration_4(flux_pstar-flux_r1,y);
    # Note: infiltration_4(incoming, capacity)
    inflow_i = flux_pstar - flux_r1
    flux_i = infiltration_4(inflow_i, y_inf, nearzero=nearzero)

    # flux_r2 = effective_1(flux_pstar-flux_r1,flux_i);
    flux_r2 = effective_1(inflow_i, flux_i, nearzero=nearzero)

    # --- Evaporation (Uses OLD States S1..S5) ---
    # flux_e1 = evap_13(c,0,flux_estar,S1,delta_t); (delta_t assumed 1.0)
    flux_e1 = evap_13(
        c_evap,
        torch.tensor(0.0, device=P.device),
        flux_estar,
        S1,
        nearzero=nearzero,
    )

    # flux_e2 = evap_14(c,1,flux_estar,S2,S1,0.1,delta_t);
    flux_e2 = evap_14(
        c_evap,
        torch.tensor(1.0, device=P.device),
        flux_estar,
        S2,
        S1,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )

    # flux_e3 = evap_14(c,2,flux_estar,S3,S2,0.1,delta_t);
    flux_e3 = evap_14(
        c_evap,
        torch.tensor(2.0, device=P.device),
        flux_estar,
        S3,
        S2,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )

    # flux_e4 = evap_14(c,3,flux_estar,S4,S3,0.1,delta_t);
    flux_e4 = evap_14(
        c_evap,
        torch.tensor(3.0, device=P.device),
        flux_estar,
        S4,
        S3,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )

    # flux_e5 = evap_14(c,4,flux_estar,S5,S4,0.1,delta_t);
    flux_e5 = evap_14(
        c_evap,
        torch.tensor(4.0, device=P.device),
        flux_estar,
        S5,
        S4,
        torch.tensor(0.1, device=P.device),
        nearzero=nearzero,
    )

    # --- Percolation / Overflow (Chain Reaction) ---
    layer_cap = smax / 5.0

    # flux_q1 = saturation_1(flux_i, S1,smax/5);
    # Logic: How much of flux_i overflows S1?
    flux_q1 = saturation_1(flux_i, S1, layer_cap, nearzero=nearzero)

    # flux_q2 = saturation_1(flux_q1,S2,smax/5);
    flux_q2 = saturation_1(flux_q1, S2, layer_cap, nearzero=nearzero)

    # flux_q3 = saturation_1(flux_q2,S3,smax/5);
    flux_q3 = saturation_1(flux_q2, S3, layer_cap, nearzero=nearzero)

    # flux_q4 = saturation_1(flux_q3,S4,smax/5);
    flux_q4 = saturation_1(flux_q3, S4, layer_cap, nearzero=nearzero)

    # flux_r3 = saturation_1(flux_q4,S5,smax/5);
    flux_r3 = saturation_1(flux_q4, S5, layer_cap, nearzero=nearzero)

    # --- Groundwater Split ---
    # flux_rg = split_1(g,flux_r3);
    flux_rg = split_1(g_rech, flux_r3, nearzero=nearzero)

    # flux_r3star = split_1(1-g,flux_r3);
    flux_r3star = split_1(1.0 - g_rech, flux_r3, nearzero=nearzero)

    # flux_qg = baseflow_1(kg,S6);
    flux_qg = baseflow_1(kg, S6, nearzero=nearzero)

    # --- State Updates (dS) ---
    # dS1 = flux_i  - flux_e1 - flux_q1;
    S1_new = torch.clamp(S1 + flux_i - flux_e1 - flux_q1, min=nearzero)

    # dS2 = flux_q1 - flux_e2 - flux_q2;
    S2_new = torch.clamp(S2 + flux_q1 - flux_e2 - flux_q2, min=nearzero)

    # dS3 = flux_q2 - flux_e3 - flux_q3;
    S3_new = torch.clamp(S3 + flux_q2 - flux_e3 - flux_q3, min=nearzero)

    # dS4 = flux_q3 - flux_e4 - flux_q4;
    S4_new = torch.clamp(S4 + flux_q3 - flux_e4 - flux_q4, min=nearzero)

    # dS5 = flux_q4 - flux_e5 - flux_r3;
    S5_new = torch.clamp(S5 + flux_q4 - flux_e5 - flux_r3, min=nearzero)

    # dS6 = flux_rg - flux_qg;
    S6_new = torch.clamp(S6 + flux_rg - flux_qg, min=nearzero)

    # --- Outputs ---
    # flux_qr = route(flux_r1+flux_r2+flux_r3star, uh); (Processed in convolution phase)
    flux_qr_in = flux_r1 + flux_r2 + flux_r3star

    # Total Evap
    flux_ea = flux_evap_base + flux_e1 + flux_e2 + flux_e3 + flux_e4 + flux_e5

    return (
        flux_qr_in,
        flux_qg,
        flux_ea,
        S1_new,
        S2_new,
        S3_new,
        S4_new,
        S5_new,
        S6_new,
    )


# Compile
_smar_production_step = torch.compile(_smar_production_step_impl)


# ==============================================================================
# 3. Model Class (SmarModel)
# ==============================================================================


class Smar(UnifyV2):
    """
    SMAR Model (MARRMoT m_40)

    Architecture:
    1. Production: 6-store physics, outputs Surface Flow (qr_in) & Baseflow (qg).
    2. Convolution: Routes Surface Flow using Gamma UH (Nash Cascade).
    3. Summation: Q = Routed_Surface + Baseflow.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        if config is None:
            config = {}
        config.setdefault("model_name", "smar")
        super().__init__(config, device, backend)

        # Initialize Unit Hydrograph (Gamma / Nash Cascade)
        # Using bounds of nk_delay to define max lag
        self.uh = DplGamma6(max_lag=int(SMAR_PARAMS_BOUNDS["nk_delay"][1]))

    def _init_states(self, n_grid: int) -> Tuple[torch.Tensor, ...]:
        # Initialize 6 states
        states = tuple(
            torch.zeros((n_grid, self.nmul), device=self.device) + self.nearzero
            for _ in range(6)
        )
        return states

    def _run_model(
        self,
        x_dict: dict,
        states: Tuple[torch.Tensor, ...],
        dy_params: Dict[str, torch.Tensor],
        static_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict['x_phy']
        n_steps, n_grid = forcing.shape[:2]
        nmul = self.nmul
        nearzero = self.nearzero

        # Unbind forcing
        P_seq = forcing[..., 0:1].expand(-1, -1, nmul).unbind(0)
        PET_seq = forcing[..., 2:3].expand(-1, -1, nmul).unbind(0)

        # Unpack Parameters
        h_runoff = static_params["h_runoff"]
        y_inf = static_params["y_inf"]
        smax = static_params["smax"]
        c_evap = static_params["c_evap"]
        g_rech = static_params["g_rech"]
        kg = static_params["kg"]

        # Routing Parameters
        n_res = static_params["n_res"]  # n (shape)
        nk_delay = static_params["nk_delay"]  # nk (mean delay)

        S1, S2, S3, S4, S5, S6 = states

        # ==========================================================
        # Phase 1: Production Loop
        # ==========================================================
        raw_qr_list = []  # Surface runoff to route
        raw_qg_list = []  # Baseflow
        # ea_list = []

        for t in range(n_steps):
            flux_qr_in, flux_qg, flux_ea, S1, S2, S3, S4, S5, S6 = (
                _smar_production_step(
                    P_seq[t],
                    PET_seq[t],
                    S1,
                    S2,
                    S3,
                    S4,
                    S5,
                    S6,
                    h_runoff,
                    y_inf,
                    smax,
                    c_evap,
                    g_rech,
                    kg,
                    nearzero,
                )
            )
            raw_qr_list.append(flux_qr_in)
            raw_qg_list.append(flux_qg)
            # ea_list.append(flux_ea)

        # Stack: (T, B, M)
        qr_stack = torch.stack(raw_qr_list, dim=0)
        qg_stack = torch.stack(raw_qg_list, dim=0)

        # ==========================================================
        # Phase 2: Convolution (Nash Cascade / Gamma)
        # ==========================================================

        # 1. Flatten for Conv1d: (B*M, T)
        B_total = n_grid * nmul
        qr_flat = qr_stack.permute(1, 2, 0).reshape(B_total, n_steps)

        # 2. UH Params: (B*M, 2)
        # Convert nk_delay -> k (scale parameter)
        # MATLAB: k = nk / n
        n_flat = n_res.reshape(B_total, 1)
        nk_flat = nk_delay.reshape(B_total, 1)
        k_flat = nk_flat / (n_flat + nearzero)

        # Concatenate [n, k] for the DplGamma6 UH layer
        uh_params = torch.cat([n_flat, k_flat], dim=1)

        # 3. Apply Convolution
        routed_qr_flat = self.uh(qr_flat, uh_params)

        # 4. Reshape back: (T, B, M)
        routed_qr = routed_qr_flat.view(n_grid, nmul, n_steps).permute(2, 0, 1)

        # ==========================================================
        # Phase 3: Aggregation
        # ==========================================================
        # Q = Routed Surface + Baseflow
        Qsim_out = routed_qr + qg_stack

        # ==========================================================
        # Finalize
        # ==========================================================
        if self.initialize:
            return (S1, S2, S3, S4, S5, S6)

        return self._finalize_output(Qsim_out)
