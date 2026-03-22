import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.effective import effective_1
from ..flux.split import split_1
from ..flux.evap import evap_1, evap_16
from ..flux.saturation import saturation_1, saturation_9
from ..flux.baseflow import baseflow_1
from ..flux.smooth import smooth_threshold_storage_logistic


def baseflow_6(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 6: Quadratic outflow if storage threshold is exceeded
    Fixed version for TCM S4.
    """
    # 1. 解决量级问题：
    # TCM 的 k2 物理单位极其敏感。为了让模型能学到 [0,1] 范围内的参数，
    # 我们在这里对 p1 进行缩放，假设 S 的单位是 mm。
    # 如果不缩放，k2 必须在 1e-4 级别才正常。
    # 这里除以 1000 是一个经验值，保证 S=100mm, k2=0.5 时，流量约为 5mm/d 而不是 5000mm/d
    scale_factor = 1000.0
    k2_scaled = p1 / scale_factor

    # 2. 计算二次流：
    # 注意：不要在这里直接用 minimum(S) 截断梯度，
    # 而是让它保持公式形态，具体的质量守恒截断(clamp)应该在 tcm_step 外部做，
    # 或者使用软截断（Soft Minimum）来保留梯度，但简单的做法是先算出来。
    q_unconstrained = k2_scaled * S.pow(2)

    # 3. 阈值逻辑修正：
    # sf 在 S > p2 时为 1。我们需要的是“当 S > p2 时有流量”。
    # 所以应该乘以 sf，而不是 (1-sf)。
    sf = smooth_threshold_storage_logistic(S, p2, nearzero=nearzero)

    q_out = q_unconstrained * sf

    # 4. 再次处理梯度截断 (极其重要技巧)：
    # 如果直接用 min(q, S)，当 q > S 时，梯度断裂。
    # 这里我们返回计算值，但在 tcm_step 里你已经写了：
    # flux_q = torch.minimum(flux_q, S4_tmp - nearzero)
    # 这部分保留即可，但为了防止 baseflow_6 内部数值爆炸，可以做一个稍微宽松的约束

    return q_out


# Parameter range dictionary (based on MARRMoT m_25_tcm_6p_4s)
# Note: fa is the fraction of mean(P) that forms abstraction rate.
# The actual abstraction rate ca = fa * mean(P) must be pre-computed
# from the catchment's mean precipitation before calling tcm_step.
TCM_PARAMS_BOUNDS = {
    "phi": [0.0, 1.0],  # Fraction preferential recharge [-]
    "rc": [1.0, 2000.0],  # Maximum soil moisture depth [mm]
    "gam": [0.0, 1.0],  # Fraction of Ep reduction with depth [-]
    "k1": [0.0, 1.0],  # Runoff coefficient [d-1]
    "fa": [0.0, 1.0],  # Fraction of mean(P) that forms abstraction rate [-]
    "k2": [0.0, 1.0],  # Runoff coefficient [mm-1 d-1]
}

# Parameter description dictionary
TCM_PARAMS_DESC = {
    "phi": "Fraction preferential recharge [-]",
    "rc": "Maximum soil moisture depth [mm]",
    "gam": "Fraction of Ep reduction with depth [-]",
    "k1": "Runoff coefficient [d-1]",
    "fa": "Fraction of mean(P) that forms abstraction rate [-]",
    "k2": "Runoff coefficient [mm-1 d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for TCM model.
    S1: Upper soil moisture store
    S2: Soil moisture deficit store (0 = fully saturated)
    S3: Fast routing reservoir
    S4: Slow routing reservoir
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def tcm_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TCM_PARAMS_BOUNDS keys
    phi: torch.Tensor,
    rc: torch.Tensor,
    gam: torch.Tensor,
    k1: torch.Tensor,
    fa: torch.Tensor,
    k2: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    # Pre-computed mean precipitation for abstraction: ca = fa * mean(P)
    mean_P: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Thames Catchment Model (TCM) single-step calculation.

    MATLAB reference: m_25_tcm_6p_4s
    - fa is fraction of mean(P) forming abstraction rate: ca = fa * mean(P)
    - mean_P should be pre-computed from the entire precipitation time series
    - S2 is a deficit store (StoreSigns = -1): increases with ET, decreases with qex1
    - flux_qex2 uses saturation_9: passes qex1 through when S2 deficit is near zero

    Model reference:
    Moore, R. J., & Bell, V. A. (2001). Comparison of rainfall-runoff models
    for flood forecasting. Part 1: Literature review of models.
    """
    # Abstraction rate [mm/d] = fa * mean(P), matching MATLAB init()
    ca = fa * mean_P

    # --- 1. Pre-process ---
    flux_pn = effective_1(P, PET, nearzero=nearzero)
    zeros = torch.zeros_like(P)
    flux_pn = torch.clamp(flux_pn, min=zeros, max=P)
    flux_en = P - flux_pn  # Interception Loss

    flux_pby = split_1(phi, flux_pn, nearzero=nearzero)
    flux_pin = flux_pn - flux_pby

    # --- 2. Upper Store (S1) ---
    # In MATLAB ODE: dS1 = flux_pin - flux_ea - flux_qex1
    # Sequential: add flux_pin first, then compute saturation excess
    S1 = S1 + flux_pin

    # Saturation overflow: saturation_1(flux_pin, S1, rc)
    # When S1 approaches rc, excess water flows out
    flux_qex1 = saturation_1(flux_pin, S1, rc, nearzero=nearzero)
    flux_qex1 = torch.minimum(flux_qex1, S1)
    S1 = S1 - flux_qex1

    # Evap from S1
    flux_ea = evap_1(S1, PET, nearzero=nearzero)
    flux_ea = torch.minimum(flux_ea, S1)
    S1 = S1 - flux_ea
    S1_new = torch.clamp(S1, min=nearzero)

    # --- 3. Deficit Store (S2) ---
    # evap_16: gam * Ep, active when S1 > 0.01 (smooth threshold)
    # Uses full PET per MATLAB: flux_et = evap_16(gam, Inf, S1, 0.01, Ep, dt)
    inf_tensor = torch.full_like(S1_new, float("inf"))
    flux_et = evap_16(
        gam,
        inf_tensor,
        S1_new,
        torch.tensor(0.01, device=P.device),
        PET,
        nearzero=nearzero,
    )

    # S2 is a deficit store: ET deepens deficit, qex1 fills it
    # dS2 = flux_et + flux_qex2 - flux_qex1  (MATLAB ODE)
    # Sequential: first compute qex2 from current S2, then update
    # flux_qex2 = saturation_9(flux_qex1, S2, 0.01):
    #   passes qex1 through when S2 deficit is near zero (saturated)
    flux_qex2 = saturation_9(flux_qex1, S2, torch.tensor(0.01, device=P.device), nearzero=nearzero)

    S2_new = torch.clamp(S2 + flux_et + flux_qex2 - flux_qex1, min=nearzero)

    # --- 4. Fast Routing (S3) ---
    inflow_S3 = flux_qex2 + flux_pby
    S3 = S3 + inflow_S3

    flux_quz = baseflow_1(k1, S3, nearzero=nearzero)
    flux_quz = torch.minimum(flux_quz, S3)
    S3 = S3 - flux_quz
    S3_new = torch.clamp(S3, min=nearzero)

    # --- 5. Slow Routing (S4) ---
    S4 = S4 + flux_quz

    # Abstraction loss: ca = fa * mean(P)
    flux_a = torch.minimum(ca, S4)
    S4 = S4 - flux_a

    # Baseflow: baseflow_6(k2, 0, S4) — quadratic, threshold=0
    flux_q = baseflow_6(
        k2, torch.tensor(0.0, device=P.device), S4, nearzero=nearzero
    )
    flux_q = torch.minimum(flux_q, S4)
    S4 = S4 - flux_q
    S4_new = torch.clamp(S4, min=nearzero)

    # --- 6. Output ---
    Qsim = flux_q
    # Ea = interception loss + S1 evap + deep ET; abstraction is a separate sink
    Ea = flux_en + flux_ea + flux_et

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new
