import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.split import split_1
from ..flux.evap import evap_1, evap_7
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_5
from ..flux.percolation import percolation_4
from ..flux.soilmoisture import soilmoisture_1, soilmoisture_2
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (based on MARRMoT m_33_sacramento_11p_5s)
SACRAMENTO_PARAMS_BOUNDS = {
    "pctim": [0.0, 1.0],  # Fraction impervious area [-]
    "smax": [1.0, 2000.0],  # Maximum total storage depth [mm]
    "f1": [0.005, 0.995],  # fraction of smax that is Maximum uztwm [-]
    "f2": [0.005, 0.995],  # fraction of smax-uztwm that is Maximum uzfwm [-]
    "kuz": [0.0, 1.0],  # Interflow runoff coefficient [d-1]
    "rexp": [0.0, 7.0],  # Base percolation rate non-linearity factor [-]
    "f3": [0.005, 0.995],  # fraction of remainder that is Maximum lztwm [-]
    "f4": [0.005, 0.995],  # fraction of remainder that is Maximum lzfwpm [-]
    "pfree": [
        0.0,
        1.0,
    ],  # Fraction of percolation directed to free water stores [-]
    "klzp": [0.0, 1.0],  # Primary baseflow runoff coefficient [d-1]
    "klzs": [0.0, 1.0],  # Supplemental baseflow runoff coefficient [d-1]
}

# Parameter description dictionary
SACRAMENTO_PARAMS_DESC = {
    "pctim": "Fraction impervious area [-]",
    "smax": "Maximum total storage depth [mm]",
    "f1": "Fraction of smax that is Maximum upper zone tension water storage [mm]",
    "f2": "Fraction of smax-uztwm that is Maximum upper zone free water storage [mm]",
    "kuz": "Interflow runoff coefficient [d-1]",
    "rexp": "Base percolation rate non-linearity factor [-]",
    "f3": "Fraction of smax-uztwm-uzfwm that is Maximum lower zone tension water storage [mm]",
    "f4": "Fraction of remainder that is Maximum lower zone primary free water storage [mm]",
    "pfree": "Fraction of percolation directed to free water stores [-]",
    "klzp": "Primary baseflow runoff coefficient [d-1]",
    "klzs": "Supplemental baseflow runoff coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Create initial states for Sacramento model.
    S1: Upper zone tension water
    S2: Upper zone free water
    S3: Lower zone tension water
    S4: Lower zone primary free water
    S5: Lower zone supplemental free water
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


def deficit_based_distribution(
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-performance, gradient-stable implementation of MARRMoT's deficitBasedDistribution.

    Logic:
    1. Calculate relative deficits: rd = (Smax - S) / Smax
    2. If sum(rd) > 0: f1 = rd1 / (rd1 + rd2)
    3. If sum(rd) == 0 (both full): f1 = S1max / (S1max + S2max)
    """
    # 1. Clamp states to max to ensure non-negative deficit (Stability fix)
    #    Although logic theoretically prevents S > Smax, numerical drift can cause it.
    S1_safe = torch.minimum(S1, S1max)
    S2_safe = torch.minimum(S2, S2max)

    # 2. Calculate Relative Deficits (rd)
    #    MATLAB: rd = (S - Smax) / Smax (which is negative).
    #    Here we use Positive Deficit Ratio for cleaner math: (Smax - S) / Smax
    #    The ratio result is identical.
    rd1 = (S1max - S1_safe) / (S1max + nearzero)
    rd2 = (S2max - S2_safe) / (S2max + nearzero)

    sum_rd = rd1 + rd2

    # 3. Calculation for Case A: Deficit exists
    #    Add nearzero to denominator to protect gradient even if masked out later
    f1_deficit = rd1 / (sum_rd + nearzero)

    # 4. Calculation for Case B: Both stores full (sum_rd approx 0)
    #    Distribute based on capacity size
    sum_cap = S1max + S2max
    f1_capacity = S1max / (sum_cap + nearzero)

    # 5. Differentiable Switch using torch.where
    #    If total relative deficit is significant, use deficit-based split.
    #    Otherwise (stores are full), use capacity-based split.
    condition = sum_rd > nearzero
    f1 = torch.where(condition, f1_deficit, f1_capacity)

    # 6. Enforce conservation and bounds
    f1 = torch.clamp(f1, 0.0, 1.0)
    f2 = 1.0 - f1

    return f1, f2


def sacramento_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    pctim: torch.Tensor,
    smax: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    kuz: torch.Tensor,
    rexp: torch.Tensor,
    f3: torch.Tensor,
    f4: torch.Tensor,
    pfree: torch.Tensor,
    klzp: torch.Tensor,
    klzs: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Sacramento Soil Moisture Accounting (Sac-SMA) model single-step calculation.
    Optimized for PyTorch gradients and stability.
    """

    # --- 0. Setup Derived Parameters ---
    # Using torch.maximum/minimum to ensure physical constraints on parameters
    uztwm = f1 * smax
    uzfwm = torch.maximum(
        torch.tensor(0.005 / 4.0, device=P.device), f2 * (smax - uztwm)
    )
    lztwm = torch.maximum(
        torch.tensor(0.005 / 4.0, device=P.device), f3 * (smax - uztwm - uzfwm)
    )
    lzfwpm = torch.maximum(
        torch.tensor(0.005 / 4.0, device=P.device),
        f4 * (smax - uztwm - uzfwm - lztwm),
    )
    lzfwsm = torch.maximum(
        torch.tensor(0.005 / 4.0, device=P.device),
        (1.0 - f4) * (smax - uztwm - uzfwm - lztwm),
    )

    pbase = lzfwpm * klzp + lzfwsm * klzs

    # Base percolation rate multiplication factor
    # Stability: Add nearzero to denominator
    denom_zperc = lzfwsm * klzs + lzfwpm * klzp + nearzero
    zperc_num = (lztwm + lzfwsm * (1.0 - klzs)) / denom_zperc + (
        lzfwpm * (1.0 - klzp)
    ) / denom_zperc
    zperc = torch.minimum(torch.tensor(100000.0, device=P.device), zperc_num)

    # --- 1. Surface Split ---
    flux_qdir = split_1(pctim, P)
    flux_peff = F.relu(P - flux_qdir)

    # --- 2. Upper Zone Processes (S1, S2) ---
    # Inflow to S1
    flux_twexu = saturation_1(flux_peff, S1, uztwm)
    zeros = torch.zeros_like(flux_twexu)
    # Bound constraint
    flux_twexu = torch.clamp(flux_twexu, min=zeros, max=flux_peff)

    S1_tmp = torch.clamp(S1 + flux_peff - flux_twexu, min=nearzero)

    # Evaporation from upper zone tension water (S1)
    flux_euztw = evap_7(S1_tmp, uztwm, PET, nearzero=nearzero)
    flux_euztw = torch.minimum(
        flux_euztw, S1_tmp - nearzero
    )  # Prevent negative state

    S1_new = torch.clamp(S1_tmp - flux_euztw, min=nearzero)

    # Inflow to S2 from S1 excess
    flux_qsur = saturation_1(flux_twexu, S2, uzfwm)
    flux_qsur = torch.clamp(flux_qsur, min=zeros, max=flux_twexu)

    S2_tmp = torch.clamp(S2 + flux_twexu - flux_qsur, min=nearzero)

    # Evaporation from upper zone free water (S2)
    pet_rem_s2 = F.relu(PET - flux_euztw)
    flux_euzfw = evap_1(S2_tmp, pet_rem_s2, nearzero=nearzero)
    flux_euzfw = torch.minimum(flux_euzfw, S2_tmp - nearzero)

    S2_tmp_evap = torch.clamp(S2_tmp - flux_euzfw, min=nearzero)

    # Interflow from upper zone free water (S2)
    flux_qint = interflow_5(kuz, S2_tmp_evap, nearzero=nearzero)
    flux_qint = torch.minimum(flux_qint, S2_tmp_evap - nearzero)

    S2_tmp_int = torch.clamp(S2_tmp_evap - flux_qint, min=nearzero)

    # Rebalance between S1 and S2
    flux_ru = soilmoisture_1(
        S1_new, uztwm, S2_tmp_int, uzfwm, nearzero=nearzero
    )

    S1_final = torch.clamp(S1_new + flux_ru, min=nearzero)
    S2_tmp_ru = torch.clamp(S2_tmp_int - flux_ru, min=nearzero)

    # Percolation from S2 to lower zone
    # Calculate deficits explicitly
    lztw_def = F.relu(lztwm - S3)
    lzfwp_def = F.relu(lzfwpm - S4)
    lzfws_def = F.relu(lzfwsm - S5)
    total_def = lztw_def + lzfwp_def + lzfws_def
    total_lmask = lztwm + lzfwpm + lzfwsm

    flux_pc = percolation_4(
        pbase,
        zperc,
        rexp,
        total_def,
        total_lmask,
        S2_tmp_ru,
        uzfwm,
        nearzero=nearzero,
    )
    flux_pc = torch.minimum(flux_pc, S2_tmp_ru - nearzero)

    S2_new = torch.clamp(S2_tmp_ru - flux_pc, min=nearzero)

    # --- 3. Lower Zone Processes (S3, S4, S5) ---
    # Split percolation into tension water and free water components
    flux_pctw = split_1(1.0 - pfree, flux_pc)

    # Tension water S3 update
    S3_tmp = torch.clamp(S3 + flux_pctw, min=nearzero)

    # Evaporation from lower tension water (S3)
    pet_rem_s3 = F.relu(pet_rem_s2 - flux_euzfw)
    flux_elztw = evap_7(S3_tmp, lztwm, pet_rem_s3, nearzero=nearzero)
    flux_elztw = torch.minimum(flux_elztw, S3_tmp - nearzero)

    S3_tmp_evap = torch.clamp(S3_tmp - flux_elztw, min=nearzero)

    # Excess from S3 directs to free water stores
    # Logic corrected: flux_pctw causes overflow.
    # flux_twexl is the part of flux_pctw that cannot fit in S3
    flux_twexl = saturation_1(flux_pctw, S3, lztwm)

    S3_new_tmp = torch.clamp(
        S3_tmp_evap + flux_pctw - flux_twexl - flux_elztw, min=nearzero
    )

    # -----------------------------------------------------------------
    # REPLACED SECTION: Optimized Deficit Based Distribution
    # -----------------------------------------------------------------
    # Calculate split fractions for Primary (p) and Supplemental (s) stores
    # dist_p will correspond to f1 in the helper, dist_s to f2
    dist_p, dist_s = deficit_based_distribution(
        S4, lzfwpm, S5, lzfwsm, nearzero=nearzero
    )

    # Distribute S3 overflow (flux_twexl)
    flux_twexlp = dist_p * flux_twexl
    flux_twexls = dist_s * flux_twexl

    # Distribute dedicated free water percolation (pfree part)
    flux_pcfw = F.relu(flux_pc - flux_pctw)
    flux_pcfwp = dist_p * flux_pcfw
    flux_pcfws = dist_s * flux_pcfw
    # -----------------------------------------------------------------

    # Baseflow from primary and supplemental stores
    flux_qbfp = baseflow_1(klzp, S4, nearzero=nearzero)
    flux_qbfs = baseflow_1(klzs, S5, nearzero=nearzero)

    # Rebalance between S3 and free water stores S4, S5
    flux_rlp = soilmoisture_2(
        S3_new_tmp, lztwm, S4, lzfwpm, S5, lzfwsm, nearzero=nearzero
    )
    flux_rls = soilmoisture_2(
        S3_new_tmp, lztwm, S5, lzfwsm, S4, lzfwpm, nearzero=nearzero
    )

    # Finalize state updates
    S3_new = torch.clamp(S3_new_tmp + flux_rlp + flux_rls, min=nearzero)

    S4_new = torch.clamp(
        S4 + flux_twexlp + flux_pcfwp - flux_rlp - flux_qbfp, min=nearzero
    )

    S5_new = torch.clamp(
        S5 + flux_twexls + flux_pcfws - flux_rls - flux_qbfs, min=nearzero
    )

    # --- 4. Output Aggregation ---
    Qsim = flux_qdir + flux_qsur + flux_qint + flux_qbfp + flux_qbfs
    Ea = flux_euztw + flux_euzfw + flux_elztw

    return Qsim, Ea, S1_final, S2_new, S3_new, S4_new, S5_new
