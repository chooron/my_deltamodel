import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.area import area_1
from ..flux.infiltration import infiltration_5, infiltration_4
from ..flux.interception import interception_5
from ..flux.effective import effective_1
from ..flux.saturation import saturation_11, saturation_12
from ..flux.evap import evap_1, evap_19
from ..flux.recharge import recharge_3, recharge_4
from ..flux.baseflow import baseflow_8

# Parameter range dictionary (based on MARRMoT m_23_lascam_24p_3s)
LASCAM_PARAMS_BOUNDS = {
    "af": [0.0, 200.0],  # Catchment-scale infiltration parameter [mm/d]
    "bf": [
        0.0,
        5.0,
    ],  # Catchment-scale infiltration non-linearity parameter [-]
    "stot": [1.0, 2000.0],  # Total catchment storage [mm]
    "xa": [0.01, 0.99],  # Fraction of Stot that is Amax [-]
    "xf": [0.01, 0.99],  # Fraction of Stot-Amax that is depth Fmax [-]
    "na": [0.01, 0.99],  # Fraction of Amax that is Amin [-]
    "ac": [0.0, 5.0],  # Variable contributing area scaling [-]
    "bc": [0.0, 10.0],  # Variable contributing area non-linearity [-]
    "ass": [0.0, 5.0],  # Subsurface saturation area scaling [-]
    "bss": [0.0, 10.0],  # Subsurface saturation area non-linearity [-]
    "c_inf": [0.0, 200.0],  # Maximum infiltration rate [mm/d]
    "ag": [0.0, 5.0],  # Interception base parameter [mm/d]
    "bg": [0.0, 1.0],  # Interception fraction parameter [-]
    "gf": [0.0, 1.0],  # F-store evaporation scaling [-]
    "df": [0.0, 10.0],  # F-store evaporation non-linearity [-]
    "rd": [0.0, 1.0],  # Recharge time parameter [d-1]
    "ab": [0.0, 1.0],  # Groundwater flow scaling [-]
    "bb": [0.01, 200.0],  # Groundwater flow base rate [mm/d]
    "ga": [0.0, 1.0],  # A-store evaporation scaling [-]
    "da": [0.0, 10.0],  # A-store evaporation non-linearity [-]
    "aa": [0.01, 200.0],  # Subsurface storm flow rate [mm/d]
    "ba": [1.0, 5.0],  # Subsurface storm flow non-linearity [-]
    "gb": [0.0, 1.0],  # B-store evaporation scaling [-]
    "db": [0.0, 10.0],  # B-store evaporation non-linearity [-]
}

# Parameter description dictionary
LASCAM_PARAMS_DESC = {
    "af": "Catchment-scale infiltration parameter [mm/d]",
    "bf": "Catchment-scale infiltration non-linearity parameter [-]",
    "stot": "Total catchment storage [mm]",
    "xa": "Fraction of Stot that is Amax [-]",
    "xf": "Fraction of Stot-Amax that is depth Fmax [-]",
    "na": "Fraction of Amax that is Amin [-]",
    "ac": "Variable contributing area scaling [-]",
    "bc": "Variable contributing area non-linearity [-]",
    "ass": "Subsurface saturation area scaling [-]",
    "bss": "Subsurface saturation area non-linearity [-]",
    "c_inf": "Maximum infiltration rate [mm/d]",
    "ag": "Interception base parameter [mm/d]",
    "bg": "Interception fraction parameter [-]",
    "gf": "F-store evaporation scaling [-]",
    "df": "F-store evaporation non-linearity [-]",
    "rd": "Recharge time parameter [d-1]",
    "ab": "Groundwater flow scaling [-]",
    "bb": "Groundwater flow base rate [mm/d]",
    "ga": "A-store evaporation scaling [-]",
    "da": "A-store evaporation non-linearity [-]",
    "aa": "Subsurface storm flow rate [mm/d]",
    "ba": "Subsurface storm flow non-linearity [-]",
    "gb": "B-store evaporation scaling [-]",
    "db": "B-store evaporation non-linearity [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for LASCAM model.
    S1: F-store (Infiltration)
    S2: A-store (Contributing area)
    S3: B-store (Groundwater)
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3


def lascam_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching LASCAM_PARAMS_BOUNDS keys
    af: torch.Tensor,
    bf: torch.Tensor,
    stot: torch.Tensor,
    xa: torch.Tensor,
    xf: torch.Tensor,
    na: torch.Tensor,
    ac: torch.Tensor,
    bc: torch.Tensor,
    ass: torch.Tensor,
    bss: torch.Tensor,
    c_inf: torch.Tensor,
    ag: torch.Tensor,
    bg: torch.Tensor,
    gf: torch.Tensor,
    df: torch.Tensor,
    rd: torch.Tensor,
    ab: torch.Tensor,
    bb: torch.Tensor,
    ga: torch.Tensor,
    da: torch.Tensor,
    aa: torch.Tensor,
    ba: torch.Tensor,
    gb: torch.Tensor,
    db: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    LASCAM (Large-Scale Catchment Model) single-step calculation.

    Model reference:
    Sivapalan, M., Ruprecht, J. K., & Viney, N. R. (1996). Water and salt
    balance modelling to predict the effects of land-use changes in forested
    catchments. 1. Small catchment water balance model. Hydrological Processes, 10(3).
    """

    # --- 0. Auxiliary Parameters ---
    amax = xa * stot
    fmax = xf * (stot - amax)
    bmax = (1.0 - xf) * (stot - amax)
    amin = na * amax

    # --- 1. Surface and Area Processes (S2 dependency) ---
    # Variable areas
    tmp_phiss = area_1(ass, bss, S2, amin, amax, nearzero=nearzero)
    tmp_phic = area_1(ac, bc, S2, amin, amax, nearzero=nearzero)

    # Interception
    flux_pg = interception_5(bg, ag, P, nearzero=nearzero)
    flux_ei = F.relu(P - flux_pg)

    # Saturation excess runoff from contributing area
    flux_qse = saturation_11(ac, bc, S2, amin, amax, flux_pg, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=flux_pg)

    # Potential infiltration flux reach the soil
    rem_pg = F.relu(flux_pg - flux_qse)

    # Infiltration capacity into unsaturated zone
    flux_pc = infiltration_4(rem_pg, c_inf, nearzero=nearzero)
    flux_pc = torch.clamp(flux_pc, min=zeros, max=rem_pg)

    # Infiltration excess runoff
    flux_qie = F.relu(rem_pg - flux_pc)

    # --- 2. Store-Specific Inflows and Subsurface Splitting ---
    # Subsurface storm flow and saturation
    flux_qsse = saturation_12(tmp_phiss, tmp_phic, flux_pc, nearzero=nearzero)

    # Infiltration capacity into F-store S1
    tmp_fss = infiltration_5(af, bf, S3, bmax, S1, fmax, nearzero=nearzero)

    # Inflow to S1 (F-store)
    # limit_ratio = min(1, (1-tmp_phiss)/(1-tmp_phic))
    limit_ratio = torch.clamp(
        (1.0 - tmp_phiss) / (1.0 - tmp_phic + nearzero), min=0.0, max=1.0
    )
    potential_fa = flux_pc * limit_ratio
    flux_fa = infiltration_4(potential_fa, tmp_fss, nearzero=nearzero)
    flux_fa = torch.minimum(flux_fa, potential_fa)

    # Infiltration excess from S1 infiltration attempt
    flux_qsie = F.relu(flux_pc - (flux_fa + flux_qsse))

    # --- 3. Sequential Updates: S1 (F-store) ---
    S1_tmp = S1 + flux_fa
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # Evaporation from S1
    flux_ef = evap_19(gf, df, S1_tmp, fmax, PET, nearzero=nearzero)
    flux_ef = torch.minimum(flux_ef, S1_tmp - nearzero)

    S1_tmp2 = S1_tmp - flux_ef
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # Recharge to S3 from S1
    flux_rf = recharge_3(rd, S1_tmp2, nearzero=nearzero)
    flux_rf = torch.minimum(flux_rf, S1_tmp2 - nearzero)

    S1_new = S1_tmp2 - flux_rf
    S1_new = torch.clamp(S1_new, min=nearzero)

    # --- 4. Sequential Updates: S3 (B-store / Groundwater) ---
    # Baseflow from groundwater
    flux_qb = baseflow_8(bb, ab, S3, bmax, nearzero=nearzero)
    flux_qb = torch.minimum(flux_qb, S3 - nearzero)

    S3_tmp = (
        S3 - flux_qb
    )  # Start with b-flow because it's in multiple stores' ODEs

    # Evaporation from S3
    flux_eb = evap_19(gb, db, S3_tmp, bmax, PET, nearzero=nearzero)
    flux_eb = torch.minimum(flux_eb, S3_tmp - nearzero)

    S3_tmp2 = S3_tmp - flux_eb

    # Inflows to S3 (already calculated flux_rf, and soon flux_ra from S2)
    # But wait, dS2 and dS3 are coupled. We need to sequentially calculate flows.
    # dS2 = flux_qsse + flux_qsie + flux_qb - outflows
    # S2 calculation needs qb.

    # --- 5. Sequential Updates: S2 (A-store) ---
    S2_tmp = S2 + flux_qsse + flux_qsie + flux_qb
    S2_tmp = torch.clamp(S2_tmp, min=nearzero)

    # Evaporation from S2 (two components)
    flux_ea1 = evap_1(S2_tmp, tmp_phic * PET, nearzero=nearzero)
    flux_ea2 = evap_19(ga, da, S2_tmp, amax, PET, nearzero=nearzero)
    flux_ea_total = flux_ea1 + flux_ea2
    flux_ea_total = torch.minimum(flux_ea_total, S2_tmp - nearzero)

    S2_tmp2 = S2_tmp - flux_ea_total
    S2_tmp2 = torch.clamp(S2_tmp2, min=nearzero)

    # Storm flow and Recharge from S2
    # flux_qa: saturation excess storm flow
    flux_qa = saturation_11(
        aa, ba, S2_tmp2, amin, amax, torch.ones_like(P), nearzero=nearzero
    )
    flux_qa = torch.minimum(flux_qa, S2_tmp2 - nearzero)

    S2_tmp3 = S2_tmp2 - flux_qa

    # flux_ra: groundwater recharge from A to B
    flux_ra = recharge_4(tmp_phic, tmp_fss, nearzero=nearzero)
    flux_ra = torch.minimum(flux_ra, S2_tmp3 - nearzero)

    S2_new = S2_tmp3 - flux_ra
    S2_new = torch.clamp(S2_new, min=nearzero)

    # Finalize S3 update
    S3_new = S3_tmp2 + flux_rf + flux_ra
    S3_new = torch.clamp(S3_new, min=nearzero)

    # --- 6. Output Aggregation ---
    # Qsim = qse + qie + qa
    # Ea = ei + ef + ea1 + ea2 + eb
    Qsim = flux_qse + flux_qie + flux_qa
    Ea = flux_ei + flux_ef + flux_ea_total + flux_eb

    return Qsim, Ea, S1_new, S2_new, S3_new
