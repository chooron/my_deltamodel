import torch
from typing import Tuple

@torch.jit.script
def exphydro_timestep_loop(
    P: torch.Tensor,
    T: torch.Tensor,
    Lday: torch.Tensor,
    SNOWPACK: torch.Tensor,
    SOILWATER: torch.Tensor,
    parTmin: torch.Tensor,
    parTmax: torch.Tensor,
    parDf: torch.Tensor,
    parSmax: torch.Tensor,
    parQmax: torch.Tensor,
    parf: torch.Tensor,
    nearzero: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor
]:
    """
    ExpHydro Model Timestep Loop (JIT Optimized Version)

    Parameters
    ----------
    P, T, Lday : torch.Tensor
        Forcing data, shape: (T, B, E)
        T=Time steps, B=Basins, E=Ensemble members (nmul)
        P: Precipitation
        T: Temperature
        Lday: Day length (normalized, for PET calculation)
    SNOWPACK, SOILWATER : torch.Tensor
        Initial states, shape: (B, E)
    par* : torch.Tensor
        Model parameters. Static shape: (B, E), Dynamic shape: (T, B, E)
        parTmin: Minimum temperature for snow
        parTmax: Maximum temperature for melt
        parDf: Degree-day factor
        parSmax: Maximum soil water storage
        parQmax: Maximum baseflow rate
        parf: Baseflow decline rate
    nearzero : float
        Small epsilon to prevent division by zero or negative states

    Returns
    -------
    Tuple containing time-series outputs (shape (T, B, E)) and final states (shape (B, E)).
    """
    n_steps = P.shape[0]
    n_grid = P.shape[1]
    nmul = P.shape[2]
    device = P.device

    # --- Initialize Output Tensors ---
    Q_out = torch.zeros((n_steps, n_grid, nmul), dtype=torch.float32, device=device)
    snowfall_out = torch.zeros_like(Q_out)
    rainfall_out = torch.zeros_like(Q_out)
    melt_out = torch.zeros_like(Q_out)
    pet_out = torch.zeros_like(Q_out)
    evap_out = torch.zeros_like(Q_out)
    baseflow_out = torch.zeros_like(Q_out)
    surfaceflow_out = torch.zeros_like(Q_out)
    
    # State tracking outputs
    snowpack_out = torch.zeros_like(Q_out)
    soilwater_out = torch.zeros_like(Q_out)

    # --- Check for Dynamic Parameters ---
    parTmin_is_dynamic = parTmin.dim() == 3
    parTmax_is_dynamic = parTmax.dim() == 3
    parDf_is_dynamic = parDf.dim() == 3
    parSmax_is_dynamic = parSmax.dim() == 3
    parQmax_is_dynamic = parQmax.dim() == 3
    parf_is_dynamic = parf.dim() == 3

    for t in range(n_steps):
        # --- 1. Get Parameters for Current Timestep ---
        Tmin = parTmin[t] if parTmin_is_dynamic else parTmin
        Tmax = parTmax[t] if parTmax_is_dynamic else parTmax
        Df = parDf[t] if parDf_is_dynamic else parDf
        Smax = parSmax[t] if parSmax_is_dynamic else parSmax
        Qmax = parQmax[t] if parQmax_is_dynamic else parQmax
        f = parf[t] if parf_is_dynamic else parf

        # Current forcing
        Te = T[t]
        Pr = P[t]
        Ld = Lday[t]

        # --- 2. Bucket 1: Surface (Snow & Melt) ---
        
        # Precipitation Partitioning (Rain vs Snow)
        # Using strict threshold logic
        # If Temp < Tmin -> Snow
        is_snow = (Te < Tmin).float()
        is_rain = (Te >= Tmin).float() # Using Tmin as the threshold based on Julia code
        
        snowfall = is_snow * Pr
        rainfall = is_rain * Pr

        # Melt Calculation
        # melt potential = Df * (Temp - Tmax) if Temp > Tmax
        melt_potential = Df * torch.clamp(Te - Tmax, min=0.0)
        # Actual melt cannot exceed existing snowpack
        melt = torch.min(melt_potential, SNOWPACK)

        # PET Calculation (Hamon approximation logic from Julia code)
        # es = 0.611 * exp((17.3 * T) / (T + 237.3))
        # term1 = 29.8 * Lday * 24 * es / (T + 273.2)
        # pet = term1 + melt  <-- Note: Julia code adds melt to PET
        es = 0.611 * torch.exp((17.3 * Te) / (Te + 237.3))
        hamon_term = 29.8 * Ld * 24.0 * es / (Te + 273.2)
        pet = hamon_term + melt

        # Update Snowpack State
        SNOWPACK = torch.clamp(SNOWPACK + snowfall - melt, min=nearzero)

        # --- 3. Bucket 2: Soil ---
        
        # Evaporation
        # evap = pet * min(1.0, soilwater / Smax)
        soil_ratio = torch.clamp(SOILWATER / (Smax + 1e-6), min=0.0, max=1.0)
        evap = pet * soil_ratio
        
        # Baseflow
        # baseflow = Qmax * exp(-f * max(0, Smax - soilwater))
        # Note: 'Smax - soilwater' is the deficit
        deficit = torch.clamp(Smax - SOILWATER, min=0.0)
        baseflow = Qmax * torch.exp(-f * deficit)

        # Surface Flow (Saturation Excess)
        # surfaceflow = max(0, soilwater - Smax)
        # This usually happens *after* adding inputs, but Julia code defines fluxes first.
        # However, standard mass balance updates state based on fluxes.
        # Let's verify Julia dfluxes: d(soilwater) = (rainfall + melt) - (evap + flow)
        # Flow depends on current state.
        
        # In strict logic, surface flow happens when water exceeds capacity.
        # But based on the provided equations: surfaceflow ~ max(0.0, soilwater - Smax)
        # It uses the *current* state (start of step).
        surfaceflow = torch.clamp(SOILWATER - Smax, min=0.0)
        
        flow = baseflow + surfaceflow

        # Update Soilwater State
        # dS/dt = In - Out
        # In = rainfall + melt
        # Out = evap + flow
        # Note: To avoid negative states, we often clamp the outflows or the final state.
        # A robust way is to clamp the final state.
        SOILWATER = torch.clamp(SOILWATER + (rainfall + melt) - (evap + flow), min=nearzero)

        # --- 4. Record Outputs ---
        Q_out[t] = flow
        snowfall_out[t] = snowfall
        rainfall_out[t] = rainfall
        melt_out[t] = melt
        pet_out[t] = pet
        evap_out[t] = evap
        baseflow_out[t] = baseflow
        surfaceflow_out[t] = surfaceflow
        
        # Record states
        snowpack_out[t] = SNOWPACK
        soilwater_out[t] = SOILWATER

    return (
        Q_out, snowfall_out, rainfall_out, melt_out, pet_out, 
        evap_out, baseflow_out, surfaceflow_out, snowpack_out, soilwater_out,
        SNOWPACK, SOILWATER
    )