# Raven v4.1 Chapter 3 Process Formula Catalog

Source:
- Raven User's and Developer's Manual v4.1, Chapter 3
- Official PDF: https://raven.uwaterloo.ca/files/v4.1/RavenUsersManual_v4.1.pdf

Notes:
- This file is a normalized extraction of the Chapter 3 process families and their governing formulas.
- Symbols were converted to ASCII where needed for readability and later implementation.
- The focus here is process names and formulas, not the surrounding explanatory text.
- Some formulas in the PDF use helper variables; those helper definitions are preserved below.

## 3.1 Precipitation Partitioning

### `RAINSNOW_DATA`
Given snowfall rate `P_s` and rainfall rate `P_r` directly from forcing:

```text
snowfall = P_s
rainfall = P_r
```

### `RAINSNOW_DINGMAN`
Fraction of precipitation falling as snow:

```text
snow_frac = clamp(1.0 - 0.5 * exp(-2.2 * (T_s - T_air)^1.3), 0, 1)
snowfall  = P * snow_frac
rainfall  = P - snowfall
```

### `RAINSNOW_HBV`
Smooth linear transition across the temperature interval `TT +/- TTI/2`:

```text
snowfall = P * clamp((TT + TTI/2 - T) / TTI, 0, 1)
rainfall = P - snowfall
```

### `RAINSNOW_UBCWM`
With mean daily temperature `T` and daily range `DeltaT`:

```text
T_max = T + 0.5 * DeltaT
T_min = T - 0.5 * DeltaT
snow_frac = f(T_min, T_max, threshold temperatures)
snowfall  = P * snow_frac
rainfall  = P - snowfall
```

The manual gives a piecewise partition based on daily temperature bounds relative to the rain/snow thresholds.

## 3.2 Abstraction / Interception / Depression Storage

### `ABST_FILL`
Fill abstraction storage up to a maximum:

```text
abstraction = min(P, S_max - S)
```

### `ABST_MAX`
Maximum abstraction rate:

```text
abstraction = min(P, A_max)
```

### `ABST_RATIO`
Fixed-ratio abstraction:

```text
abstraction = r * P
```

### `ABST_DERIVED`
Residual abstraction after other fluxes are computed:

```text
abstraction = P - sum(other outgoing precipitation-partition fluxes)
```

## 3.3 Potential Snowmelt / Potential Refreezing

### `POTMELT_DATA`
Potential melt prescribed directly by forcing:

```text
M_pot = forcing_melt
```

### `POTMELT_DEGREE_DAY`
Standard degree-day melt:

```text
M_pot = M_A * max(T - T_m, 0)
```

### `POTMELT_DD_FREEZE`
Degree-day melt or freezing depending on sign of temperature departure:

```text
M_pot = M_A * (T - T_m)
```

### `POTMELT_HBV`
HBV-style seasonal correction using potential evapotranspiration correction `c_pet`:

```text
M_pot = M_A * c_pet * max(T - T_m, 0)
```

### `POTMELT_HBV_ROS`
Rain-on-snow correction:

```text
M_pot = max(M_A * c_pet * (T - T_m) + C_ROS * rainfall * max(T - T_m, 0), 0)
```

### `POTMELT_HMETS`
HMETS time-varying degree-day factor:

```text
M_A = min(M_A,max, M_A,min * (1 + alpha_melt * cumulative_melt))
M_pot = M_A * max(T - T_m, 0)
```

### `POTMELT_RESTRICTED`
Potential melt restricted by currently available snow:

```text
M_dd   = M_A * max(T - T_m, 0)
M_pot  = min(S_snow + snowfall, M_dd)
```

## 3.4 Snow Balance / Snowpack Liquid Water

### `SNOBAL_SIMPLE_MELT`
Single-layer snow bucket:

```text
M_A     = min(M_A,max, M_A,min * (1 + alpha_melt * cumulative_melt))
M_pot   = M_A * max(T - T_m, 0)
melt    = min(S_snow, M_pot)
outflow = melt + rainfall

dS_snow/dt = snowfall - melt
d(cumulative_melt)/dt = if S_snow > 0 then melt else -cumulative_melt
```

### `SNOBAL_HBV`
HBV snowpack with liquid water retention and refreezing:

```text
M_A       = min(M_A,max, M_A,min * (1 + alpha_melt * cumulative_melt))
M_pot     = M_A * max(T - T_m, 0)
melt      = min(S_snow, M_pot)
refreeze  = min(S_liq, K_A * max(T_f - T, 0))
outflow   = max(S_liq + rainfall + melt - C_swi * S_snow, 0)

dS_snow/dt = snowfall - melt + refreeze
dS_liq/dt  = melt + rainfall - refreeze - outflow
d(cumulative_melt)/dt = if S_snow > 0 then melt else -cumulative_melt
```

### `SNOBAL_HMETS`
HMETS snow balance with variable liquid water capacity:

```text
M_A      = min(M_A,max, M_A,min * (1 + alpha_melt * cumulative_melt))
M_pot    = M_A * max(T - T_m, 0)
melt     = min(S_snow, M_pot)
refreeze = min(S_liq, K_f * max(T_f - T, 0)^F)
SWI      = max(SWI_min, SWI_max * (1 - alpha_SWI * cumulative_melt))
outflow  = max(S_liq + rainfall + melt - SWI * S_snow, 0)

dS_snow/dt = snowfall - melt + refreeze
dS_liq/dt  = melt + rainfall - refreeze - outflow
d(cumulative_melt)/dt = if S_snow > 0 then melt else -cumulative_melt
```

### `SNOBAL_CEMA_NIEGE`
Cold-content / seasonal hysteresis snow module:

```text
G = previous thermal state index
G' = update(G, snowfall, temperature, melt threshold)
M_pot = K_f * max(T, 0)
melt  = (0.9 * min(G / G_thresh, 1) + 0.1) * M_pot
liquid_out = rainfall + melt
```

The manual defines an antecedent thermal state `G` and uses it to scale actual melt.

### `SNOBAL_COLD_CONTENT`
Cold-content approach:

```text
cold_content(t+Delta_t) = cold_content(t) + energy_inputs - energy_used_for_warming
melt = max(available_melt_energy - cold_content_deficit, 0)
```

### `SNOBAL_TWO_LAYER`
Two-layer snow formulation:

```text
upper and lower snow storages are updated separately,
with melt/refreeze/water transfer between layers and a final outflow from the lower layer.
```

## 3.5 Open Water / Ponded Water Evaporation

### `OWEVAP_DATA`
Directly prescribed:

```text
E_ow = forcing_evaporation
```

### `OWEVAP_HARGREAVES_1985`

```text
E_ow = 0.0023 * R_a * (T_mean + 17.8) * sqrt(T_max - T_min)
```

### `OWEVAP_HARGREAVES`
Raven-normalized Hargreaves-style formulation:

```text
E_ow = K_H * R_a * (T_mean + C_H) * sqrt(max(T_max - T_min, 0))
```

### `OWEVAP_PENMAN_SIMPLE`
Energy + aerodynamic combination:

```text
E_ow = (Delta / (Delta + gamma)) * (R_n / lambda)
     + (gamma / (Delta + gamma)) * f(u) * (e_s - e_a)
```

### `OWEVAP_PENMAN_MONTEITH`
Penman-Monteith over open water:

```text
E_ow = [Delta (R_n - G) + rho_a c_p (e_s - e_a) / r_a] / [lambda (Delta + gamma)]
```

## 3.6 Soil Evaporation / Actual ET Limitation

### `SOILEVAP_ALL`

```text
E_s = min(PET * c_pet, S)
```

### `SOILEVAP_LINEAR`

```text
E_s = PET * c_pet * min(S / S_tension, 1)
```

### `SOILEVAP_ROOT`
Square-root scaling:

```text
E_s = PET * c_pet * sqrt(min(S / S_tension, 1))
```

### `SOILEVAP_TOPMODEL`

```text
E_s = PET * c_pet * min(S / S_tension, 1)
```

### `SOILEVAP_HBV`
HBV with snow suppression:

```text
E_s = PET * c_pet * I(snow_depth = 0) * min(S / S_tension, 1)
```

### `SOILEVAP_HBV_ORESUND`
HBV-Oresund style threshold:

```text
E_s = PET * c_pet * min(S / (LP * FC), 1)
```

### `SOILEVAP_PRMS`
PRMS-style relation using recharge storage and soil storage:

```text
E_s = f(RECHR, SOIL_MOIST, PET_remaining)
```

The manual gives a piecewise dependence on upper-zone recharge and lower soil storage.

### `SOILEVAP_SACSMA`
Sacramento tension/free water evapotranspiration:

```text
E_uztw = UZTWC / UZTWM * PET
E_uzfw = max(PET - E_uztw, 0) * g(UZFWC)
E_lztw = h(LZTWC, deficit terms, remaining PET)
```

### `SOILEVAP_GR4J`
Production-store evaporation:

```text
E_s = S * (2 - S / X1) * tanh(PET / X1) / (1 + (1 - S / X1) * tanh(PET / X1))
```

### `SOILEVAP_UBC`
UBCWM-style soil evaporation with upper/lower zone controls:

```text
E_s = PET * f(soil saturation, wilting / field-capacity thresholds)
```

### `SOILEVAP_VIC`
Field-capacity-limited evaporation:

```text
E_s = PET * c_pet * min(S / S_fc, 1)
```

### `SOILEVAP_PDM`
PDM/HYMOD-style nonlinear reduction:

```text
E_s = PET * [1 - (1 - S / S_max)^b]
```

### `SOILEVAP_GR4J_EXT`
GR4J-type evaporation with extended storage scaling:

```text
E_s = PET * F_gr4j_ext(S / S_max, parameters)
```

### `SOILEVAP_HYPR`
HYPR formulation:

```text
x = S / S_max
E_s = PET * c_pet * (1 + x - (1 + x^n)^(1/n))
```

## 3.7 Infiltration / Percolation / Recharge

### `INF_HMETS`

```text
INF = P_eff * max(1 - C_runoff * S / S_max, 0)
```

### `INF_VIC_ARNO`

```text
INF = P_eff * [1 - (1 - S / S_max)^b_inf]
```

### `INF_HBV`

```text
INF = P_eff * [1 - (S / S_max)^beta]
```

### `INF_GR4J`
GR4J production-store infiltration:

```text
INF = X1 * (1 - (S / X1)^2) * tanh(P_eff / X1) / (1 + (S / X1) * tanh(P_eff / X1))
```

### Groundwater recharge / percolation algorithms

#### `PERC_CONSTANT`

```text
PERC = min(S, P_max)
```

#### `PERC_LINEAR`

```text
PERC = k_perc * S
```

#### `PERC_POWER_LAW`

```text
PERC = k_perc * S^n
```

#### `PERC_PRMS`
PRMS gravity drainage / recharge:

```text
PERC = k_sat * (S / S_max)^exp
```

#### `PERC_SACRAMENTO`
Sacramento lower-zone demand form:

```text
PERC_demand = P_base * [1 + ZPERC * Deficit^(1 + REXP)]
PERC = PERC_demand * (UZFWC / UZFWM)
```

where `Deficit` is the normalized lower-zone moisture deficiency.

#### `PERC_GAWSER`

```text
PERC = k_gw * max(S - S_thresh, 0)
```

#### `PERC_GAWSER_CONSTRAIN`
Constrained GAWSER percolation:

```text
PERC = min(S_available, k_gw * max(S - S_thresh, 0))
```

## 3.8 Surface Runoff, Overflow, Interflow, Baseflow, Bottom Drainage

### Overflow / saturation-excess runoff

#### `OVERFLOW_ALL`

```text
Q_over = max(S - S_max, 0)
```

#### `OVERFLOW_THRESHOLD`

```text
Q_over = max(S - S_thresh, 0)
```

#### `OVERFLOW_LINEAR`

```text
Q_over = k_over * max(S - S_thresh, 0)
```

#### `OVERFLOW_NONLINEAR`

```text
Q_over = k_over * max(S - S_thresh, 0)^n
```

#### `OVERFLOW_GR4J`
Routing-store overflow:

```text
Q_over = S * [1 - (1 + ((4/9) * S / X3)^4)^(-1/4)]
```

### Baseflow / interflow / runoff from storage

#### `BASE_CONSTANT`

```text
Q_b = Q_max
```

#### `BASE_LINEAR`

```text
Q_b = k_b * S
```

#### `BASE_LINEAR_ANALYTIC`

```text
Q_b = S * (1 - exp(-k_b))
```

#### `BASE_POWER_LAW`

```text
Q_b = k_b * S^n
```

#### `BASE_GR4J`

```text
Q_b = S * [1 - (1 + ((4/9) * S / X3)^4)^(-1/4)]
```

#### `BASE_VIC`
Two-branch VIC baseflow relation:

```text
if S < Ws * S_max:
    Q_b = D_s,max * (S / (Ws * S_max))^c
else:
    Q_b = D_s,max + (D_max - D_s,max) * (S - Ws * S_max) / ((1 - Ws) * S_max)
```

#### `BASE_TOPMODEL`
Exponential recession in storage deficit:

```text
Q_b = Q_b,max * exp(-Def / m)
```

#### `BASE_THRESH_POWER`

```text
Q_b = k_b * max(S - S_thresh, 0)^n
```

#### `BASE_THRESH_STOR`

```text
Q_b = k_b * max(S - S_thresh, 0)
```

#### `INTERFLOW_PRMS`
PRMS lateral drainage:

```text
Q_if = K_if * f(S, field_capacity, slope or travel-time parameters)
```

The manual expresses this as a storage-dependent interflow release from the gravity reservoir.

### Bottom drainage

#### `BOTTOMDRAIN_LINEAR`

```text
Q_d = k_d * S
```

#### `BOTTOMDRAIN_POWER`

```text
Q_d = k_d * S^n
```

#### `BOTTOMDRAIN_THRESH`

```text
Q_d = k_d * max(S - S_thresh, 0)
```

## Implementation-Oriented Crosswalk

This catalog maps cleanly to reusable flux families:

- precipitation partitioning
- abstraction / interception
- potential melt
- snow balance
- open-water evaporation
- soil evaporation
- infiltration
- percolation / recharge
- overflow / surface runoff
- interflow
- baseflow
- bottom drainage

## Gaps / Caution

- A few manual formulas are piecewise or use helper variables defined across several lines/pages; those were normalized here instead of copied verbatim from the PDF layout.
- Where the PDF gives a long intermediate derivation but ends in a compact algorithmic expression, the compact implementation-ready form is recorded.
- The next step can use this file as the checklist for systematically expanding `src/fluxes` file-by-file.