# my_deltamodel

A differentiable hydrological modeling framework forked from [MHPI/generic_deltamodel](https://github.com/mhpi/generic_deltamodel), extended for large-scale continental hydrological calibration research.

---

## Paper

This repository supports the paper:

> **From Global Exploration to Local Descent: A Massively Parallel Framework and Benchmark for Continental Hydrological Calibration**
> *Submitted to Environmental Modelling and Software (under review)*

The paper presents a massively parallel calibration framework that combines global parameter space exploration (via multi-start Latin Hypercube Sampling) with local gradient-based descent, benchmarked across 40+ hydrological model structures on the CAMELS dataset.

---

## Repository Structure

```
my_deltamodel/
├── dmg/                          # Core differentiable modeling framework
│   ├── core/                     # Data loaders, samplers, metrics
│   ├── models/
│   │   ├── criterion/            # Loss functions (KGE, inverse KGE, etc.)
│   │   ├── delta_models/         # Differentiable Parameter Learning (dPL) model
│   │   ├── neural_networks/      # NN architectures + multi-start calibration
│   │   └── phy_models/
│   │       ├── core/             # 40+ differentiable hydrological models
│   │       ├── specialv2/        # Unified-input variants with routing
│   │       ├── flux/             # Reusable flux functions
│   │       └── unithydro/        # Unit hydrograph routing modules
│   └── trainers/                 # Training and calibration trainer classes
├── project/
│   └── diff_compare/             # Main experiment code (paper results)
│       ├── conf/                 # Hydra configuration files
│       ├── plots/                # Figure generation scripts
│       └── tests/                # Model simulation and evaluation scripts
└── data/                         # CAMELS dataset + ERA5-Land climate data
```

---

## Key Components

### Hydrological Models (`dmg/models/phy_models/core/`)

40+ differentiable hydrological model structures adapted from [MARRMoT](https://github.com/wknoben/MARRMoT), implemented as PyTorch operations to support gradient-based optimization. Models include:

- **HBV96**, **GR4J**, **HYMOD**, **Sacramento**, **TOPMODEL**, **VIC**, **Xinanjiang**
- **MOPEX** variants (mopex1–5), **FLEX** variants (flexb, flexi, flexis)
- Regional models: alpine1/2, australia, collie1–3, newzealand1/2, plateau, susannah1/2, etc.

Each model defines `PARAMS_BOUNDS` for parameter ranges and a `*_step()` function for single-timestep computation.

**Special variants** (`dmg/models/phy_models/specialv2/`): model versions with unified forcing inputs and unit hydrograph routing modules, used for fair cross-model comparison.

### Multi-Start Calibration (`dmg/models/neural_networks/calibrate.py`)

Implements the `Calibrate` class for generating multiple initial parameter sets per basin. Supports:
- **LHS + logit transform** (`lhs_logit`): Latin Hypercube Sampling for space-filling initialization
- **Uniform** and **normal** random initialization

Output shape: `(num_basins, num_params, num_starts)` — enables massively parallel multi-start optimization on GPU.

### Trainer (`dmg/trainers/cal_trainer.py`)

`CalTrainer` handles the full calibration loop: data loading, loss computation, optimizer/scheduler management, checkpointing, and resumption.

### Loss Functions (`dmg/models/criterion/`)

| File | Description |
|------|-------------|
| `kge_loss.py` | Kling-Gupta Efficiency loss for general streamflow calibration |
| `kge_inverse_loss.py` | Inverse-flow KGE (`Q' = 1/(Q + ε)`) for low-flow performance |

---

## Paper Experiments (`project/diff_compare/`)

Scripts reproducing the paper's calibration benchmark:

| Script | Description |
|--------|-------------|
| `calibrate_models.py` | Calibrate all standard models with KGE loss |
| `calibrate_models_invkge.py` | Calibrate all standard models with inverse KGE loss |
| `calibrate_special_models.py` | Calibrate unified-input/routing variants with KGE loss |
| `calibrate_special_models_invkge.py` | Calibrate unified-input/routing variants with inverse KGE loss |
| `batchsize_ablation.py` | Ablation study on batch size |
| `nmul_ablation.py` | Ablation study on number of multi-start initializations |
| `check_calibrate_result.py` | Validate and summarize calibration results |

Configuration files are in `conf/` and managed via [Hydra](https://hydra.cc/).

Figure generation scripts are in `plots/`.

---

## Data

The framework uses the [CAMELS](https://ral.ucar.edu/solutions/products/camels) (Catchment Attributes and Meteorology for Large-sample Studies) dataset with 559–671 US basins, plus ERA5-Land climate forcing data (1995–2010).

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥ 2.8, Hydra ≥ 1.3, NeuralHydrology ≥ 1.12, PyTorch Lightning ≥ 2.5.

---

## Other Projects

| Directory | Description |
|-----------|-------------|
| `project/blend_formula/` | Differentiable model blending with learnable process weights |
| `project/flex_mopex/` | Flexible MOPEX with learnable structural weights |
| `project/hydro_selection/` | Neural network-based hydrological model selection |
| `project/deal_hydro/` | Parameter analysis and MC dropout uncertainty estimation |
