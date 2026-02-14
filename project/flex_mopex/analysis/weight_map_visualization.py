import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from pathlib import Path

# Paths for weight arrays
alpha = "0_05"
base_data_path = Path(
    "/workspace/my_deltamodel/project/flex_mopex/output/camels_671/"
    "train1980-1995/no_multi/"
    "MultiHeadNet_E50_R365_B100_n16_noLn_noWU_42/"
    f"FlexMopexV1/alpha_{alpha}/NseAicBatchLoss/stat"
)
data_range = "test1995-2010_Ep50"

weight_files = {
    "w_int": base_data_path / data_range / "w_int.npy",
    "w_phen": base_data_path / data_range / "w_phen.npy",
    "w_snow": base_data_path / data_range / "w_snow.npy",
    "w_sub": base_data_path / data_range / "w_sub.npy",
}


def reduce_to_station(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0)
    if arr.ndim == 3:
        # Typical shape: (time, stations, 1)
        if arr.shape[-1] == 1:
            return arr.mean(axis=0).squeeze(-1)
        if arr.shape[0] == 1:
            return arr[0].mean(axis=0)
        return arr.mean(axis=(0, 1))
    raise ValueError(f"Unsupported weight shape: {arr.shape}")


print("Loading weight arrays:")
weights_per_type = {}
for name, path in weight_files.items():
    print(f"  {name}: {path}")
    raw = np.load(path)
    weights_per_type[name] = reduce_to_station(raw)
    print(f"    raw shape: {raw.shape} -> station shape: {weights_per_type[name].shape}")

# CAMELS basin locations
camels_shp_path = "/workspace/my_deltamodel/data/camels_loc/camels_671_loc.shp"
gdf = gpd.read_file(camels_shp_path)

# gage_id order
gage_id_file = "/workspace/my_deltamodel/data/gage_id.txt"
with open(gage_id_file, "r") as f:
    gage_ids = [line.strip() for line in f.readlines()]

sample_weight = next(iter(weights_per_type.values()))
if len(gage_ids) != sample_weight.shape[0]:
    raise ValueError(
        f"gage_id count ({len(gage_ids)}) != station count ({sample_weight.shape[0]})"
    )

gage_id_col = None
for col in ["GAGE_ID", "gage_id", "STAID", "gauge_id", "hru_id"]:
    if col in gdf.columns:
        gage_id_col = col
        break

if gage_id_col is None:
    for col in gdf.columns:
        if col != "geometry":
            gage_id_col = col
            break

gdf[gage_id_col] = gdf[gage_id_col].astype(str).str.zfill(8)
gage_id_to_idx = {gid: idx for idx, gid in enumerate(gage_ids)}

for name, values in weights_per_type.items():
    gdf[name] = gdf[gage_id_col].map(
        lambda gid: values[gage_id_to_idx[gid]] if gid in gage_id_to_idx else np.nan
    )

# Output directory
output_dir = "/workspace/my_deltamodel/project/flex_mopex/analysis"
os.makedirs(output_dir, exist_ok=True)

usa_bounds = gdf.total_bounds

for name in weights_per_type.keys():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    lons = gdf["lon"].values
    lats = gdf["lat"].values
    weights_values = gdf[name].values

    vmin = 0.0
    vmax = 1.0

    scatter = ax.scatter(
        lons,
        lats,
        c=weights_values,
        cmap="RdBu_r",
        s=30,
        edgecolors="black",
        linewidths=0.3,
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
    )

    ax.set_title(f"{name} - Station Weights", fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(usa_bounds[0], usa_bounds[2])
    ax.set_ylim(usa_bounds[1], usa_bounds[3])

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04, aspect=18)
    cbar.set_label("Weight", fontsize=11)

    output_path = os.path.join(output_dir, f"{alpha}", f"{name}_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

print("All weight maps generated.")
