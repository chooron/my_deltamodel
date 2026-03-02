import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Base path
base_data_path = Path(
    "/workspace/my_deltamodel/project/flex_mopex/output/camels_671/train1980-1995/no_multi/MultiHeadNetDyn_E50_R365_B100_n16_noLn_noWU_42/FlexMopexV2/NseDynAicBatchLoss/stat"
)

data_range = "test1995-2010_Ep50"
data_path = base_data_path / data_range

# Create output directory for individual basin plots
output_dir = data_path / "basin_weights_plots"
output_dir.mkdir(exist_ok=True)

# Load all weight files
weight_files = ['w_int.npy', 'w_phen.npy', 'w_snow.npy', 'w_sub.npy']
weights_data = {}

for wf in weight_files:
    weight_name = wf.replace('.npy', '')
    weights_data[weight_name] = np.load(data_path / wf).squeeze()  # Shape: (5110, 671)
    print(f"Loaded {weight_name}: {weights_data[weight_name].shape}")

# Get number of basins and time steps
n_timesteps, n_basins = weights_data['w_int'].shape
time_steps = np.arange(n_timesteps)

print(f"\nGenerating plots for {n_basins} basins...")
print(f"Output directory: {output_dir}")

# Loop through each basin and create individual plots
for basin_idx in range(n_basins):
    # Create figure with 2x2 subplots for the 4 weights
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (weight_name, weight_data) in enumerate(weights_data.items()):
        ax = axes[idx]

        # Plot the weight time series for this basin
        basin_weights = weight_data[:, basin_idx]
        ax.plot(time_steps, basin_weights, linewidth=1.5, color='steelblue')

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Weight Value', fontsize=11)
        ax.set_title(f'{weight_name.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = basin_weights.mean()
        std_val = basin_weights.std()
        min_val = basin_weights.min()
        max_val = basin_weights.max()

        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add overall title
    fig.suptitle(f'Basin {basin_idx} - Temporal Weight Evolution',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f'basin_{basin_idx:03d}_weights.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    # Print progress every 50 basins
    if (basin_idx + 1) % 50 == 0:
        print(f"  Processed {basin_idx + 1}/{n_basins} basins...")

print(f"\nAll plots saved successfully!")
print(f"Total files created: {n_basins}")
print(f"Location: {output_dir}")
