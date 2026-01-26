import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.ticker as mticker

# Consistent plotting style
plt.rcParams.update({
    "font.family": "serif",
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',   
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})

cur_dir = Path(__file__).resolve().parent

# File name mapping per model
model_file_map = {
    "xaj": "xinanjiang",
    "hbv": "hbv96",
    "hymod": "hymod",
    "collie1": "collie1",
}

# Load per-model train/test losses and convert to (1 - loss)
train_dfs, test_dfs = {}, {}
loss_cols = None
eps = 1e-6  # avoid log(0)

for model, prefix in model_file_map.items():
    train_path = cur_dir / "csv" / f"{prefix}-train-loss.csv"
    test_path = cur_dir / "csv" / f"{prefix}-test-loss.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    cols = [c for c in df_train.columns if c != "epoch"]
    if loss_cols is None:
        loss_cols = cols
    else:
        # Align columns order if needed
        df_train = df_train[["epoch"] + loss_cols]
        df_test = df_test[["epoch"] + loss_cols]

    df_train[loss_cols] = np.clip(1.0 - df_train[loss_cols], eps, None)
    df_test[loss_cols] = np.clip(1.0 - df_test[loss_cols], eps, None)

    train_dfs[model] = df_train
    test_dfs[model] = df_test

# Blue gradient for the batch-size curves
colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(loss_cols)))

# Median loss curves for each model
kge_median_files = {
    "xaj": cur_dir / "csv" / "xaj_loss_curve_medians.csv",
    "hbv": cur_dir / "csv" / "hbv_loss_curve_medians.csv",
    "hymod": cur_dir / "csv" / "hymod_loss_curve_medians.csv",
    "collie1": cur_dir / "csv" / "collie1_loss_curve_medians.csv",
}

invkge_median_files = {
    "xaj": cur_dir / "csv" / "xaj_invkge_loss_curve_medians.csv",
    "hbv": cur_dir / "csv" / "hbv_invkge_loss_curve_medians.csv",
    "hymod": cur_dir / "csv" / "hymod_invkge_loss_curve_medians.csv",
    "collie1": cur_dir / "csv" / "collie1_invkge_loss_curve_medians.csv",
}


def load_median_data(file_map):
    data = {}
    for name, path in file_map.items():
        if path.exists():
            df_med = pd.read_csv(path)
            if {"epoch", "median_loss"}.issubset(df_med.columns):
                data[name] = df_med[["epoch", "median_loss"]].copy()
            else:
                data[name] = None
        else:
            data[name] = None
    return data


kge_median_data = load_median_data(kge_median_files)
invkge_median_data = load_median_data(invkge_median_files)

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True)
axes = axes.reshape(2, 4)


def plot_loss(
    ax,
    df,
    median_df=None,
    show_ylabel=False,
    ylabel_text="",
    show_top_axis=False,
    show_bottom_xlabel=False,
    panel_label="",
):
    # Bottom axis: train/test epochs (0-100)
    for col, color in zip(loss_cols, colors):
        ax.plot(df["epoch"], df[col], label=col, color=color, linewidth=1.4)

    # Optional top axis: median curve with its own (potentially 1-10000) epoch scale
    if median_df is not None:
        train_max = df["epoch"].max()
        med_max = median_df["epoch"].max()
        scale = train_max / med_max if med_max else 1.0
        ax.plot(
            median_df["epoch"] * scale,
            median_df["median_loss"],
            label="Baseline (MARRMoT)",
            color="#D64045",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
        )

        if show_top_axis:
            top_ax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda x: x * (med_max / train_max if train_max else 1.0),
                    lambda x: x * (train_max / med_max if med_max else 1.0),
                ),
            )
            top_ax.set_xlabel("")
            top_ax.tick_params(labelsize=9)
            top_ax.grid(False)

    if panel_label:
        ax.text(
            0.98,
            0.94,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    if show_bottom_xlabel:
        ax.set_xlabel("Epoch/Iter")
        ax.tick_params(labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(ylabel_text)
    ax.grid(axis="both", linestyle="--", alpha=0.35, zorder=0)
    # Log y-scale with fixed tick set only
    ax.set_yscale("log")
    yticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())


models = ["collie1", "hymod", "xaj", "hbv"]
display_names = {
    "xaj": "m28(xinanjiang)",
    "hbv": "m37(hbv)",
    "hymod": "m29(hymod)",
    "collie1": "m01(collie1)",
}

for j, model in enumerate(models):
    med_df_kge = kge_median_data.get(model)
    plot_loss(
        axes[0, j],
        train_dfs[model],
        median_df=med_df_kge,
        show_ylabel=(j == 0),
        ylabel_text="1-KGE(Q)" if j == 0 else "",
        show_top_axis=True,
        show_bottom_xlabel=False,
        panel_label=f"({chr(97 + j)})",
    )
    # Column header indicating the model
    axes[0, j].set_title(display_names.get(model, model.upper()), fontsize=11, fontweight="bold")
    med_df_invkge = invkge_median_data.get(model)
    plot_loss(
        axes[1, j],
        test_dfs[model],
        median_df=med_df_invkge,
        show_ylabel=(j == 0),
        ylabel_text="1-KGE(1/Q)" if j == 0 else "",
        show_top_axis=False,
        show_bottom_xlabel=True,
        panel_label=f"({chr(101 + j)})",
    )

handles, labels = axes[0, 0].get_legend_handles_labels()
title_handle = plt.Line2D([], [], linestyle="none", marker=None, color="none")
fig.legend(
    [title_handle] + handles,
    ["Number of Start"] + labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(loss_cols) + 2,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

figures_dir = cur_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
save_path = figures_dir / "Figure_6_loss_curves.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
