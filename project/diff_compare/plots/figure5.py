import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path

# 1. 应用你的 HESS 期刊风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',   
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.0,
    'figure.dpi': 300
})

# 2. 读取真实数据
cur_dir = Path(__file__).resolve().parents[0]
csv_path = cur_dir / "csv" / "stats_calc_time.csv"
df = pd.read_csv(csv_path)

# 3. 指标计算
N_SAMPLES = 559
df["T_epoch"] = df["T_step"] * df["k"]
df["f_update"] = 60.0 / (df["T_step"])

# 4. 提取离散变量
batches = sorted(df["batch_size"].unique())
pred_lens = sorted(df["pred_len"].unique())

# 5. 基线配置（按模型）
baseline_by_model = {
    # model_name: (median_iter_time, median_min_update)
    "hymod": (0.0274973, 2182.04),
    "xinanjiang": (0.0312969, 1917.12),
    "collie1": (0.00782065, 7671.12),
    "hbv96": (0.0286971, 2090.8),
}

model_order = ["collie1", "hymod", "xinanjiang", "hbv96"]
display_name_by_model = {
    "collie1": "m01 (collie1)",
    "hymod": "m29 (hymod)",
    "xinanjiang": "m28 (xinanjiang)",
    "hbv96": "m37 (hbv)",
}

# 6. 绘图逻辑（2 行 4 列：每列一个模型，上行为 T_epoch，下行为 f_update）
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=False)

# 定义颜色 (每一条 PredLen 线一种颜色)
colors = list(plt.cm.Blues(np.linspace(0.35, 0.9, len(pred_lens))))

# 关键技巧：设置 X 轴的错位 (Dodge)
x_indices = np.arange(len(batches))
width = 0.18

def plot_metric(
    ax,
    data,
    metric_key,
    y_label,
    baseline_value,
    baseline_label="baseline",
    baseline_text_x=0.98,
    baseline_text_y_offset_frac=0.012,
    y_locator=None,
    y_formatter=None,
):
    for i, p_len in enumerate(pred_lens):
        sub_df = data[data["pred_len"] == p_len]
        stats = (
            sub_df.groupby("batch_size")[metric_key]
            .agg(["mean", "std"])
            .reindex(batches)
        )

        offset = (i - (len(pred_lens) - 1) / 2) * width
        x_pos = x_indices + offset

        ax.bar(
            x_pos,
            stats["mean"],
            width=width,
            color=colors[i],
            alpha=0.85,
            label=f"{p_len}",
            zorder=2,
        )
        ax.errorbar(
            x_pos,
            stats["mean"],
            yerr=stats["std"],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=3,
        )

    ax.axhline(
        y=baseline_value,
        color="black",
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * baseline_text_y_offset_frac
    baseline_text_y = baseline_value + y_offset
    baseline_text_y = baseline_value if baseline_text_y > y_max else baseline_text_y
    ax.text(
        baseline_text_x,
        baseline_text_y,
        baseline_label,
        transform=ax.get_yaxis_transform(),
        ha="right" if baseline_text_x > 0.5 else "left",
        va="bottom",
        fontsize=9,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.25",
        },
    )
    if y_locator is not None:
        ax.yaxis.set_major_locator(y_locator)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(batches)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

for col_idx, model in enumerate(model_order):
    model_df = df[df["model"] == model]

    iter_baseline, update_baseline = baseline_by_model.get(model, (None, None))

    # 上行：T_epoch
    plot_metric(
        axes[0, col_idx],
        model_df,
        "T_epoch",
        r"$T_{epoch}$ (s)",
        iter_baseline,
        baseline_text_x=0.98,
    )

    # 下行：f_update（千为单位显示）
    plot_metric(
        axes[1, col_idx],
        model_df,
        "f_update",
        r"$f_{update}$ ($\times 10^3$ updates/min)",
        update_baseline,
        baseline_text_x=0.04,
        y_locator=ticker.MultipleLocator(5000),
        y_formatter=ticker.FuncFormatter(lambda y, _: f"{y/1000:g}"),
    )

    axes[0, col_idx].set_title(display_name_by_model.get(model, model))

# 仅底行显示 x 轴标签/刻度，最左列显示 y 轴标签，其他子图隐藏对应标签以节省空间
for col_idx in range(len(model_order)):
    axes[0, col_idx].set_xlabel("")
    axes[0, col_idx].tick_params(labelbottom=False)

for row in range(2):
    for col_idx in range(1, len(model_order)):
        axes[row, col_idx].set_ylabel("")

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
for idx, label in enumerate(panel_labels):
    r, c = divmod(idx, len(model_order))
    axes[r, c].text(
        0.02,
        0.98,
        label,
        transform=axes[r, c].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )

handles, labels = axes[0, 0].get_legend_handles_labels()
title_handle = Line2D([], [], linestyle="none", marker=None, color="none")
fig.legend(
    [title_handle] + handles,
    ["Predict Length (ρ)"] + labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=len(pred_lens) + 1,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

save_path_png = cur_dir / "figures" / "Figure_5_param_update.png"
plt.savefig(save_path_png, dpi=300, bbox_inches="tight")