import os
import sys
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv("PROJ_PATH"))
from dmg.core.data import load_json
from dmg.core.post import plot_cdf
from project.better_estimate import load_config

font_family = "Times New Roman"
plt.rcParams.update(
    {
        "font.family": font_family,
        "font.serif": [font_family],
        "mathtext.fontset": "custom",
        "mathtext.rm": font_family,
        "mathtext.it": font_family,
        "mathtext.bf": font_family,
        "axes.unicode_minus": False,
    }
)

lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
tcn_config = load_config(r"conf/config_dhbv_tcn.yaml")
tsmixer_config = load_config(r"conf/config_dhbv_tsmixer.yaml")
transformer_config = load_config(r"conf/config_dhbv_transformer.yaml")

METRIC = "kge"
metrics = []
# 1. Load the evaluation metrics.
for config in [lstm_config, tcn_config, tsmixer_config, hope_config, transformer_config]:
    metrics.append(load_json(os.path.join(config["out_path"], "metrics.json")))
model_labels = [
    r"$\delta \mathrm{MG}_{\mathrm{LSTM}}$",
    r"$\delta \mathrm{MG}_{\mathrm{TCN}}$",
    r"$\delta \mathrm{MG}_{\mathrm{TimeMixer}}$",
    r"$\delta \mathrm{MG}_{\mathrm{S4D}}$",
    r"$\delta \mathrm{MG}_{\mathrm{Transformer}}$",
]
plot_config = {
    "xbounds": (0.0, 1),
    "ybounds": (0, 1),
    "show_arrow": True,
    "fontsize": 18,
    "legend_fontsize": 16,
    "ticksize": 18,
    "linewidth": 1.8,
    "show_count_label": True,
    "count_threshold": 0.6,
    "axis_width": 2.0,
}
# 2. Plot the CDF for NSE.
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True
)
plot_cdf(
    ax=axes[0],
    metrics=metrics,
    metric_names=["nse"],
    model_labels=model_labels,
    xlabel="nse".upper(),
    figure_number="(a)",
    **plot_config,
)
plot_cdf(
    ax=axes[1],
    metrics=metrics,
    metric_names=["kge"],
    model_labels=model_labels,
    xlabel="kge".upper(),
    figure_number="(b)",
    **plot_config,
)
fig.tight_layout()
fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        "project/better_estimate/visualize/figures/cdf_plot.png",
    ),
    dpi=300,
)
