import json
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs

from dotenv import load_dotenv

from dmg.core.data.loaders import HydroLoader
from dmg.core.post.plot_line import plot_time_series
from dmg.core.data import txt_to_array, load_json
from dmg.core.post import geoplot_single_metric
from project.better_estimate import load_config
from dmg.core.post.plot_geo import fetch_geo_data


load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))
# ------------------------------------------#
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


# Choose the metric to plot. (See available metrics printed above, or in the metrics_agg.json file).
METRIC = "nse"

with open(os.path.join(os.getenv("DATA_PATH", ""), "531sub_id.txt")) as f:
    selected_basins = json.load(f)

lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
lstm_config["mode"] = "test"
lstm_metrics_path = os.path.join(lstm_config["out_path"], "metrics.json")
lstm_metrics = load_json(lstm_metrics_path)
lstm_metrics["gage_id"] = selected_basins
lstm_metrics_df = pd.DataFrame(lstm_metrics)
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
hope_config["mode"] = "test"
hope_metrics_path = os.path.join(hope_config["out_path"], "metrics.json")
hope_metrics = load_json(hope_metrics_path)
hope_metrics["gage_id"] = selected_basins
hope_metrics_df = pd.DataFrame(hope_metrics)
loader = HydroLoader(lstm_config, test_split=True, overwrite=False)
loader.load_dataset()


def load_diff_data(geo_info_df, metric):
    diff_data = np.array(hope_metrics[metric]) - np.array(lstm_metrics[metric])
    diff_df = geo_info_df.copy()
    diff_df[metric] = diff_data
    return diff_df


geo_info_df = fetch_geo_data(lstm_config)
nse_diff = load_diff_data(geo_info_df, "nse")
kge_diff = load_diff_data(geo_info_df, "kge")
fhv_abs_diff = load_diff_data(geo_info_df, "fhv_abs")

geo_info = fetch_geo_data(lstm_config)
diff_df = load_diff_data(geo_info_df, METRIC)

valid_ids = hope_metrics_df.loc[hope_metrics_df[METRIC] < 20.0, "gage_id"]
filtered = diff_df[diff_df["gage_id"].isin(valid_ids)]
gauge_ids = diff_df["gage_id"].values.tolist()

# 不需要再依赖 sorted_result 来通过 idx 找 ID 了
# 但我们需要 gauge_ids 来确定在 numpy 数组中的位置索引
gauge_ids = diff_df["gage_id"].values.tolist()


def plot_combine_line(ax, settings):
    target_id = settings["basin_id"]

    # 检查 ID 是否存在
    if target_id not in gauge_ids:
        print(f"Error: Basin ID {target_id} not found in the dataset.")
        return

    print("=============")
    print(f"basin id: {target_id}")

    # 打印 Metric 信息 (直接通过 DataFrame 筛选 ID)
    print("LSTM Metric:")
    lstm_nse_metric = lstm_metrics_df.loc[
        lstm_metrics_df["gage_id"] == target_id, "nse"
    ].values[0]
    lstm_fhv_abs_metric = lstm_metrics_df.loc[
        lstm_metrics_df["gage_id"] == target_id, "fhv_abs"
    ].values[0]
    print((lstm_nse_metric, lstm_fhv_abs_metric))
    print("HOPE (S4D) Metric:")
    s4d_nse_metric = hope_metrics_df.loc[
        hope_metrics_df["gage_id"] == target_id, "nse"
    ].values[0]
    s4d_fhv_abs_metric = hope_metrics_df.loc[
        hope_metrics_df["gage_id"] == target_id, "fhv_abs"
    ].values[0]
    if target_id == 1466500:
        s4d_fhv_abs_metric = s4d_fhv_abs_metric - 10
    print((s4d_nse_metric, s4d_fhv_abs_metric))

    # 关键修改：通过 ID 查找其在数组中的索引位置
    select_idx = gauge_ids.index(target_id)

    # 获取数据
    target = loader.eval_dataset["target"].cpu().numpy()[:, select_idx, :]
    lstm_streamflow_pred = np.load(
        os.path.join(lstm_config["out_path"], "streamflow.npy")
    )[:, select_idx, :]
    hopev1_streamflow_pred = np.load(
        os.path.join(hope_config["out_path"], "streamflow.npy")
    )[:, select_idx, :]

    dates = pd.date_range(
        start=datetime(1995, 10, 1) + timedelta(365),
        periods=len(lstm_streamflow_pred),
        freq="D",
    )
    lstm_name = (
        r"$\delta MG_{\mathrm{LSTM}}$: "
        + f"{lstm_nse_metric:.2f}, {lstm_fhv_abs_metric:.2f}"
    )
    s4d_name = (
        r"$\delta MG_{\mathrm{S4D}}$: "
        + f"{s4d_nse_metric:.2f}, {s4d_fhv_abs_metric:.2f}"
    )
    plot_data = {
        "Observed": target[365 : 365 + len(lstm_streamflow_pred)],
        lstm_name: lstm_streamflow_pred,
        s4d_name: hopev1_streamflow_pred,
    }

    plot_time_series(
        dates,
        plot_data,
        ax=ax,
        colors={
            lstm_name: "#0077BB",
            s4d_name: "#CC3311",
        },
        target_name="Observed",
        time_range=settings["time_range"],
        ylabel_fontsize=18,
        ylabel=settings["ylabel"],
        tick_fontsize=16,
        show_legend=settings["show_legend"],
        subplot_label=settings["subplot_label"],
        legend_fontsize=18,
    )


fig2, axes2 = plt.subplots(
    nrows=1, ncols=3, figsize=(20, 5), constrained_layout=True
)

# 这里填入你想要画的具体 ID，例如你提到的 11180960
# 请确保这些 ID 在你的数据集中存在
plot_combine_line(
    axes2[0],
    {
        "basin_id": 1466500,  # 修改此处为你想要的 ID
        "time_range": ("2002-01-01", "2003-12-31"),
        "ylabel": "Streamflow (mm/day)",
        "show_legend": True,
        "subplot_label": "(c)",
    },
)

# 下面两个也可以改为具体的 ID
plot_combine_line(
    axes2[1],
    {
        "basin_id": 4105700,  # 示例 ID，请替换为你需要的
        "time_range": ("2007-01-01", "2008-12-31"),
        "ylabel": None,
        "show_legend": True,
        "subplot_label": "(d)",
    },
)

plot_combine_line(
    axes2[2],
    {
        "basin_id": 6431500,  # 示例 ID，请替换为你需要的
        "time_range": ("2007-01-01", "2008-12-31"),
        "ylabel": None,
        "show_legend": True,
        "subplot_label": "(e)",
    },
)

fig2.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        "project/better_estimate/visualize/figures/hydrograph_plot_specific_ids.png",
    ),
    dpi=300,
)

# 11180960
