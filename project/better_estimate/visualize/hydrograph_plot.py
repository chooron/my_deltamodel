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
METRIC = "fhv_abs"

with open(os.path.join(os.getenv("DATA_PATH", ""), "531sub_id.txt")) as f:
    selected_basins = json.load(f)
    
lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
lstm_config["mode"] = "test"
lstm_metrics_path = os.path.join(lstm_config["out_path"], "metrics.json")
lstm_metrics = load_json(lstm_metrics_path)
lstm_metrics['gage_id'] = selected_basins
lstm_metrics_df = pd.DataFrame(lstm_metrics)
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
hope_config["mode"] = "test"
hope_metrics_path = os.path.join(hope_config["out_path"], "metrics.json")
hope_metrics = load_json(hope_metrics_path)
hope_metrics['gage_id'] = selected_basins
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
sorted_result = filtered.sort_values(by=METRIC, ascending=False)


def plot_combine_line(ax, settings):
    print("=============")
    print(f"basin id: {sorted_result['gage_id'].values[settings['idx']]}")
    print(
        lstm_metrics_df.loc[
            lstm_metrics_df["gage_id"]
            == sorted_result["gage_id"].values[settings["idx"]],
            METRIC,
        ]
    )
    print(
        hope_metrics_df.loc[
            hope_metrics_df["gage_id"]
            == sorted_result["gage_id"].values[settings["idx"]],
            METRIC,
        ]
    )
    select_idx = gauge_ids.index(
        sorted_result["gage_id"].values[settings["idx"]]
    )
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
    plot_data = {
        "target": target[365 : 365 + len(lstm_streamflow_pred)],
        "LSTM": lstm_streamflow_pred,
        "S4D": hopev1_streamflow_pred,
    }

    plot_time_series(
        dates,
        plot_data,
        ax=ax,
        colors={"LSTM": "#0077BB", "S4D": "#CC3311"},
        time_range=settings["time_range"],
        ylabel_fontsize=18,
        ylabel=settings["ylabel"],
        tick_fontsize=16,
        legend_fontsize=18,
    )


fig2, axes2 = plt.subplots(
    nrows=1, ncols=3, figsize=(20, 5), constrained_layout=True
)
# 0: ('1999-01-01', '1999-12-31'), 1: ('2004-01-01', '2004-12-31'), 2: ('2003-01-01', '2003-12-31')
plot_combine_line(
    axes2[0],
    {
        "idx": 0,
        "time_range": ("2007-03-01", "2007-12-01"),
        "ylabel": "Streamflow (mm/day)",
    },
)
plot_combine_line(
    axes2[1],
    {"idx": 1, "time_range": ("2002-01-01", "2002-12-01"), "ylabel": None},
)
plot_combine_line(
    axes2[2],
    {"idx": -3, "time_range": ("2008-01-01", "2008-12-01"), "ylabel": None},
)
# plot_combine_line(
#     axes2[3],
#     {"idx": -2, "time_range": ("2002-01-01", "2003-01-01"), "ylabel": None},
# )
fig2.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        "project/better_estimate/visualize/figures/hydrograph_plot.png",
    ),
    dpi=300,
)

# 11180960
