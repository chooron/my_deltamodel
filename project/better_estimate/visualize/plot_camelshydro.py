# 绘制CAMELS-Hydro属性中Baseflowindex与runoff ratio的比较图
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs

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

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore
from dmg.core.data.loaders import HydroLoader  # noqa
from dmg.core.post.plot_line import plot_time_series  # noqa
from dmg.core.data import txt_to_array, load_json  # noqa
from dmg.core.post.plot_geo import geoplot_single_metric, fetch_geo_data  # noqa
from dmg.core.post.plot_baseflowindex_scatter import plot_baseflow_scatter  # noqa
from dmg.core.data.loaders import HydroLoader  # noqa
from dmg.core.post import plot_cdf  # noqa
from project.better_estimate import load_config  # noqa

# load config
lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
lstm_out_path = lstm_config["out_path"]
hope_out_path = hope_config["out_path"]

# read predict data
lstm_gwflow = np.load(
    os.path.join(lstm_out_path, "gwflow.npy"), allow_pickle=True
)
hope_gwflow = np.load(
    os.path.join(hope_out_path, "gwflow.npy"), allow_pickle=True
)

lstm_flow = np.load(
    os.path.join(lstm_out_path, "streamflow.npy"), allow_pickle=True
)
hope_flow = np.load(
    os.path.join(hope_out_path, "streamflow.npy"), allow_pickle=True
)

lstm_ssflow = np.load(
    os.path.join(lstm_out_path, "ssflow.npy"), allow_pickle=True
)
hope_ssflow = np.load(
    os.path.join(hope_out_path, "ssflow.npy"), allow_pickle=True
)

# read forcing data
loader = HydroLoader(lstm_config, test_split=True, overwrite=False)  # type: ignore
loader.load_dataset()
forcing_data = (
    loader.eval_dataset["x_phy"]
    .detach()
    .cpu()
    .numpy()[365 : len(lstm_ssflow) + 365, :, :]
)
sum_prcp = forcing_data[:, :, 0].sum(axis=0)  # 累计降雨量

# calculate the bfi and runoff ratio
baseflow_index_lstm = (lstm_gwflow.sum(axis=0) / lstm_flow.sum(axis=0))[:, 0]
baseflow_index_hope = (hope_gwflow.sum(axis=0) / hope_flow.sum(axis=0))[:, 0]
runoff_ratio_lstm = lstm_flow.sum(axis=0)[:, 0] / sum_prcp
runoff_ratio_hope = hope_flow.sum(axis=0)[:, 0] / sum_prcp
ssflow_ratio_lstm = lstm_ssflow.sum(axis=0)[:, 0] / lstm_flow.sum(axis=0)[:, 0]
ssflow_ratio_hope = hope_ssflow.sum(axis=0)[:, 0] / lstm_flow.sum(axis=0)[:, 0]

data_path = os.getenv("DATA_PATH", "")
gage_ids = np.load(os.path.join(data_path, "gage_id.npy"), allow_pickle=True)
hydroinfos = pd.read_csv(os.path.join(data_path, "camels_hydro.txt"), sep=";")
subset_path = os.path.join(data_path, "531sub_id.txt")
with open(subset_path) as f:
    selected_basins = json.load(f)
selected_basins = [int(basin_id) for basin_id in selected_basins]
selected_hydroinfos = hydroinfos[hydroinfos["gauge_id"].isin(selected_basins)]
real_baseflow_index = np.array(selected_hydroinfos.baseflow_index.tolist())
real_runoff_ratio = np.array(selected_hydroinfos.runoff_ratio.tolist())

erorr_lstm_runoff_ratio = runoff_ratio_lstm - real_runoff_ratio
erorr_hope_runoff_ratio = runoff_ratio_hope - real_runoff_ratio
erorr_lstm_baseflow_index = baseflow_index_lstm - real_baseflow_index
erorr_hope_baseflow_index = baseflow_index_hope - real_baseflow_index

geo_info_df = fetch_geo_data(lstm_config)
lstm_hydroeval_df = geo_info_df.copy()
lstm_hydroeval_df["runoff_ratio_error"] = erorr_lstm_runoff_ratio
lstm_hydroeval_df["baseflow_index_error"] = erorr_lstm_baseflow_index
lstm_hydroeval_df["ssflow_ratio"] = ssflow_ratio_lstm
hope_hydroeval_df = geo_info_df.copy()
hope_hydroeval_df["runoff_ratio_error"] = erorr_hope_runoff_ratio
hope_hydroeval_df["baseflow_index_error"] = erorr_hope_baseflow_index
hope_hydroeval_df["ssflow_ratio"] = ssflow_ratio_hope

vmin_dict = {
    "runoff_ratio_error": -0.2,
    "baseflow_index_error": -0.6,
    "ssflow_ratio": 0.0,
}
vmax_dict = {
    "runoff_ratio_error": 0.2,
    "baseflow_index_error": 0.6,
    "ssflow_ratio": 0.7,
}

fontsize = 16
labelsize = 14

fig = plt.figure(figsize=(16, 12))

# 使用 GridSpec 控制列宽比例，右列稍窄（为 tick 留空间）
gs = fig.add_gridspec(
    3,
    3,
    width_ratios=[1, 1, 0.75],  # 右边略窄以平衡 tick
    wspace=0.02,
    hspace=0.02,
    left=0.01,
    right=0.99,
    top=0.99,
    bottom=0.01,
)

axes = []
for i in range(3):
    row_axes = []
    # 左两列地图
    for j in range(2):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.Mercator())
        row_axes.append(ax)
    # 右一列普通坐标，设为正方形比例
    ax = fig.add_subplot(gs[i, 2])
    ax.set_aspect("equal", adjustable="box")  # 正方形
    row_axes.append(ax)
    axes.append(row_axes)

for i in range(3):
    pos = axes[i][2].get_position()
    # y0 上移一点以保持居中视觉，height 缩短约10%
    axes[i][2].set_position(
        [
            pos.x0,
            pos.y0 + pos.height * 0.14,  # 上移5%
            pos.width,
            pos.height * 0.8,  # 压短10%
        ]
    )

for i, k in enumerate(
    ["runoff_ratio_error", "baseflow_index_error", "ssflow_ratio"]
):
    if k == "ssflow_ratio":
        cmap = "Reds"
    else:
        cmap = "coolwarm"
    for j, df in enumerate([lstm_hydroeval_df, hope_hydroeval_df]):
        geoplot_single_metric(
            df,
            k,
            ax=axes[i][j],
            vmin=vmin_dict[k],
            vmax=vmax_dict[k],
            cmap=cmap,
            marker="^",
            title="",
            fontsize=fontsize,
            labelsize=labelsize,
            cax_pos=[0.80, 0.1, 0.18, 0.03],
        )

plot_baseflow_scatter(
    real_runoff_ratio,
    runoff_ratio_lstm,
    runoff_ratio_hope,
    ax=axes[0][2],
    alpha=0.2,
    fontsize_labels=fontsize,
    fontsize_ticks=14,
    fontsize_legend=fontsize,
)
plot_baseflow_scatter(
    real_baseflow_index,
    baseflow_index_lstm,
    baseflow_index_hope,
    ax=axes[1][2],
    alpha=0.2,
    fontsize_labels=fontsize,
    fontsize_ticks=14,
    fontsize_legend=fontsize,
)

metrics = [
    {"ssflow_ratio": df["ssflow_ratio"].values.tolist()}
    for df in [lstm_hydroeval_df, hope_hydroeval_df]
]
plot_cdf(
    ax=axes[2][2],
    metrics=metrics,
    metric_names=["ssflow_ratio"],
    model_labels=["LSTM", "S4D"],
    xlabel="SSFLOW_RATIO",
    xbounds=(0.0, 1),
    fontsize=fontsize,
    legend_fontsize=fontsize,
    ticksize=14,
    linewidth=1.8,
    colors=["#5B84B1FF", "#D94F4FFF"],
    show_count_label=False,
    show_legend=False,
    show_median_label=False,
    count_threshold=0.6,
    axis_width=2.0,
)

fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        r"project/better_estimate/visualize/figures/camels_hydro_compare.png",
    ),
    dpi=300,
)
