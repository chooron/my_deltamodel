# 根据ERA5L读取的数据可视化结果
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime, timedelta
import cartopy.crs as ccrs

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore

from project.better_estimate import load_config  # noqa
from dmg.core.post.plot_geo import geoplot_single_metric, fetch_geo_data  # noqa

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
select_basins = json.load(
    open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r")
)


def load_model_inner_state(out_path):
    inner_evap = np.load(os.path.join(out_path, "AET_hydro.npy"))
    inner_soilwater = np.load(os.path.join(out_path, "soilwater.npy"))
    inner_swe = np.load(os.path.join(out_path, "SWE.npy"))
    select_basins = json.load(
        open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r")
    )

    start_date = datetime(1995, 10, 1) + timedelta(days=365)
    date_range = pd.date_range(
        start=start_date, periods=inner_evap.shape[0], freq="D"
    )
    inner_evap_df = (
        pd.DataFrame(
            data=inner_evap.squeeze(),
            index=date_range,
            columns=select_basins,
        )
        .resample("MS")
        .median()
    )
    inner_soilwater_df = (
        pd.DataFrame(
            data=inner_soilwater.squeeze(),
            index=date_range,
            columns=select_basins,
        )
        .resample("MS")
        .median()
    )
    inner_swe_df = (
        pd.DataFrame(
            data=inner_swe.squeeze(),
            index=date_range,
            columns=select_basins,
        )
        .resample("MS")
        .median()
    )
    return inner_evap_df, inner_soilwater_df, inner_swe_df


def calculate_corr(inner_evap_df, inner_soilwater_df, inner_swe_df):
    evap_corr, soilwater_corr, swe_corr = [], [], []
    reference_index = inner_evap_df.index.to_list()
    for i in range(len(select_basins)):
        basin_col_name = select_basins[i]
        for df1, df2, corr in zip(
            [inner_evap_df, inner_soilwater_df, inner_swe_df],
            [era5l_evap_df, era5l_soilwater_df, era5l_swe_df],
            [evap_corr, soilwater_corr, swe_corr],
        ):
            data_1 = df1.loc[:, basin_col_name].values
            data_2 = df2.loc[reference_index, str(basin_col_name)].values
            correlation_matrix = np.corrcoef(data_1, data_2)
            correlation_value = correlation_matrix[0, 1]
            corr.append(correlation_value)
    return np.array(evap_corr), np.array(soilwater_corr), np.array(swe_corr)


# load era5l data
era5l_evap_df = pd.read_csv(
    os.path.join(
        os.getenv("DATA_PATH"),
        "era5l_data",
        "output_e_mm_1995-2010_monthly.csv",
    ),
    parse_dates=["time"],
    index_col="time",
)
era5l_soilwater_df = pd.read_csv(
    os.path.join(
        os.getenv("DATA_PATH"),
        "era5l_data",
        "output_soilwater_mm_1995-2010_monthly.csv",
    ),
    parse_dates=["time"],
    index_col="time",
)
era5l_swe_df = pd.read_csv(
    os.path.join(
        os.getenv("DATA_PATH"),
        "era5l_data",
        "output_sd_mm_1995-2010_monthly.csv",
    ),
    parse_dates=["time"],
    index_col="time",
)
era5l_cols = era5l_evap_df.columns[:]

lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
lstm_evap_df, lstm_soilwater_df, lstm_swe_df = load_model_inner_state(
    lstm_config["out_path"]
)
hope_evap_df, hope_soilwater_df, hope_swe_df = load_model_inner_state(
    hope_config["out_path"]
)
lstm_evap_corr, lstm_soilwater_corr, lstm_swe_corr = calculate_corr(
    lstm_evap_df, lstm_soilwater_df, lstm_swe_df
)
hope_evap_corr, hope_soilwater_corr, hope_swe_corr = calculate_corr(
    hope_evap_df, hope_soilwater_df, hope_swe_df
)

# get geo data
geo_info_df = fetch_geo_data(lstm_config)
evap_corr_diff = geo_info_df.copy()
evap_corr_diff["evap_corr_diff"] = hope_evap_corr - lstm_evap_corr
print(np.nanargmax(hope_evap_corr - lstm_evap_corr))
soilwater_corr_diff = geo_info_df.copy()
soilwater_corr_diff["soilwater_corr_diff"] = (
    hope_soilwater_corr - lstm_soilwater_corr
)
swe_corr_diff = geo_info_df.copy()
swe_corr_diff["swe_corr_diff"] = hope_swe_corr - lstm_swe_corr


def plot_timeseries_compare(
    ax, index, lstm, hope, obs, title="", fontsize=12, norm=False
):
    """统一风格的折线图"""
    if norm:
        obs = obs
        lstm = lstm - lstm.mean() +  obs.mean()
        hope = hope - hope.mean() +  obs.mean()

    ax.plot(index, obs, color="black", lw=2.2, label="ERA5L")
    ax.plot(index, lstm, color="#5B84B1FF", lw=2.0, label="LSTM", linestyle="--")
    ax.plot(index, hope, color="#D94F4FFF", lw=2.0, label="S4D", linestyle="--")

    # 在图内添加注释（右上角）
    ax.text(
        0.5, 0.95, title,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="center", va="top",
    )

    ax.tick_params(labelsize=fontsize - 2)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(fontsize=fontsize - 2, frameon=False, loc="best")
    ax.set_xlim(index[0], index[-1])


fontsize = 14
labelsize = 12

fig = plt.figure(figsize=(16, 5))

# ===== 布局：两行三列，行高一致，宽度自动调整 =====
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1., 0.8],
    hspace=0.0,
    wspace=0.1,
    left=0.05,
    right=0.95,
    top=0.99,
    bottom=0.05,
)

axes = [[None] * 3 for _ in range(2)]

# ===== 第一行：地理绘图 =====
for i, (df, nm) in enumerate(
    zip(
        [evap_corr_diff, soilwater_corr_diff, swe_corr_diff],
        ["evap_corr_diff", "soilwater_corr_diff", "swe_corr_diff"],
    )
):
    ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
    geoplot_single_metric(
        df,
        nm,
        ax=ax,
        vmin=-0.5,
        vmax=0.5,
        cmap="coolwarm",
        marker="^",
        title="",
        alpha=0.6,
        fontsize=fontsize,
        labelsize=labelsize,
        cax_pos=[0.82, 0.2, 0.15, 0.03],
    )
    axes[0][i] = ax


# ===== 第二行：时间序列折线图（与上排对齐） =====
def get_maxdiff_idx(hope, lstm, thresh=0.5, isabs=True):
    diff = hope - lstm
    mask = (hope >= thresh) | (lstm >= thresh)
    if not np.any(mask):
        return np.nan
    if isabs:
        masked_diff = np.abs(diff) * mask
    else:
        masked_diff = diff * mask
    out_idx = np.nanargmax(masked_diff)
    print((np.nanmax(masked_diff), hope[out_idx], lstm[out_idx]))
    return np.nanargmax(masked_diff)


evap_idx = get_maxdiff_idx(hope_evap_corr, lstm_evap_corr, thresh=0.9, isabs=False)
soilwater_idx = get_maxdiff_idx(
    hope_soilwater_corr, lstm_soilwater_corr, thresh=0.9
)
swe_idx = get_maxdiff_idx(hope_swe_corr, lstm_swe_corr, thresh=0.9, isabs=False)

select_ids = [
    select_basins[evap_idx],
    select_basins[soilwater_idx],
    select_basins[swe_idx],
]

for i, (lstm_df, hope_df, era5l_df, basin, title) in enumerate(
    [
        (lstm_evap_df, hope_evap_df, era5l_evap_df, select_ids[0], "Evap"),
        (
            lstm_soilwater_df,
            hope_soilwater_df,
            era5l_soilwater_df,
            select_ids[1],
            "SoilWater",
        ),
        (lstm_swe_df, hope_swe_df, era5l_swe_df, select_ids[2], "SWE"),
    ]
):
    index = lstm_df.index.to_list()
    ax = fig.add_subplot(gs[1, i])
    plot_timeseries_compare(
        ax,
        index=index,
        lstm=lstm_df.loc[index, basin],
        hope=hope_df.loc[index, basin],
        obs=era5l_df.loc[index, str(basin)],
        title=f"{title} (basin {basin})",
        fontsize=fontsize,
        norm=title == "SoilWater",
    )
    # 调整纵向位置以匹配上方地图
    pos = ax.get_position()
    ax.set_position(
        [pos.x0, pos.y0 + pos.height * 0.18, pos.width, pos.height * 0.95]
    )
print((lstm_evap_corr.mean(), lstm_soilwater_corr.mean(), np.nanmean(lstm_swe_corr),))
print((hope_evap_corr.mean(), hope_soilwater_corr.mean(), np.nanmean(hope_swe_corr),))
fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        r"project/better_estimate/visualize/figures/era5l_compare.png",
    ),
    dpi=300,
)