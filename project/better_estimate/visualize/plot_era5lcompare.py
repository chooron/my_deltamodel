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
    ax,
    index,
    lstm,
    hope,
    obs,
    lstm_r,  # 新增：传入LSTM的相关系数
    hope_r,  # 新增：传入S4D/HOPE的相关系数
    title="",
    fontsize=14,
    norm=False,
    varname=None,
):
    """统一风格的折线图"""
    # 归一化处理
    if norm:
        obs_mean = obs.mean()
        lstm = lstm - lstm.mean() + obs_mean
        hope = hope - hope.mean() + obs_mean

    # 绘图
    ax.plot(index, obs, color="black", lw=2.2, label=varname, zorder=1)

    # 在Legend中添加 R 值
    ax.plot(
        index,
        lstm,
        color="#5B84B1FF",
        lw=2.0,
        label=r"$\delta MG_{\mathrm{LSTM}}$: " + f"($R$={lstm_r:.2f})",
        linestyle="--",
        zorder=2,
    )
    ax.plot(
        index,
        hope,
        color="#D94F4FFF",
        lw=2.0,
        label=r"$\delta MG_{\mathrm{S4D}}$: " + f"($R^2$={hope_r:.2f})",
        linestyle="--",
        zorder=3,
    )

    # 标题放在图内右上角，带白色半透明底色
    ax.text(
        0.4, 0.98, title,
        transform=ax.transAxes,
        fontsize=fontsize+2,
        fontweight="bold",
        va="top", ha="right",
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
    )

    # 坐标轴设置
    ax.tick_params(labelsize=fontsize - 2)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # Legend 设置：右上角
    leg = ax.legend(
        fontsize=18, 
        loc='upper right', 
        ncol=1,
        frameon=True,           # 必须开启边框
        fancybox=True,          # 开启圆角
        framealpha=0.7,         # 背景透明度
        facecolor='white',      # 背景颜色
        edgecolor='black',      # 边框颜色
        handlelength=1.0,       # 【新增】减小线条长度 (建议 1.0 - 1.5)
        handletextpad=0.4,      # 【新增】减小线条与文字的距离 (建议 0.3 - 0.5)
    )
    leg.get_frame().set_linewidth(1.0)
    ax.set_xlim(index[0], index[-1])



fontsize = 18
labelsize = 16

# 调整整体比例，地图更宽，折线图更协调
fig = plt.figure(figsize=(14, 10))  # 更高更宽，地图不再扁
gs = fig.add_gridspec(
    3,
    2,
    width_ratios=[1.2, 1.2],  # 左右比例调整
    hspace=0.18,  # 行间距略增
    wspace=0.13,  # 列间距略增
    left=0.05,
    right=0.97,
    top=0.97,
    bottom=0.06,
)

axes_list = []  # 用于存储所有 ax 以便后续打标签
cbar_names = {
    "evap_corr_diff": "ET Diff",
    "soilwater_corr_diff": "Soilwater Diff",
    "swe_corr_diff": "SWE Diff",
}

# ===== 第一行：地理绘图 =====
for i, (df, nm, highlight_basin_id) in enumerate(
    zip(
        [evap_corr_diff, soilwater_corr_diff, swe_corr_diff],
        ["evap_corr_diff", "soilwater_corr_diff", "swe_corr_diff"],
        [6431500, 2069700, 13331500],
    )
):
    ax = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
    geoplot_single_metric(
        df,
        nm,
        ax=ax,
        vmin=-0.5,
        vmax=0.5,
        cmap="coolwarm",
        marker="^",
        title="",  # 地图不再需要额外标题，或者你可以根据需要添加
        alpha=0.8,
        highlight_basin_ids=[highlight_basin_id],
        basin_id_col="gage_id",
        cbar_title=cbar_names[nm],
        fontsize=fontsize,
        labelsize=labelsize,
        cax_pos=[0.78, 0.2, 0.15, 0.03],
    )
    axes_list.append(ax)


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
    return out_idx


# 获取最大差异的索引
evap_idx = get_maxdiff_idx(
    hope_evap_corr, lstm_evap_corr, thresh=0.9, isabs=False
)
soilwater_idx = get_maxdiff_idx(
    hope_soilwater_corr, lstm_soilwater_corr, thresh=0.9
)
swe_idx = get_maxdiff_idx(hope_swe_corr, lstm_swe_corr, thresh=0.9, isabs=False)

select_ids = [
    select_basins[evap_idx],
    select_basins[soilwater_idx],
    select_basins[swe_idx],
]

# 对应的索引列表，用于提取 R 值
indices_list = [evap_idx, soilwater_idx, swe_idx]

# ===== 第二行：时间序列折线图 =====
for i, (
    lstm_df,
    hope_df,
    era5l_df,
    basin,
    title,
    metric_idx,
    lstm_corr_arr,
    hope_corr_arr,
) in enumerate(
    [
        (
            lstm_evap_df,
            hope_evap_df,
            era5l_evap_df,
            select_ids[0],
            "ET (mm)",
            indices_list[0],
            lstm_evap_corr,
            hope_evap_corr,
        ),
        (
            lstm_soilwater_df,
            hope_soilwater_df,
            era5l_soilwater_df,
            select_ids[1],
            "SoilWater (mm)",
            indices_list[1],
            lstm_soilwater_corr,
            hope_soilwater_corr,
        ),
        (
            lstm_swe_df,
            hope_swe_df,
            era5l_swe_df,
            select_ids[2],
            "SWE (mm)",
            indices_list[2],
            lstm_swe_corr,
            hope_swe_corr,
        ),
    ]
):
    index = lstm_df.index.to_list()
    ax = fig.add_subplot(gs[i, 1])

    # 获取该站点的 R 值
    val_r_lstm = lstm_corr_arr[metric_idx]
    val_r_hope = hope_corr_arr[metric_idx]

    # 获取序号字母 (d/e/f)
    letters = ['d', 'e', 'f']
    title_with_label = f"({letters[i]}) USGS {basin}"
    plot_timeseries_compare(
        ax,
        index=index,
        lstm=lstm_df.loc[index, basin],
        hope=hope_df.loc[index, basin],
        obs=era5l_df.loc[index, str(basin)],
        lstm_r=val_r_lstm,  # 传入 R 值
        hope_r=val_r_hope,  # 传入 R 值
        title=title_with_label,  # 标题与序号合并
        fontsize=fontsize,
        varname=title,
        norm=title == "SoilWater (mm)",
    )
    pos = ax.get_position()
    # pos.x0: 左侧位置 (不变)
    # pos.y0: 底部位置 (增加这个值，图就会向上跑)
    # pos.width: 宽度 (不变)
    # pos.height: 高度 (不变)

    # 调整建议：
    # 如果想贴得很紧：尝试 + 0.1 到 + 0.15
    # 如果想保留标题位置但靠近一点：尝试 + 0.05 到 + 0.08

    axes_list.append(ax)

# ===== 添加 (a), (b), (c)... 序号标签 =====
import string

letters = string.ascii_lowercase  # 'abcdef...'

for i, ax in enumerate(axes_list):
    # 将序号放在左上角外部
    # transform=ax.transAxes 使得坐标基于子图大小 (0,0)左下 (1,1)右上
    # x=-0.1, y=1.05 大致位于左上角外部

    # 针对不同行的微调（可选，通常统一设置即可）
    x_offset = 0.08
    y_offset = 0.85

    # 只给前三个地图加序号，后面三个折线图已在标题内合并序号
    if i < 3:
        ax.text(
            x_offset,
            y_offset,
            f"({letters[i]})",
            transform=ax.transAxes,
            fontsize=fontsize + 2,
            fontweight="bold",
            va="bottom",
            ha="left",
            bbox=dict(
                facecolor='white',  # 背景色
                alpha=0.8,          # 透明度
                edgecolor='none',   # 边框颜色 (设为 'none' 隐藏边框，设为 'black' 显示边框)
                pad=0.2             # 内边距 (让背景稍微比文字大一点)
            )
        )

print(
    (
        lstm_evap_corr.mean(),
        lstm_soilwater_corr.mean(),
        np.nanmean(lstm_swe_corr),
    )
)
print(
    (
        hope_evap_corr.mean(),
        hope_soilwater_corr.mean(),
        np.nanmean(hope_swe_corr),
    )
)
fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        r"project/better_estimate/visualize/figures/era5l_compare.png",
    ),
    dpi=300,
)
