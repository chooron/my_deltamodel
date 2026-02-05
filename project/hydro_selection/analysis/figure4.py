import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from dotenv import load_dotenv

# ==========================================
# 0. 环境与路径配置
# ==========================================
load_dotenv()

# 添加项目根目录以导入 dmg
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(os.getenv("PROJ_PATH", "."))

# 尝试导入模型元数据，如果环境不匹配则使用 Mock 数据防止报错
from dmg.models.phy_models.core import PARAM_INFO # noqa

# 路径配置
DATA_PATH = Path("/workspace/my_deltamodel/data")
CSV_DIR = Path(__file__).resolve().parent / "csv"
SHP_PATH = DATA_PATH / "camels_loc" / "camels_671_loc.shp"
ID_559_PATH = DATA_PATH / "559sub_id.txt"

# 文件名配置 (假设 Log KGE 文件存在)
FILE_CONFIG = {
    "norm": {
        "dif": CSV_DIR / "dif_test_kge.csv",
        "mar": CSV_DIR / "marrmot_test_kge.csv",
    },
    "log": {
        # 使用 invKGE (1/Q) 作为低流量指标
        "dif": CSV_DIR / "dif_test_invkge.csv",
        "mar": CSV_DIR / "marrmot_test_invkge.csv",
    },
}

# ==========================================
# 1. 绘图风格设置 (HESS Style)
# ==========================================
plt.rcParams.update(
    {
        'font.family': 'serif',             # 声明使用衬线字体
        'font.serif': ['STIXGeneral'],  # 指定具体的衬线字体为 Times New Roman
        'mathtext.fontset': 'stix',  
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "scatter.edgecolors": "none",
    }
)

# 配色: RdBu_r (Red=Positive/Good, Blue=Negative/Bad)
# 强制 Center=0
CMAP_NAME = "RdBu_r"
VMIN, VMAX = -0.2, 0.2
norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=0, vmax=VMAX)
COLOR_GAIN = "#D64045"  # Red for gain
COLOR_LOSS = "#2274A5"  # Blue for loss
Y_LIMITS = (-4, 10)  # Adjustable y-axis limits for ranked curves
LEFT_Y_LIMITS = (-1.0, 2.5)  # Tighter range for primary (left) axis

# ==========================================
# 2. 数据处理函数
# ==========================================


def _get_param_count(model_name):
    """获取模型参数个数"""
    bounds = PARAM_INFO.get(model_name, {})
    if hasattr(bounds, "keys"):
        return len(bounds)
    return len(bounds) if isinstance(bounds, (list, tuple)) else 0


def _pad_basin_id(value):
    """Normalize basin IDs to zero-padded 8-digit strings."""
    sval = str(value).strip()
    if sval == "":
        return sval
    try:
        return f"{int(float(sval)) :08d}"
    except ValueError:
        return sval


def _resolve_model_list(model_list, available_columns):
    """Resolve model names to available metric columns, applying simple aliases."""
    resolved = []
    for name in model_list:
        if name in available_columns:
            resolved.append(name)
        elif name == "hbv96" and "hbv" in available_columns:
            resolved.append("hbv")
        # silently skip if not present
    return resolved


def load_and_process_data():
    """
    核心数据加载逻辑:
    1. 读取 559 ID 列表
    2. 读取 Shapefile 并过滤
    3. 读取 Normal 和 Log 的 CSV
    4. 计算 Delta (Dif - Mar)
    5. 按 Simple/Complex/All 分组聚合
    """
    print("Loading data...")

    # 1. Load Basin IDs
    if not ID_559_PATH.exists():
        raise FileNotFoundError(f"Missing ID file: {ID_559_PATH}")
    # 假设 txt 是一列 ID
    subset_path = os.path.join(
        os.getenv("DATA_PATH", "."), "559sub_id.txt"
    )
    with open(subset_path) as f:
        valid_ids_raw = json.load(f)
    valid_ids = [_pad_basin_id(x) for x in np.atleast_1d(valid_ids_raw).tolist()]
    print(valid_ids)

    # 2. Load Shapefile
    if not SHP_PATH.exists():
        raise FileNotFoundError(f"Missing Shapefile: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH)
    print(gdf.head())

    # 确保 gage_id 是字符串以便匹配
    gdf["gage_id"] = gdf["gage_id"].apply(_pad_basin_id)

    # 过滤 559
    gdf = gdf[gdf["gage_id"].isin(valid_ids)].copy()
    if gdf.empty:
        raise ValueError("Filtered GeoDataFrame is empty; check gage_id matching and input files.")

    # 统一到 WGS84，并存储经纬度用于点图
    gdf = gdf.to_crs(epsg=4326)
    # gdf["lon"] = gdf.geometry.x
    # gdf["lat"] = gdf.geometry.y

    # 设置 gage_id 为索引，方便后续 join
    gdf = gdf.set_index("gage_id")

    # 3. Load Metrics & Calculate Delta
    # 容器用于存储不同流况的聚合数据
    metrics_map = {}  # {'norm': df_delta, 'log': df_delta}

    for flow_type, paths in FILE_CONFIG.items():
        if not paths["dif"].exists() or not paths["mar"].exists():
            print(f"Warning: {flow_type} CSVs not found. Skipping.")
            continue

        df_dif = pd.read_csv(paths["dif"])
        df_mar = pd.read_csv(paths["mar"])

        for df_name, df in ("dif", df_dif), ("mar", df_mar):
            if "basin_id" not in df.columns:
                raise KeyError(f"Missing basin_id column in {df_name} CSV: {paths[df_name]}")
            df["basin_id"] = df["basin_id"].apply(_pad_basin_id)

        df_dif = df_dif[df_dif["basin_id"].isin(valid_ids)].set_index("basin_id")
        df_mar = df_mar[df_mar["basin_id"].isin(valid_ids)].set_index("basin_id")

        # 计算 Delta (dMoT - MARRMoT)
        # 确保列名一致
        common_cols = df_dif.columns.intersection(df_mar.columns)
        df_delta = df_dif[common_cols] - df_mar[common_cols]
        metrics_map[flow_type] = df_delta

    if not metrics_map:
        raise ValueError("No valid metric data loaded.")

    # 4. Group Models
    all_models = metrics_map["norm"].columns.tolist()
    param_counts = {m: _get_param_count(m) for m in all_models}

    groups = {
        "All": all_models,
        "Simple": [m for m, p in param_counts.items() if p < 5],
        "Complex": [m for m, p in param_counts.items() if p > 10],
    }

    print(
        f"Model Groups: All({len(groups['All'])}), Simple({len(groups['Simple'])}), Complex({len(groups['Complex'])})"
    )

    # 5. Aggregate per Basin (Mean across models in group)
    # 结果结构: data_ready[row_name][col_name] = Series(index=basin_id)
    data_ready = {}

    for group_name, model_list in groups.items():
        data_ready[group_name] = {}

        # Normal Flow Delta
        if "norm" in metrics_map:
            cols_norm = _resolve_model_list(model_list, metrics_map["norm"].columns)
            if cols_norm:
                # axis=1 mean: 对每个流域，计算该组模型的平均提升
                data_ready[group_name]["norm"] = metrics_map["norm"][
                    cols_norm
                ].mean(axis=1)
            else:
                data_ready[group_name]["norm"] = pd.Series(dtype=float)

        # Low Flow Delta
        if "log" in metrics_map:
            cols_log = _resolve_model_list(model_list, metrics_map["log"].columns)
            if cols_log:
                data_ready[group_name]["log"] = metrics_map["log"][cols_log].mean(
                    axis=1
                )
            else:
                data_ready[group_name]["log"] = pd.Series(dtype=float)

    return gdf, data_ready


def _plot_ranked_curve(ax, data_norm, data_log, show_xlabel=False):
    """Plot ranked improvement curves with separate y-axes for KGE(Q) and KGE(1/Q)."""
    # Drop NaNs to avoid gaps
    y_norm = np.sort(np.asarray(pd.Series(data_norm).dropna().values))
    y_log = np.sort(np.asarray(pd.Series(data_log).dropna().values))

    if y_norm.size == 0 or y_log.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return []

    x_norm = np.linspace(0, 100, len(y_norm))
    x_log = np.linspace(0, 100, len(y_log))

    ax_right = ax.twinx()

    line_norm = ax.plot(x_norm, y_norm, color="black", lw=1.3, label="KGE(Q)", zorder=3)[0]
    line_log = ax_right.plot(x_log, y_log, color="0.35", lw=1.2, ls="--", label="KGE(1/Q)", zorder=2)[0]

    ax.fill_between(x_norm, 0, y_norm, where=y_norm >= 0, color=COLOR_GAIN, alpha=0.26)
    ax.fill_between(x_norm, 0, y_norm, where=y_norm < 0, color=COLOR_LOSS, alpha=0.26)
    ax_right.fill_between(x_log, 0, y_log, where=y_log >= 0, color=COLOR_GAIN, alpha=0.16)
    ax_right.fill_between(x_log, 0, y_log, where=y_log < 0, color=COLOR_LOSS, alpha=0.16)

    pct_gain_q = np.mean(y_norm > 0) * 100
    pct_gain_lq = np.mean(y_log > 0) * 100
    ax.text(
        2,
        LEFT_Y_LIMITS[1] * 0.78,
        f"KGE(Q) Improve: {pct_gain_q:.1f}%",
        color=COLOR_GAIN,
        fontweight="bold",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax_right.text(
        2,
        Y_LIMITS[1] * 0.60,
        f"KGE(1/Q) Improve: {pct_gain_lq:.1f}%",
        color="0.35",
        fontweight="bold",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    # Zero baselines on both axes
    ax.axhline(0, color="k", lw=0.8, ls="-")
    ax_right.axhline(0, color="0.4", lw=0.6, ls=":")

    ax.set_xlim(0, 100)
    ax.set_ylim(LEFT_Y_LIMITS)
    ax_right.set_ylim(Y_LIMITS)

    if show_xlabel:
        ax.set_xlabel("Percentile of Basins (%)")
    else:
        ax.set_xticklabels([])

    ax.set_ylabel(r"Sorted Gain $\Delta$KGE(Q)", color="black")
    ax_right.set_ylabel(r"Sorted Gain $\Delta$KGE(1/Q)", color="0.35")

    ax.grid(True, linestyle=":", alpha=0.4)
    ax.tick_params(axis="y", colors="black")
    ax_right.tick_params(axis="y", colors="0.35")

    return [line_norm, line_log]


# ==========================================
# 3. 绘图核心逻辑
# ==========================================


def plot_spatial_statistical_composite(gdf, data_structure):
    # 布局: 3行 x 3列，第三列为排序增益曲线
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1, 1, 0.7],
        height_ratios=[1, 1, 1],
        wspace=0.03,
        hspace=0.06,
    )

    rows = ["All", "Simple", "Complex"]
    row_titles = [
        "Grand Ensemble (All Models)",
        "Low (<5 Params)",
        "High (>8 Params)",
    ]

    # 预计算地图范围，稍作留白
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.02, (maxy - miny) * 0.02

    # 循环绘制

    for i, group in enumerate(rows):
        # 获取该行的数据
        s_norm = data_structure[group].get("norm")
        s_log = data_structure[group].get("log")

        # --- Col 1: Normal Flow Map (points) ---
        ax_map1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        gdf_plot1 = gdf.copy()
        gdf_plot1["val"] = s_norm

        sc1 = ax_map1.scatter(
            gdf_plot1["lon"],
            gdf_plot1["lat"],
            c=gdf_plot1["val"],
            cmap=CMAP_NAME,
            norm=norm,
            s=16,
            linewidths=0,
            alpha=0.65,
            transform=ccrs.PlateCarree(),
        )
        ax_map1.set_extent(
            [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
            crs=ccrs.PlateCarree(),
        )
        for feat in (
            cfeature.LAND.with_scale("50m"),
            cfeature.OCEAN.with_scale("50m"),
            cfeature.BORDERS.with_scale("50m"),
            cfeature.STATES.with_scale("50m"),
            cfeature.COASTLINE.with_scale("50m"),
        ):
            ax_map1.add_feature(feat, edgecolor="0.55", facecolor="none", linewidth=0.6)
        ax_map1.set_aspect(1.3, adjustable="box")
        ax_map1.set_axis_off()

        if i == 0:
            ax_map1.set_title(
                r"(a) KGE(Q) Map ($\Delta$KGE(Q))",
                fontsize=11,
                fontweight="bold",
            )
        # Row Label (左侧)
        ax_map1.text(
            -0.1,
            0.5,
            row_titles[i],
            transform=ax_map1.transAxes,
            va="center",
            ha="right",
            fontsize=11,
            rotation=90,
            fontweight="bold",
        )

        # --- Col 2: Low Flow Map (points) ---
        ax_map2 = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
        gdf_plot2 = gdf.copy()
        gdf_plot2["val"] = s_log

        sc2 = ax_map2.scatter(
            gdf_plot2["lon"],
            gdf_plot2["lat"],
            c=gdf_plot2["val"],
            cmap=CMAP_NAME,
            norm=norm,
            s=16,
            linewidths=0,
            alpha=0.65,
            transform=ccrs.PlateCarree(),
        )
        ax_map2.set_extent(
            [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
            crs=ccrs.PlateCarree(),
        )
        for feat in (
            cfeature.LAND.with_scale("50m"),
            cfeature.OCEAN.with_scale("50m"),
            cfeature.BORDERS.with_scale("50m"),
            cfeature.STATES.with_scale("50m"),
            cfeature.COASTLINE.with_scale("50m"),
        ):
            ax_map2.add_feature(feat, edgecolor="0.55", facecolor="none", linewidth=0.6)
        ax_map2.set_aspect(1.3, adjustable="box")
        ax_map2.set_axis_off()

        if i == 0:
            ax_map2.set_title(
                r"(b) KGE(1/Q) Map ($\Delta$KGE(1/Q))",
                fontsize=11,
                fontweight="bold",
            )

        # --- Col 3: Ranked Improvement Curve ---
        ax_stat = fig.add_subplot(gs[i, 2])
        handles = _plot_ranked_curve(ax_stat, s_norm, s_log, show_xlabel=(i == len(rows) - 1))

        if i == 0:
            ax_stat.set_title(
                "(c) Gain Distribution Profile", fontsize=11, fontweight="bold"
            )
            if handles:
                labels = [h.get_label() for h in handles]
                ax_stat.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9)

    # --- Global Colorbar (Bottom) ---
    # 创建一个 dummy mappable
    sm = plt.cm.ScalarMappable(cmap=CMAP_NAME, norm=norm)
    sm.set_array([])

    # 位置: [left, bottom, width, height] — 放在前两列下方，矩形端点
    cbar_ax = fig.add_axes([0.12, 0.02, 0.52, 0.02])
    cbar = fig.colorbar(
        sm, cax=cbar_ax, orientation="horizontal", extend="neither"
    )
    cbar.set_label(
                r"Performance Gain ($\Delta$KGE(Q), $\Delta$KGE(1/Q))",
        labelpad=4,
    )
    # 刻度
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.ax.set_xticklabels(
        ["baseline (MARRMoT) better", "neutral", "ours (dMoT) better"]
    )

    # Caption Print
    caption = """
    Figure 4: Spatial-statistical composite analysis of performance gains.
    Rows aggregate results for: (top) the entire model suite, (middle) simple models, and (bottom) complex models.
    Columns distinguish between normal flow (Col 1) and low-flow (Col 2) regimes. 
    Red markers indicate basins where dMoT outperforms MARRMoT (Delta > 0).
    Column 3 compares the distribution of gains, highlighting robustness in low-flow simulations (Orange).
    """
    print(caption)

    return fig


# ==========================================
# 4. 主程序
# ==========================================
def main():
    try:
        # 1. 加载和处理数据
        gdf, data_structure = load_and_process_data()

        # 2. 绘图
        fig = plot_spatial_statistical_composite(gdf, data_structure)
        cur_dir = Path(__file__).resolve().parents[0]

        # 3. 保存
        save_path_png = cur_dir / Path("figures/Figure_4_Spatial_Stats.png")

        if not save_path_png.parent.exists():
            save_path_png.parent.mkdir(parents=True)

        plt.savefig(save_path_png, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path_png}")

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
