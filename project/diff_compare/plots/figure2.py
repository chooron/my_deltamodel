import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import wilcoxon
from matplotlib.ticker import SymmetricalLogLocator, NullFormatter
from dotenv import load_dotenv
load_dotenv()

# 为了读取 PARAM_INFO 和 NUMBER_INFO，补充路径
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.path.append(os.getenv("PROJ_PATH", ""))
from dmg.models.phy_models.core import PARAM_INFO, NUMBER_INFO # noqa

# ==========================================
# 1. 绘图风格设置 (HESS Style)
# ==========================================
plt.rcParams.update(
    {
        'font.family': 'serif',             # 声明使用衬线字体
        'font.serif': ['STIXGeneral'],  # 指定具体的衬线字体为 Times New Roman
        'mathtext.fontset': 'stix',       
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,  # X轴标签多，字体稍小
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.0,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
    }
)

# ==========================================
CSV_DIR = Path(__file__).resolve().parent / "csv"

DIF_TEST_FILE = CSV_DIR / "dif_test_kge2.csv"
MARRMOT_TEST_FILE = CSV_DIR / "marrmot_test_kge.csv"

# Use invKGE (1/Q) instead of train-set KGE
DIF_INV_FILE = CSV_DIR / "dif_test_invkge.csv"
MARRMOT_INV_FILE = CSV_DIR / "marrmot_test_invkge.csv"

def _read_metric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path, index_col=0)


def _param_count(bounds) -> int:
    if hasattr(bounds, "keys"):
        return len(bounds)
    try:
        return len(bounds)
    except Exception:
        return 0


def _percentile_limits(series: pd.Series, lo: float = 1.0, hi: float = 99.0, pad: float = 0.1):
    vals = series.dropna()
    if vals.empty:
        return None
    lo_v, hi_v = np.nanpercentile(vals, [lo, hi])
    if not np.isfinite(lo_v) or not np.isfinite(hi_v):
        return None
    if lo_v == hi_v:
        hi_v = lo_v + 1e-3
    span = hi_v - lo_v
    return (lo_v - pad * span, hi_v + pad * span)


def load_and_prepare_data():
    # 1) 读取四个 CSV (test KGE + test invKGE)
    df_dif_test = _read_metric_csv(DIF_TEST_FILE)
    df_mar_test = _read_metric_csv(MARRMOT_TEST_FILE)

    df_dif_inv = _read_metric_csv(DIF_INV_FILE)
    df_mar_inv = _read_metric_csv(MARRMOT_INV_FILE)

    # 2) 过滤可用模型列
    available_cols = (
        set(df_dif_test.columns)
        & set(df_mar_test.columns)
        & set(df_dif_inv.columns)
        & set(df_mar_inv.columns)
        & set(NUMBER_INFO.keys())
        & set(PARAM_INFO.keys())
    )
    if not available_cols:
        raise ValueError("No overlapping model columns across CSVs and metadata.")

    # 3) 按 参数数量 -> 编号 排序
    sortable = []
    for m in available_cols:
        p_count = _param_count(PARAM_INFO[m])
        sortable.append((p_count, NUMBER_INFO[m], m))
    sortable.sort()
    ordered_models = [m for _, __, m in sortable]
    ordered_params = [pc for pc, __, _ in sortable]
    ordered_labels = [f"m{NUMBER_INFO[m]:02d}" for m in ordered_models]

    # 4) 生成 records
    records = []
    for model, n_param in zip(ordered_models, ordered_params):
        # 差值计算
        delta_test = df_dif_test[model] - df_mar_test[model]
        delta_train = df_dif_inv[model] - df_mar_inv[model]
        delta_gap = (df_dif_test[model] - df_dif_inv[model]) - (
            df_mar_test[model] - df_mar_inv[model]
        )

        for val_test, val_train, val_gap in zip(
            delta_test.values, delta_train.values, delta_gap.values
        ):
            records.append(
                {
                    "model_id": model,
                    "model_label": f"m{NUMBER_INFO[model]:02d}",
                    "num_params": n_param,
                    "delta_test": val_test,
                    "delta_train": val_train,
                    "delta_gap": val_gap,
                }
            )

    df_all = pd.DataFrame(records)
    return df_all, ordered_models, ordered_params, ordered_labels

# ==========================================
# 3. 绘图核心逻辑
# ==========================================


def plot_structural_analysis(data, model_order, model_labels, param_list):
    # 创建 3行1列 画布，共享 X 轴
    # height_ratios 可以微调，这里设为均等
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True
    )

    # --- 颜色映射 (Color Mapping) ---
    # 使用单一蓝色系渐变，参数越多颜色越深
    cmap = plt.cm.Blues
    p_min, p_max = min(param_list), max(param_list)
    if p_min == p_max:
        # 避免 Normalize 端点相同导致的警告
        p_min -= 0.5
        p_max += 0.5
    norm = mcolors.Normalize(vmin=p_min, vmax=p_max)
    # 创建一个字典：{Model_Name: Color}，确保 Seaborn 能正确上色
    palette = {m: cmap(norm(p)) for m, p in zip(model_order, param_list)}

    # 根据分位数给 panel (b) 设置可视范围，抑制极端值对显示的影响
    ylim_delta_train = _percentile_limits(data["delta_train"], lo=1.0, hi=99.0, pad=0.12)

    # 定义要画的列和对应的 Y轴范围、标签
    panels = [
        {
            "col": "delta_test",
            "ylim": None,
                "ylabel": r"Test Diff (dMoT - MARRMoT)",
            "title": "(a) Test performance gain",
            "ref_line": 0,
        },
        {
            "col": "delta_train",
            "ylim": ylim_delta_train,
                "ylabel": r"KGE(1/Q) Diff (dMoT - MARRMoT)",
            "title": "(b) Low-flow gain (KGE 1/Q)",
            "ref_line": 0,
            "yscale": {"type": "symlog", "linthresh": 0.1, "linscale": 1.0},
        },
    ]

    x_positions = np.arange(len(model_order))

    # --- 循环绘制 3 个 Panel ---
    for i, ax in enumerate(axes):
        cfg = panels[i]

        # 1. 灰色斑马线背景
        for j in range(len(model_order)):
            if j % 2 == 0:
                ax.axvspan(
                    j - 0.5, j + 0.5, color="#f0f0f0", alpha=0.5, zorder=0, lw=0
                )

        # 2. 绘制参考线 (y=0)
        ax.axhline(
            cfg["ref_line"],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            zorder=1,
        )

        # 3. 绘制箱线图 (Boxplot)
        # showfliers=False: 为了整洁，通常隐藏离群点，或者设为很小的点
        sns.boxplot(
            x="model_id",
            y=cfg["col"],
            data=data,
            order=model_order,  # 关键：按参数复杂度排序
            palette=palette,  # 关键：按参数复杂度着色
            ax=ax,
            linewidth=0.8,
            width=0.7,
            showfliers=False,  # 这里隐藏了离群点以突出趋势，如需显示改为 True
            zorder=2,
        )

        # 若需要对数或 symlog 坐标轴，优先设置
        scale_cfg = cfg.get("yscale")
        if scale_cfg:
            ax.set_yscale(
                scale_cfg.get("type", "symlog"),
                linthresh=scale_cfg.get("linthresh", 0.05),
                linscale=scale_cfg.get("linscale", 1.0),
            )
            if scale_cfg.get("type", "symlog") == "symlog":
                linthresh = scale_cfg.get("linthresh", 0.1)
                locator = SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1,))
                locator.set_params(numticks=9)
                ax.yaxis.set_major_locator(locator)
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(axis="y", pad=4)

        # 应用预设 y 轴范围（尤其是 panel b 的分位数裁剪）
        if cfg["ylim"] is not None:
            ax.set_ylim(cfg["ylim"])

        # 3b. 计算胜率与显著性
        win_rates = []
        pvals = []
        col_series = cfg["col"]
        for model in model_order:
            vals = data.loc[data["model_id"] == model, col_series].dropna()
            if vals.empty:
                win_rates.append(np.nan)
                pvals.append(np.nan)
                continue
            win_rates.append((vals > 0).mean() * 100)
            try:
                _, p_val = wilcoxon(vals, alternative="greater", zero_method="pratt")
            except Exception:
                p_val = np.nan
            pvals.append(p_val)

        # 3c. 绘制右轴胜率散点
        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        ax2.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=0)
        ax2.plot(
            x_positions,
            win_rates,
            color="red",
            linestyle="-",
            linewidth=0.8,
            marker="D",
            markersize=4,
            label="Win Rate",
            zorder=10,
        )
        # 两个面板都保持对称的右轴标签
        ax2.set_ylabel("Win Rate (%)", color="red")
        ax2.tick_params(axis="y", colors="red")
        ax2.set_yticks([0, 25, 50, 75, 100])

        # 3d. 显著性标注 (p < 0.05)
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        star_offset = y_span * 0.03
        for j, (model, p_val) in enumerate(zip(model_order, pvals)):
            if p_val is not None and not np.isnan(p_val) and p_val < 0.05:
                vals = data.loc[data["model_id"] == model, col_series].dropna()
                if vals.empty:
                    continue
                capped = np.nanpercentile(vals, 95)  # 抑制离群点
                star_y = capped + star_offset
                # 保证星号停留在当前坐标范围内
                margin = y_span * 0.05
                star_y = min(max(star_y, y_min + margin), y_max - margin)
                ax.text(
                    j,
                    star_y,
                    "*",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=10,
                    zorder=11,
                )

        # 收紧左右留白
        ax.set_xlim(-0.6, len(model_order) - 0.4)

        # 4. 美化设置
        ax.set_ylabel(cfg["ylabel"], fontsize=10, labelpad=10)
        if cfg["ylim"] is not None:
            ax.set_ylim(cfg["ylim"])
        ax.set_title(
            cfg["title"], loc="left", fontsize=11, fontweight="bold", pad=6
        )

        # 去除多余边框
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 添加 "Better" 箭头指示
        # 只在 Panel (a) 画一次或者每个都画
        if i == 0:
            ax.text(
                0.02,
                0.9,
                "dMoT better \u2191",
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )

    # --- X轴设置 (只在最底下的 Panel 设置) ---
    axes[-1].set_xlabel(
        "Model ID (Sorted by Complexity: Low \u2192 High)",
        fontsize=11,
        labelpad=8,
    )
    axes[-1].set_xticklabels(model_labels, rotation=90)  # 垂直旋转标签

    # --- 添加 Colorbar 说明参数复杂度 ---
    # 创建一个假的 ScalarMappable 用于显示 Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # 放在图外右侧，覆盖两行子图总高度
    cbar_ax = fig.add_axes([1.02, 0.08, 0.018, 0.82])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Number of Parameters", rotation=270, labelpad=12)

    return fig


def main():
    df, model_order, param_list, model_labels = load_and_prepare_data()
    fig = plot_structural_analysis(df, model_order, model_labels, param_list)

    plt.savefig(
        f"{os.path.dirname(os.path.abspath(__file__))}/figures/Figure_2_Structural_Analysis.png",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
