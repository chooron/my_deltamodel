import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from adjustText import adjust_text
from dotenv import load_dotenv

load_dotenv()

# 为了读取 PARAM_INFO 和 NUMBER_INFO，补充路径
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.path.append(os.getenv("PROJ_PATH", ""))
from dmg.models.phy_models.core import PARAM_INFO, NUMBER_INFO  # noqa

# ==========================================
# 1. 统一风格与配色配置 (HESS Style)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',   
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'scatter.edgecolors': 'black',
    'axes.grid': True,            # 开启网格
    'grid.alpha': 0.3,            # 网格淡一点
    'grid.linestyle': '--'
})

# 定义核心配色 (Color Palette)
# baseline = MARRMoT, ours = dMoT
COLOR_BASE = '#D64045'  # MARRMoT
COLOR_OURS = '#2274A5'  # dMoT
MARKER_BASE = '^'
MARKER_OURS = 'o'

# ==========================================
# 2. 数据读取 (与 figure2 一致)
# ==========================================
CSV_DIR = Path(__file__).resolve().parent / "csv"
DIF_TEST_FILE = CSV_DIR / "dif_test_invkge.csv"
DIF_TRAIN_FILE = CSV_DIR / "dif_train_invkge.csv"
MARRMOT_TEST_FILE = CSV_DIR / "marrmot_test_invkge.csv"
MARRMOT_TRAIN_FILE = CSV_DIR / "marrmot_train_invkge.csv"


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


def load_data_for_generalization():
    df_dif_test = _read_metric_csv(DIF_TEST_FILE)
    df_dif_train = _read_metric_csv(DIF_TRAIN_FILE)
    df_mar_test = _read_metric_csv(MARRMOT_TEST_FILE)
    df_mar_train = _read_metric_csv(MARRMOT_TRAIN_FILE)

    available_cols = (
        set(df_dif_test.columns)
        & set(df_dif_train.columns)
        & set(df_mar_test.columns)
        & set(df_mar_train.columns)
        & set(NUMBER_INFO.keys())
        & set(PARAM_INFO.keys())
    )
    if not available_cols:
        raise ValueError("No overlapping model columns across CSVs and metadata.")

    sortable = []
    for m in available_cols:
        p_count = _param_count(PARAM_INFO[m])
        sortable.append((p_count, NUMBER_INFO[m], m))
    sortable.sort()

    ordered_models = [m for _, __, m in sortable]
    ordered_params = [pc for pc, __, _ in sortable]

    records = []
    for model, n_param in zip(ordered_models, ordered_params):
        test_dif = df_dif_test[model].values
        train_dif = df_dif_train[model].values
        gap_dif = train_dif - test_dif

        test_mar = df_mar_test[model].values
        train_mar = df_mar_train[model].values
        gap_mar = train_mar - test_mar

        # 聚合指标：中位数更稳健
        rec = {
            "model": model,
            "params": n_param,
            "test_dif_med": np.nanmedian(test_dif),
            "train_dif_med": np.nanmedian(train_dif),
            "gap_dif_med": np.nanmedian(gap_dif),
            "test_mar_med": np.nanmedian(test_mar),
            "train_mar_med": np.nanmedian(train_mar),
            "gap_mar_med": np.nanmedian(gap_mar),
        }
        records.append(rec)

    df_stats = pd.DataFrame(records)
    return df_stats

# ==========================================
# 3. 绘图逻辑
# ==========================================
def plot_enhanced_generalization(df_stats):
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 0.9], wspace=0.08)

    # 计算 Train（若未显式提供）
    train_mar = df_stats['train_mar_med']
    train_dif = df_stats['train_dif_med']
    test_mar = df_stats['test_mar_med']
    test_dif = df_stats['test_dif_med']
    gap_dif = train_dif - test_dif
    improve_test = test_dif - test_mar  # ours 相比 baseline 的测试集提升

    def _fmt_model(model_name: str) -> str:
        num = NUMBER_INFO.get(model_name)
        return f"m{int(num):02d}" if num is not None else model_name

    # =======================================================
    # Panel (a): Generalization Trajectories (Train vs Test)
    # =======================================================
    ax1 = fig.add_subplot(gs[0])

    # 范围设定（自适应数据，但保持边距）
    all_vals = np.concatenate([train_mar, train_dif, test_mar, test_dif])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        raise ValueError("No finite values to plot.")
    # 统一下界到 0.3，防止极端值拉低视野；低于下界的点将贴边显示
    limit_min = 0.0
    limit_max = min(1.0, finite_vals.max() + 0.05)
    if limit_max - limit_min < 0.2:
        limit_max = min(1.0, limit_min + 0.3)

    clip_floor = limit_min + 0.002
    train_mar_plot = train_mar.clip(lower=clip_floor)
    test_mar_plot = test_mar.clip(lower=clip_floor)
    train_dif_plot = train_dif.clip(lower=clip_floor)
    test_dif_plot = test_dif.clip(lower=clip_floor)

    # 对角线与“Robust Zone”
    ax1.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', lw=1.5, alpha=0.5, label='Perfect Generalization')
    ax1.fill_between(
        [limit_min, limit_max],
        [limit_min - 0.05, limit_max - 0.05],
        [limit_min, limit_max],
        color='gray',
        alpha=0.08,
        zorder=0,
        label='Robust Zone (Gap < 0.05)',
    )

    # 轨迹线（Baseline -> Ours），过滤微小变化
    for i in range(len(df_stats)):
        x_pair = [train_mar_plot.iloc[i], train_dif_plot.iloc[i]]
        y_pair = [test_mar_plot.iloc[i], test_dif_plot.iloc[i]]
        if abs(x_pair[1] - x_pair[0]) > 0.005 or (y_pair[1] - y_pair[0]) > 0.01:
            ax1.plot(x_pair, y_pair, color='gray', alpha=0.3, linewidth=0.8, zorder=1)

    # 散点，大小按参数量
    sizes = 50 + (df_stats['params'] - df_stats['params'].min()) * 8
    sc_base = ax1.scatter(
        train_mar_plot,
        test_mar_plot,
        s=sizes,
        c=COLOR_BASE,
        marker=MARKER_BASE,
        edgecolors='k',
        linewidth=0.5,
        alpha=0.7,
        zorder=2,
        label='Baseline (MARRMoT)',
    )
    sc_ours = ax1.scatter(
        train_dif_plot,
        test_dif_plot,
        s=sizes,
        c=COLOR_OURS,
        marker=MARKER_OURS,
        edgecolors='k',
        linewidth=0.5,
        alpha=0.9,
        zorder=3,
        label='Ours (dMoT)',
    )

    # 如有极低的 baseline（如 collie1），将其贴边并显示真实数值
    collie_mask = df_stats['model'].str.lower() == 'collie1'
    if collie_mask.any():
        ci = df_stats[collie_mask].index[0]
        ax1.text(
            train_mar_plot.iloc[ci] + 0.006,
            test_mar_plot.iloc[ci] + 0.006,
            f"{_fmt_model('collie1')}\ntrain={train_mar.iloc[ci]:.2f}\ntest={test_mar.iloc[ci]:.2f}",
            fontsize=8.5,
            fontweight='bold',
            color='#b02a37',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5),
        )

    # 标注：提升大的（红，标在 baseline 点）、泛化最好的（黑，标在 ours 点）、泛化差的（蓝，标在 ours 点）
    texts = []
    df_stats = df_stats.copy()
    df_stats['gap_dif'] = gap_dif
    df_stats['improve_test'] = improve_test
    gen_pool = df_stats[df_stats['train_dif_med'] > 0.6]
    best_gen = gen_pool.nsmallest(2, 'gap_dif') if not gen_pool.empty else df_stats.nsmallest(2, 'gap_dif')

    worst_pool = df_stats[df_stats['train_mar_med'] > 0.5]
    worst_gen = worst_pool.nlargest(2, 'gap_mar_med') if not worst_pool.empty else df_stats.nlargest(2, 'gap_mar_med')
    improve_pool = df_stats[df_stats['train_mar_med'] > 0.5]
    if improve_pool.empty:
        improve_pool = df_stats
    best_improve = improve_pool.nlargest(1, 'improve_test')

    bbox_style = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)

    def _label_rows(rows, color, use_baseline=False, xy_offset=(0.0, 0.0)):
        for _, row in rows.iterrows():
            x_raw = row['train_mar_med'] if use_baseline else row['train_dif_med']
            y_raw = row['test_mar_med'] if use_baseline else row['test_dif_med']
            x = max(x_raw, clip_floor)
            y = max(y_raw, clip_floor)
            t = ax1.text(
                x + xy_offset[0],
                y + xy_offset[1],
                _fmt_model(row['model']),
                fontsize=9,
                fontweight='bold',
                color=color,
                bbox=bbox_style,
            )
            texts.append(t)

    _label_rows(best_improve, '#c1121f', use_baseline=True, xy_offset=(0.006, 0.006))   # 提升大：红字，标在 baseline 点
    _label_rows(best_gen, '#000000', use_baseline=False)      # 泛化最好：黑字，标在 ours 点
    _label_rows(worst_gen, COLOR_OURS, use_baseline=False)    # 泛化差：蓝字（与散点一致），标在 ours 点

    if adjust_text is not None:
        try:
            adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception:
            pass

    ax1.set_xlim(limit_min, limit_max)
    ax1.set_ylim(limit_min, limit_max)
    ax1.set_xlabel('Training Performance ($KGE(1/Q)_{train}$)')
    ax1.set_ylabel('Testing Performance ($KGE(1/Q)_{test}$)')
    ax1.set_title('(a) Optimization Trajectories & Consistency', loc='left', fontweight='bold')

    ax1.annotate(
        "Better Generalization\n(Closer to Diagonal)",
        xy=(limit_min + 0.65 * (limit_max - limit_min), limit_min + 0.62 * (limit_max - limit_min)),
        xytext=(limit_min + 0.45 * (limit_max - limit_min), limit_min + 0.45 * (limit_max - limit_min)),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='#555555'),
        fontsize=9,
        color='#555555',
        ha='center',
    )
    # 说明 marker 大小代表参数量
    size_values = [5, 8, 10]
    size_handles = [
        ax1.scatter([], [], s=50 + (v - df_stats['params'].min()) * 8, facecolors='none', edgecolors='k')
        for v in size_values
    ]

    legend_main = ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.add_artist(legend_main)
    ax1.legend(handles=size_handles, labels=["≈5", "≈8", "≈10"], title='Marker Size', loc='lower right', frameon=True, fontsize=8, title_fontsize=9)
    # 仅保留右下角 Marker Size 图例，去掉额外文本避免遮挡

    # =======================================================
    # Panel (b): Grouped Complexity (with sample sizes)
    # =======================================================
    ax2 = fig.add_subplot(gs[1])

    df_stats = df_stats.copy()
    param_max = df_stats['params'].max()
    bins = [0, 5, 8, max(param_max, 10) + 5]
    labels = ['Low (≤5)', 'Medium (5-8]', 'High (>8)']
    df_stats['complexity_group'] = pd.cut(df_stats['params'], bins=bins, labels=labels, include_lowest=True, ordered=True)

    counts = df_stats['complexity_group'].value_counts()
    new_labels = [f"{l}\n($n={counts.get(l, 0)}$)" for l in labels]

    df_stats[['gap_mar_med', 'gap_dif_med']] = df_stats[['gap_mar_med', 'gap_dif_med']].clip(lower=-1, upper=1)

    df_melt = pd.melt(
        df_stats,
        id_vars=['model', 'complexity_group'],
        value_vars=['gap_mar_med', 'gap_dif_med'],
        var_name='method',
        value_name='gap'
    )
    df_melt['method'] = df_melt['method'].map({'gap_mar_med': 'MARRMoT', 'gap_dif_med': 'dMoT'})

    sns.boxplot(
        data=df_melt,
        x='complexity_group',
        y='gap',
        hue='method',
        ax=ax2,
        palette={'MARRMoT': COLOR_BASE, 'dMoT': COLOR_OURS},
        width=0.5,
        linewidth=1.0,
        showfliers=False,
        boxprops=dict(alpha=0.7),
    )

    sns.stripplot(
        data=df_melt,
        x='complexity_group',
        y='gap',
        hue='method',
        ax=ax2,
        palette={'MARRMoT': '#9b2f2f', 'dMoT': '#1c5a7a'},
        dodge=True,
        jitter=True,
        size=4,
        alpha=0.4,
        legend=False,
    )

    ax2.set_xticklabels(new_labels)
    ax2.set_xlabel('Model Complexity Group (Parameters)')
    ax2.set_ylabel('Generalization Gap ($KGE(1/Q)_{train} - KGE(1/Q)_{test}$)')
    ax2.set_title('(b) Stability Analysis by Complexity', loc='left', fontweight='bold')
    ax2.axhline(0, color='k', ls='-', lw=0.5)

    y_max = df_melt['gap'].max()
    ax2.plot([2 - 0.2, 2 + 0.2], [y_max + 0.01, y_max + 0.01], 'k-', lw=1)
    ax2.text(2, y_max + 0.013, "Gap Reduced", ha='center', va='bottom', fontsize=9, fontweight='bold')

    handles, _ = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles[:2], labels=['Baseline (MARRMoT)', 'Ours (dMoT)'], loc='upper left')

    return fig

def main():
    df_stats = load_data_for_generalization()
    fig = plot_enhanced_generalization(df_stats)
    cur_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(f"{cur_path}/figures/Figure_S1_Complexity_Generalization.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()