import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# --- 1. 绘图风格设置 ---
plt.rcParams.update({
    # 核心字体设置
    'font.family': 'serif',             # 声明使用衬线字体
    'font.serif': ['STIXGeneral'],  # 指定具体的衬线字体为 Times New Roman
    'mathtext.fontset': 'stix',  
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.0,
    'figure.dpi': 300
})

# Minimal meaningful delta to counter sampling uncertainty (cf. Knoben et al., 2025)
UNCERTAINTY_BAND = 0.05

# ==========================================
# 2. 数据加载 (保持不变)
# ==========================================
def clean_model_name(full_name):
    parts = full_name.split('_')
    if len(parts) >= 3:
        return parts[2]
    return full_name

def load_and_align_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "csv")

    # Primary KGE
    path_marrmot_kge = os.path.join(csv_dir, "marrmot_test_kge.csv")
    path_dmot_kge = os.path.join(csv_dir, "dif_test_kge.csv")

    # Inverse KGE (optional)
    path_marrmot_invkge = os.path.join(csv_dir, "marrmot_test_invkge.csv")
    path_dmot_invkge = os.path.join(csv_dir, "dif_test_invkge.csv")

    print(f"Loading MARRMoT KGE: {path_marrmot_kge}")
    print(f"Loading dMoT KGE: {path_dmot_kge}")

    df_marrmot_kge = pd.read_csv(path_marrmot_kge, index_col=0)
    df_dmot_kge = pd.read_csv(path_dmot_kge, index_col=0)

    df_marrmot_kge.columns = [clean_model_name(c) for c in df_marrmot_kge.columns]
    df_dmot_kge.columns = [clean_model_name(c) for c in df_dmot_kge.columns]

    common_models_kge = df_marrmot_kge.columns.intersection(df_dmot_kge.columns)
    common_basins_kge = df_marrmot_kge.index.intersection(df_dmot_kge.index)

    aligned_marrmot_kge = df_marrmot_kge.loc[common_basins_kge, common_models_kge]
    aligned_dmot_kge = df_dmot_kge.loc[common_basins_kge, common_models_kge]

    aligned_marrmot_invkge = None
    aligned_dmot_invkge = None

    if os.path.exists(path_marrmot_invkge) and os.path.exists(path_dmot_invkge):
        print(f"Loading MARRMoT invKGE: {path_marrmot_invkge}")
        print(f"Loading dMoT invKGE: {path_dmot_invkge}")

        df_marrmot_invkge = pd.read_csv(path_marrmot_invkge, index_col=0)
        df_dmot_invkge = pd.read_csv(path_dmot_invkge, index_col=0)

        df_marrmot_invkge.columns = [clean_model_name(c) for c in df_marrmot_invkge.columns]
        df_dmot_invkge.columns = [clean_model_name(c) for c in df_dmot_invkge.columns]

        common_models_inv = df_marrmot_invkge.columns.intersection(df_dmot_invkge.columns)
        common_basins_inv = df_marrmot_invkge.index.intersection(df_dmot_invkge.index)

        aligned_marrmot_invkge = df_marrmot_invkge.loc[common_basins_inv, common_models_inv]
        aligned_dmot_invkge = df_dmot_invkge.loc[common_basins_inv, common_models_inv]
    else:
        print("invKGE files not found; skipping panel (b)")

    return aligned_marrmot_kge, aligned_dmot_kge, aligned_marrmot_invkge, aligned_dmot_invkge

# ==========================================
# 3. 核心绘图函数 (已针对直方图美观度优化)
# ==========================================
def plot_joint_panel(fig, outer_grid_pos, x, y, x_med, y_med, xlabel, ylabel, panel_title, 
                     limit_min=-1.0, limit_max=1.0):
    
    # --- 0. 布局设置 ---
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid_pos, 
                                 width_ratios=[4, 0.7], height_ratios=[0.7, 4],
                                 wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # --- 1. 强力数据清洗 (解决 NaN 问题) ---
    # 必须同时剔除 nan 和 inf (无穷大)
    mask_valid = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask_valid], y[mask_valid]

    if len(x_clean) == 0:
        print(f"Warning: No valid data for {panel_title}")
        return None

    # 原始数据的中位数用于统计和边缘标注，保持与直方图一致
    median_x_all = np.median(x_clean)
    median_y_all = np.median(y_clean)

    # --- A. 主图绘制 (Hexbin) ---
    ax_main.set_xlim(limit_min, limit_max)
    ax_main.set_ylim(limit_min, limit_max)
    ax_main.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', lw=1, alpha=0.6, zorder=1)
    
    # 为了 Hexbin 颜色好看，只画视图范围内的数据
    # (如果不筛选，颜色会被极值 -100 拉伸，导致 -1~1 之间全是浅色)
    hex_mask = (x_clean >= limit_min - 0.5) & (x_clean <= limit_max + 0.1) & \
               (y_clean >= limit_min - 0.5) & (y_clean <= limit_max + 0.1)
    x_hex = x_clean[hex_mask]
    y_hex = y_clean[hex_mask]
    
    if len(x_hex) > 0:
        hb = ax_main.hexbin(x_hex, y_hex, gridsize=40, cmap='Blues', mincnt=1,
                            extent=(limit_min, limit_max, limit_min, limit_max),
                            linewidths=0.02, edgecolors='face', bins='log', alpha=0.85, zorder=2)
        
        # --- 统计指标 (与原始数据一致) ---
        # 1. Win Rate: 仅当提升超过不确定性阈值才记为赢
        delta = y_clean - x_clean
        win_rate = np.mean(delta > UNCERTAINTY_BAND) * 100
        within_band = np.mean((delta >= -UNCERTAINTY_BAND) & (delta <= UNCERTAINTY_BAND)) * 100
        
        # 2. Median Delta: 使用全量有效数据的中位数，确保与边缘直方图标注一致
        median_delta = np.median(delta)
        
        stats_text = (
            f"Win Rate (> {UNCERTAINTY_BAND:.2f}): {win_rate:.1f}%\n"
            f"Neutral band (±{UNCERTAINTY_BAND:.2f}): {within_band:.1f}%\n"
            f"Median $\Delta$: {median_delta:+.2f}"
        )
        
        ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray'), zorder=11)
    else:
        hb = None

    # --- Model Median (模型中位数) ---
    # 过滤掉 NaN 的中位数点
    mask_med = np.isfinite(x_med) & np.isfinite(y_med)
    ax_main.scatter(x_med[mask_med], y_med[mask_med], s=35, marker='s', linewidth=1.2,
                    facecolors='none', edgecolors='black', 
                    label='Model Median', zorder=10)
    
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    # 面板标注放在主图右下角，避免挤占上方空间
    ax_main.text(0.97, 0.03, panel_title, transform=ax_main.transAxes, ha='right', va='bottom',
                 fontsize=11, fontweight='bold')

    # --- B. 边缘分布图 (解决 直线/扁平 问题) ---
    # 核心修复：只提取 limit 范围内的数据给 KDE。
    # 比如只取 -1 到 1 之间的数据。如果不这样做，Seaborn 会看到 -100，
    # 从而把带宽设得巨大，导致 -1 到 1 之间看起来像一条平线。
    
    def draw_safe_kde(ax, data, median_value, is_vertical=False):
        # 1. 严格截断数据
        data_visible = data[(data >= limit_min) & (data <= limit_max)]
        
        # 2. 如果数据太少，就不画了，防止报错
        if len(data_visible) < 5:
            ax.axis('off')
            return
        
        # 3. 使用未截断数据的中位数，保持与统计文本一致
        median_val = median_value

        # 4. 绘图
        if not is_vertical:
            # X轴上方图
            sns.kdeplot(x=data_visible, ax=ax, color='#555555', fill=True, alpha=0.3, 
                        linewidth=1, bw_adjust=0.8) # bw_adjust 控制平滑度
            # 画中位数虚线
            if limit_min <= median_val <= limit_max:
                ax.axvline(median_val, color='k', ls=':', lw=1)
                # 标数字
                ax.text(median_val, ax.get_ylim()[1]*0.1, f'{median_val:.2f}', 
                        fontsize=8, ha='center', va='bottom', color='k', fontweight='bold')
        else:
            # Y轴右侧图
            sns.kdeplot(y=data_visible, ax=ax, color='#555555', fill=True, alpha=0.3, 
                        linewidth=1, bw_adjust=0.8)
            if limit_min <= median_val <= limit_max:
                ax.axhline(median_val, color='k', ls=':', lw=1)
                ax.text(ax.get_xlim()[1]*0.1, median_val, f'{median_val:.2f}', 
                        fontsize=8, ha='left', va='center', color='k', fontweight='bold', rotation=270)
        
        ax.axis('off')

    # 调用绘图
    draw_safe_kde(ax_histx, x_clean, median_x_all, is_vertical=False)
    draw_safe_kde(ax_histy, y_clean, median_y_all, is_vertical=True)

    return hb

# ==========================================
# 4. 执行
# ==========================================
df_marrmot_kge, df_dmot_kge, df_marrmot_invkge, df_dmot_invkge = load_and_align_data()

# KGE(Q)
x_flat = df_marrmot_kge.values.flatten()
y_flat = df_dmot_kge.values.flatten()
x_med = df_marrmot_kge.median(axis=0).values
y_med = df_dmot_kge.median(axis=0).values

# KGE(1/Q) if available
has_invkge = df_marrmot_invkge is not None and df_dmot_invkge is not None
if has_invkge:
    x_flat_inv = df_marrmot_invkge.values.flatten()
    y_flat_inv = df_dmot_invkge.values.flatten()
    x_med_inv = df_marrmot_invkge.median(axis=0).values
    y_med_inv = df_dmot_invkge.median(axis=0).values

# === 关键设置 ===
# 手动锁死显示范围，忽略 -12241 这种异常值
# KGE 的有效范围通常在 -1 到 1 之间，再低就没意义了
LIMIT_MIN = -1.02
LIMIT_MAX = 1.02

print(f"Force Plot Limits: [{LIMIT_MIN}, {LIMIT_MAX}]")

# 绘图
n_panels = 2 if has_invkge else 1
fig = plt.figure(figsize=(10, 5))
outer_gs = GridSpec(1, n_panels, figure=fig, width_ratios=[1] * n_panels,
                    wspace=0.25, left=0.08, right=0.9, bottom=0.15, top=0.9)

# Panel (a) - KGE(Q)
hb1 = plot_joint_panel(fig, outer_gs[0],
                        x_flat, y_flat,
                        x_med, y_med,
                        "Baseline (MARRMoT)", "Ours (dMoT)", "(a) KGE(Q)",
                        limit_min=LIMIT_MIN, limit_max=LIMIT_MAX)

# Panel (b) - KGE(1/Q) if data exists
hb2 = None
if has_invkge:
    hb2 = plot_joint_panel(fig, outer_gs[1],
                           x_flat_inv, y_flat_inv,
                           x_med_inv, y_med_inv,
                           "Baseline (MARRMoT)", "Ours (dMoT)", "(b) KGE(1/Q)",
                           limit_min=LIMIT_MIN, limit_max=LIMIT_MAX)

# Colorbar
if hb1 is not None:
    cbar_left = 0.92 if n_panels == 2 else 0.86
    cbar_ax = fig.add_axes([cbar_left, 0.15, 0.015, 0.6])
    cbar = fig.colorbar(hb1, cax=cbar_ax)
    cbar.set_label('Count (log scale)', rotation=270, labelpad=15)

save_path_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", 'Figure_1_Global.png')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {save_path_png}")
