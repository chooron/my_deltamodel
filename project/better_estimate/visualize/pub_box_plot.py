import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg.core.data import load_json
from dmg.core.post.plot_statbox import plot_boxplots
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

lstm_config = load_config(r"conf/config_dhbv_pub_lstm.yaml")
hopev1_config = load_config(r"conf/config_dhbv_pub_hopev1.yaml")
lstm_criteria_path = os.path.join(lstm_config["out_path"], "metrics.json")
hopev1_criteria_path = os.path.join(hopev1_config["out_path"], "metrics.json")
lstm_df_list = []
hopev1_df_list = []
# basin_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
basin_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
for i in basin_ids:
    # for i in [11, 12, 13, 14, 15, 16, 17]:
    new_lstm_criteria_path = re.sub(
        r"pub_basin_\d+", f"pub_basin_{i}", lstm_criteria_path
    )
    new_hopev1_criteria_path = re.sub(
        r"pub_basin_\d+", f"pub_basin_{i}", hopev1_criteria_path
    )
    lstm_criteria_df = pd.DataFrame(load_json(new_lstm_criteria_path))
    hopev1_criteria_df = pd.DataFrame(load_json(new_hopev1_criteria_path))
    lstm_df_list.append(lstm_criteria_df)
    hopev1_df_list.append(hopev1_criteria_df)

group_labels1 = [f"PUB-{i + 1}" for i in range(10)]
group_labels2 = [f"PUR-{i + 1}" for i in range(7)]
criterion = "kge"
model_labels = [
    r"$\delta \mathrm{MG}_{\mathrm{LSTM}}$",
    r"$\delta \mathrm{MG}_{\mathrm{S4D}}$",
]
gridspec_options = {"width_ratios": [10, 7], "wspace": 0.1, "hspace": 0.15}
fig, ax = plt.subplots(
    nrows=2, ncols=2, figsize=(16, 8), gridspec_kw=gridspec_options
)
common_style = {
    'fontsize_label': 14,
    'fontsize_tick': 12,
    'fontsize_legend': 12,
    'subplot_label_fontsize': 16,
    'list1_color': '#3C5488', # Nature Blue
    'list2_color': '#E64B35', # Nature Red
    'legend_labels': model_labels
}

# --- 1. 左上图 (ax[0, 0]) ---
# 在这里设置图例显示在 "图片外的左上方"
plot_boxplots(
    lstm_df_list[:10], hopev1_df_list[:10],
    column_name="nse", ylabel="NSE", 
    group_labels=group_labels1, ylim=(0, 1), 
    ax=ax[0, 0],
    subplot_label="(a)",   # <--- 右上角显示 (a)
    show_legend=True,      # <--- 开启图例
    legend_loc="lower right",      # 锚点设为左下角
    legend_bbox=(1.2, 0.95),        # 将锚点移到坐标轴顶部的左侧 (即外部左上)
    **common_style
)

# --- 2. 左下图 (ax[1, 0]) ---
plot_boxplots(
    lstm_df_list[:10], hopev1_df_list[:10],
    column_name="kge", ylabel="KGE", 
    group_labels=group_labels1, ylim=(0, 1), 
    ax=ax[1, 0],
    subplot_label="(b)",   # <--- 右上角显示 (b)
    show_legend=False,     # <--- 关闭图例
    **common_style
)

# --- 3. 右上图 (ax[0, 1]) ---
plot_boxplots(
    lstm_df_list[10:], hopev1_df_list[10:],
    column_name="nse", ylabel="NSE", 
    group_labels=group_labels2, ylim=(0, 1), 
    ax=ax[0, 1],
    subplot_label="(c)",   # <--- 右上角显示 (c)
    show_legend=False,     # <--- 关闭图例
    **common_style
)

# --- 4. 右下图 (ax[1, 1]) ---
plot_boxplots(
    lstm_df_list[10:], hopev1_df_list[10:],
    column_name="kge", ylabel="KGE", 
    group_labels=group_labels2, ylim=(0, 1), 
    ax=ax[1, 1],
    subplot_label="(d)",   # <--- 右上角显示 (d)
    show_legend=False,     # <--- 关闭图例
    **common_style
)
fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        f"project/better_estimate/visualize/figures/pur_pub_boxplot.png",
    ),
    dpi=300,
)
