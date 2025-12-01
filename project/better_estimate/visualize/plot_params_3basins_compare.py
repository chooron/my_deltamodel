import os
import sys
import json
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore

from dmg import ModelHandler  # noqa
from dmg.core.utils import import_trainer  # noqa
from dmg.core.data.loaders import HydroLoader  # noqa
from project.better_estimate import load_config  # noqa

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

# 加载配置和数据
config = load_config(r"conf/config_dhbv_lstm.yaml")
out_path = config["out_path"]

loader = HydroLoader(config, test_split=True, overwrite=False)
loader.load_dataset()
eval_dataset = loader.eval_dataset
model_input = eval_dataset["xc_nn_norm"]

with open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r") as f:
    selected_basins = np.array(json.load(f))


def get_timevar_parameters(config, test_epoch, basin_id=1031500, select_idx=None):
    """获取时变参数"""
    config["mode"] = "test"
    config["test"]["test_epoch"] = test_epoch
    model = ModelHandler(config, verbose=True)
    trainer_cls = import_trainer(config["trainer"])
    trainer = trainer_cls(
        config,
        model,
        eval_dataset=eval_dataset,
        verbose=True,
    )
    nn_model = trainer.model.model_dict["Hbv_2"].nn_model
    if select_idx is None:
        select_idx = (0, model_input.shape[0])

    # load model input
    select_basin_idx = np.where(selected_basins == basin_id)[0][0]
    tmp_model_input = model_input[
        select_idx[0] : select_idx[1],
        select_basin_idx : select_basin_idx + 1,
        :,
    ]
    model_out = nn_model.predict_timevar_parameters(tmp_model_input)
    timevar_params = model_out.detach().cpu().numpy()
    return timevar_params


def denormalize_params(timevar_params, par_bounds, par_names):
    """反归一化参数"""
    low_bounds = np.array([par_bounds[k][0] for k in par_names])
    up_bounds = np.array([par_bounds[k][1] for k in par_names])
    range_vec = up_bounds - low_bounds
    min_vals = low_bounds[np.newaxis, :, np.newaxis]
    range_vals = range_vec[np.newaxis, :, np.newaxis]
    denormalized = timevar_params * range_vals + min_vals
    # 重排参数顺序: [0, 2, 1] -> [parBETA, parBETAET, parK1]
    denormalized = denormalized[:, [0, 2, 1], :]
    return denormalized


# 配置参数
lstm_config = load_config(r"conf/config_dhbv_lstm_nmul_1.yaml")
hope_config = load_config(r"conf/config_dhbv_hopev1_nmul_1.yaml")

# 定义三个流域及其时间范围
basins_info = [
    {"basin_id": 1466500, "select_idx": (2983 - 730, 2983), "name": "Basin 1466500"},
    {"basin_id": 4105700, "select_idx": (4810 - 730, 4810), "name": "Basin 4105700"},
    {"basin_id": 6431500, "select_idx": (4810 - 730, 4810), "name": "Basin 6431500"},
]

# 参数边界
par_bounds = {
    "parBETA": [1.0, 6.0],
    "parK1": [0.01, 0.5],
    "parBETAET": [0.3, 5],
}
par_names = ["parBETA", "parK1", "parBETAET"]
# 反归一化后的参数顺序: beta, gamma(parBETAET), K0(parK1)
param_labels = [r"$\beta$", r"$\gamma$", r"$K_0$"]

# 获取所有流域的参数
all_lstm_params = []
all_hope_params = []
all_time_ranges = []

for basin_info in basins_info:
    basin_id = basin_info["basin_id"]
    select_idx = basin_info["select_idx"]
    
    # 计算时间范围
    start_time = datetime(1995, 10, 1) + timedelta(days=select_idx[0])
    time_range = pd.date_range(
        start_time, freq="d", periods=select_idx[1] - select_idx[0]
    )
    all_time_ranges.append(time_range)
    
    # 获取 LSTM 参数
    lstm_params = get_timevar_parameters(
        lstm_config.copy(), 100, basin_id, select_idx=select_idx
    )
    lstm_params_denorm = denormalize_params(lstm_params, par_bounds, par_names)
    all_lstm_params.append(lstm_params_denorm)
    
    # 获取 Hope/S4D 参数
    hope_params = get_timevar_parameters(
        hope_config.copy(), 5, basin_id, select_idx=select_idx
    )
    hope_params_denorm = denormalize_params(hope_params, par_bounds, par_names)
    all_hope_params.append(hope_params_denorm)

# 创建 3x3 图形
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), constrained_layout=True)

# 颜色设置
lstm_color = "#5B84B1FF"
hope_color = "#D94F4FFF"

# 绘制每个子图
for row_idx, param_idx in enumerate(range(3)):  # 三个参数
    for col_idx, basin_idx in enumerate(range(3)):  # 三个流域
        ax = axes[row_idx, col_idx]
        time_range = all_time_ranges[basin_idx]
        
        # 提取 dim=-1 中索引为 0 的参数值
        # lstm_params shape: [time, 3, 16] -> 取 [:, param_idx, 0]
        lstm_param_values = all_lstm_params[basin_idx][:, param_idx, 0]
        hope_param_values = all_hope_params[basin_idx][:, param_idx, 0]
        
        # 绘制曲线
        ax.plot(
            time_range, lstm_param_values,
            color=lstm_color, linewidth=1.5, alpha=0.8,
            label=r'$\delta MG_{\mathrm{LSTM}}$'
        )
        ax.plot(
            time_range, hope_param_values,
            color=hope_color, linewidth=1.5, alpha=0.8,
            label=r'$\delta MG_{\mathrm{S4D}}$'
        )
        
        # 设置标题（第一行显示流域名）
        if row_idx == 0:
            ax.set_title(
                f"Basin {basins_info[basin_idx]['basin_id']}",
                fontsize=18, fontweight='bold'
            )
        
        # 设置 y 轴标签（第一列显示参数名）
        if col_idx == 0:
            ax.set_ylabel(param_labels[param_idx], fontsize=18)
        
        # 设置 x 轴标签（最后一行显示时间）
        if row_idx == 2:
            ax.set_xlabel("Time", fontsize=16)
        
        # 设置刻度字体大小
        ax.tick_params(axis='both', labelsize=14)
        
        # 旋转 x 轴日期标签
        if row_idx == 2:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        else:
            ax.set_xticklabels([])
        
        # 添加图例（仅在第一个子图）
        if row_idx == 0 and col_idx == 2:
            ax.legend(fontsize=14, loc='upper right')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加子图标号
        label = f"({string.ascii_lowercase[row_idx * 3 + col_idx]})"
        ax.text(
            0.02, 0.95,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top',
            ha='left',
            zorder=100,
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='none',
                boxstyle='round,pad=0.2'
            )
        )

# 保存图形
output_path = os.path.join(
    os.getenv("PROJ_PATH"),
    "project/better_estimate/visualize/figures/3basins_params_compare.png",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.show()
