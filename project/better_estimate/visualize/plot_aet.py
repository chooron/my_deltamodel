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
from dmg.core.post.plot_parameters import plot_parameters  # noqa
from dmg.core.post.plot_statbox import plot_temporal_volatility_boxplot  # noqa
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

config = load_config(r"conf/config_dhbv_lstm.yaml")
out_path = config["out_path"]

loader = HydroLoader(config, test_split=True, overwrite=False)
loader.load_dataset()
eval_dataset = loader.eval_dataset
model_input = eval_dataset["xc_nn_norm"]

with open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r") as f:
    selected_basins = np.array(json.load(f))


def load_model_inner_state(out_path):
    inner_evap = np.load(os.path.join(out_path, "AET_full.npy"))
    inner_soilwater = np.load(os.path.join(out_path, "SM_full.npy"))
    return inner_evap, inner_soilwater


def get_timevar_parameters(
    config, test_epoch, basin_id=1031500, select_idx=None
):
    # load model
    config["mode"] = "test"
    config["test"]["test_epoch"] = 100
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


# basin_id = 1466500 # 2983-730*2:2983
# select_idx = (2983-730, 2983)
basin_id = 6431500  # 4810-730*2:4810
select_basin_idx = np.where(selected_basins == basin_id)[0][0]
select_idx = (4810 - 730, 4810)
start_time = datetime(1995, 10, 1) + timedelta(days=select_idx[0])
time_range = pd.date_range(
    start_time, freq="d", periods=select_idx[1] - select_idx[0]
)
lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
lstm_aet_full, lstm_sm_full = load_model_inner_state(lstm_config["out_path"])
hope_aet_full, hope_sm_full = load_model_inner_state(hope_config["out_path"])
select_lstm_aet = lstm_aet_full[
    select_idx[0] : select_idx[1], select_basin_idx, :
]  # 730 * 1 * 16
select_lstm_sm = lstm_sm_full[
    select_idx[0] : select_idx[1], select_basin_idx, :
]  # 730 * 1 * 16
select_hope_aet = hope_aet_full[
    select_idx[0] : select_idx[1], select_basin_idx, :
]  # 730 * 1 * 16
select_hope_sm = hope_sm_full[
    select_idx[0] : select_idx[1], select_basin_idx, :
]  # 730 * 1 * 16

# basin_id = 4105700
# basin_id = 6409000
lstm_timevar_params = get_timevar_parameters(
    lstm_config, 100, basin_id, select_idx=select_idx
)
hope_timevar_params = get_timevar_parameters(
    hope_config, 100, basin_id, select_idx=select_idx
)
par_bounds = {
    "parBETA": [1.0, 6.0],
    "parK1": [0.01, 0.5],
    "parBETAET": [0.3, 5],
}
par_names = ["parBETA", "parK1", "parBETAET"]
low_bounds = np.array([par_bounds[k][0] for k in par_names])
up_bounds = np.array([par_bounds[k][1] for k in par_names])
range_vec = up_bounds - low_bounds
min_vals = low_bounds[np.newaxis, :, np.newaxis]
range_vals = range_vec[np.newaxis, :, np.newaxis]
lstm_timevar_params_denormalized = lstm_timevar_params * range_vals + min_vals
lstm_timevar_params_denormalized = lstm_timevar_params_denormalized[:, 2, :]
lstm_display_data = np.stack(
    [lstm_timevar_params_denormalized, select_lstm_sm, select_lstm_aet], axis=1
)

hope_timevar_params_denormalized = hope_timevar_params * range_vals + min_vals
hope_timevar_params_denormalized = hope_timevar_params_denormalized[:, 2, :]
hope_display_data = np.stack(
    [hope_timevar_params_denormalized, select_hope_sm, select_hope_aet], axis=1
)
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(20, 10),
    constrained_layout=True,
    width_ratios=[1, 1, 0.5],
)
plot_parameters(
    lstm_display_data,
    titles=[r"$\gamma$", "Soilwater", "ET"],
    median_color="#5B84B1FF",
    fig=fig,
    ts=time_range,
    axes=axes[:, 0],
    label_fontsize=20,
    tick_fontsize=18,
    legend_fontsize=20,
    smooth_method="moving_average",
    window=1,
    model_name=r"$\delta MG_{\mathrm{LSTM}}$",
)
plot_parameters(
    hope_display_data,
    titles=[r"$\gamma$", "Soilwater", "ET"],
    median_color="#D94F4FFF",
    fig=fig,
    ts=time_range,
    axes=axes[:, 1],
    label_fontsize=20,
    tick_fontsize=18,
    legend_fontsize=20,
    smooth_method="moving_average",
    show_ylabel=False,
    window=1,
    model_name=r"$\delta MG_{\mathrm{S4D}}$",
)
plot_temporal_volatility_boxplot(
    data_list=[lstm_display_data, hope_display_data],
    axes=axes[:, 2],
    model_names=[r"$\delta MG_{\mathrm{LSTM}}$", r"$\delta MG_{\mathrm{S4D}}$"],
    titles="",
    title_fontsize=20,
    label_fontsize=20,
    tick_fontsize=18,
    var_names=["$\gamma$", "Soilwater", "ET"],
    colors=["#5B84B1FF", "#D94F4FFF"],
)
for i, ax in enumerate(axes.flat):
    # 生成序号：(a), (b), (c)...
    label = f"({string.ascii_lowercase[i]})"

    ax.text(
        0.02,
        0.95,  # x, y 位置：0.02是靠左，0.95是靠上 (相对于图框)
        label,
        transform=ax.transAxes,  # 使用相对坐标系 (0到1)
        fontsize=20,  # 字体大小，通常比 tick_fontsize 大一点
        fontweight="bold",  # 加粗更醒目
        va="top",  # 垂直对齐：顶部对齐
        ha="left",  # 水平对齐：左对齐
        zorder=100,  # 确保文字在最上层
        bbox=dict(
            facecolor="white",  # 背景颜色
            alpha=0.8,  # 透明度 0.8
            edgecolor="none",  # 无边框线 (如果你想要黑框，这里改成 'black')
            boxstyle="round,pad=0.2",  # 可选：圆角矩形，pad控制文字周围留白大小
        ),
    )

fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        f"project/better_estimate/visualize/figures/{basin_id}_inner_process.png",
    ),
    dpi=300,
)
