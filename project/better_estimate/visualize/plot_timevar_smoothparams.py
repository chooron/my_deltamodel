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


hope_config = load_config(r"conf/config_dhbv_hopev4.yaml")
# basin_id = 1466500 # 2983-730*2:2983
# select_idx = (2983-730, 2983)
basin_id = 6431500 # 4810-730*2:4810
select_idx1 = (2983-730, 2983)
select_idx2 = (4810-730, 4810)
start_time1 = datetime(1995, 10, 1) + timedelta(days=select_idx1[0])
start_time2 = datetime(1995, 10, 1) + timedelta(days=select_idx2[0])
time_range1 = pd.date_range(
    start_time1, freq="d", periods=select_idx1[1] - select_idx1[0]
)
time_range2 = pd.date_range(
    start_time2, freq="d", periods=select_idx2[1] - select_idx2[0]
)
hope_timevar_params1 = get_timevar_parameters(
    hope_config, 100, 1466500, select_idx=select_idx1
)
hope_timevar_params2 = get_timevar_parameters(
    hope_config, 100, 6431500, select_idx=select_idx2
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
hope_timevar_params_denormalized1 = hope_timevar_params1 * range_vals + min_vals
hope_timevar_params_denormalized1 = hope_timevar_params_denormalized1[
    :, [0, 2, 1], :
]
hope_timevar_params_denormalized2 = hope_timevar_params2 * range_vals + min_vals
hope_timevar_params_denormalized2 = hope_timevar_params_denormalized2[
    :, [0, 2, 1], :
]

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(16, 10), constrained_layout=True
)
plot_parameters(
    hope_timevar_params_denormalized1,
    titles=[r"$\beta$", r"$\gamma$", r"$K_0$"],
    median_color="#D94F4FFF",
    fig=fig,
    ts=time_range1,
    axes=axes[:, 0],
    label_fontsize=20,
    tick_fontsize=18,
    legend_fontsize=20,
    smooth_method="moving_average",
    window=1,
    model_name=r'',
    legend_pos='top-right'
)
plot_parameters(
    hope_timevar_params_denormalized2,
    titles=[r"$\beta$", r"$\gamma$", r"$K_0$"],
    median_color="#D94F4FFF",
    fig=fig,
    ts=time_range2,
    axes=axes[:, 1],
    label_fontsize=20,
    tick_fontsize=18,
    legend_fontsize=20,
    smooth_method="moving_average",
    show_ylabel=False,
    window=1,
    model_name=r'',
    legend_pos='top-left'
)
for i, ax in enumerate(axes.flat):
    # 生成序号：(a), (b), (c)...
    label = f"({string.ascii_lowercase[i]})"
    
    ax.text(
        0.02, 0.95,          # x, y 位置：0.02是靠左，0.95是靠上 (相对于图框)
        label,
        transform=ax.transAxes, # 使用相对坐标系 (0到1)
        fontsize=20,            # 字体大小，通常比 tick_fontsize 大一点
        fontweight='bold',      # 加粗更醒目
        va='top',               # 垂直对齐：顶部对齐
        ha='left',              # 水平对齐：左对齐
        zorder=100,              # 确保文字在最上层
        bbox=dict(
            facecolor='white',  # 背景颜色
            alpha=0.8,          # 透明度 0.8
            edgecolor='none',   # 无边框线 (如果你想要黑框，这里改成 'black')
            boxstyle='round,pad=0.2' # 可选：圆角矩形，pad控制文字周围留白大小
        )
    )
fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        "project/better_estimate/visualize/figures/s5d_params_visual.png",
    ),
    dpi=300,
)
