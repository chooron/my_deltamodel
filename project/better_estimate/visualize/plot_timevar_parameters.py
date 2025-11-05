import os
import sys
import json
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


lstm_config = load_config(r"conf/config_dhbv_lstm.yaml")
hope_config = load_config(r"conf/config_dhbv_hopev1.yaml")
select_idx = (0, 730 * 2)
start_time = datetime(1995, 10, 1) + timedelta(days=365 + select_idx[0])
time_range = pd.date_range(
    start_time, freq="d", periods=select_idx[1] - select_idx[0]
)
# basin_id = 4105700
# basin_id = 6409000
basin_id = 1466500
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
lstm_timevar_params_denormalized = lstm_timevar_params_denormalized[
    :, [0, 2, 1], :
]
hope_timevar_params_denormalized = hope_timevar_params * range_vals + min_vals
hope_timevar_params_denormalized = hope_timevar_params_denormalized[
    :, [0, 2, 1], :
]

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(20, 10), constrained_layout=True
)
plot_parameters(
    lstm_timevar_params_denormalized,
    titles=["parBETA", "parBETAET", "parK1"],
    median_color="#5B84B1FF",
    fig=fig,
    ts=time_range,
    axes=axes[:, 0],
    label_fontsize=16,
    tick_fontsize=14,
    smooth_method="moving_average",
    window=1,
)
plot_parameters(
    hope_timevar_params_denormalized,
    titles=["parBETA", "parBETAET", "parK1"],
    median_color="#D94F4FFF",
    fig=fig,
    ts=time_range,
    axes=axes[:, 1],
    label_fontsize=16,
    tick_fontsize=14,
    smooth_method="moving_average",
    show_ylabel=False,
    window=5,
)
fig.savefig(
    os.path.join(
        os.getenv("PROJ_PATH"),
        f"project/better_estimate/visualize/figures/{basin_id}_params_compare.png",
    ),
    dpi=300,
)
