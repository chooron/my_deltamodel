import json
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

from dmg import ModelHandler  # noqa
from dmg.core.utils import import_trainer  # noqa
from dmg.core.post.plot_parameters import plot_parameters  # noqa
from dmg.core.data.loaders import HydroLoader  # noqa
from project.better_estimate import load_config  # noqa


config = load_config(r"conf/config_dhbv_lstm.yaml")
out_path = config["out_path"]

loader = HydroLoader(config, test_split=True, overwrite=False)
loader.load_dataset()
eval_dataset = loader.eval_dataset
model_input = eval_dataset["xc_nn_norm"]

with open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r") as f:
    selected_basins = np.array(
        json.load(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"))
    )

def get_timevar_parameters(config, test_epoch, select_idx, basin_id):
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

    # load model input
    select_basin_idx = np.where(selected_basins == 1031500)[0][0]
    tmp_model_input = model_input[0:730, select_basin_idx : select_basin_idx + 1, :]
    model_out = nn_model.predict_timevar_parameters(tmp_model_input)
    timevar_params = model_out.detach().cpu().numpy()
    return timevar_params
