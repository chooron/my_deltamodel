import os
import sys

import torch
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
proj_path = os.getenv("PROJ_PATH", ".")
if proj_path:
    sys.path.append(proj_path)
from dmg import ModelHandler  # noqa
from dmg.core.data.loaders import HydroLoader  # noqa
from dmg.core.utils import import_trainer  # noqa
from project.better_estimate import load_config  # noqa
from project.better_estimate.interpret.temporal_ig_multistep import (
    MultiStepTemporalIntegratedGradients
)  # noqa

model_name = "lstm"
config = load_config(rf"conf/config_dhbv_{model_name}.yaml")
config["mode"] = "test"
config["test"]["test_epoch"] = 100
model = ModelHandler(config, verbose=True)
loader = HydroLoader(config, test_split=True, overwrite=False)
loader.load_dataset()
eval_dataset = loader.eval_dataset
trainer_cls = import_trainer(config["trainer"])
trainer = trainer_cls(config, model, eval_dataset=eval_dataset, verbose=True)
nn_model = trainer.model.model_dict["Hbv_2"].nn_model

##############################################
# 选择一个 batch
##############################################
with open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r") as f:
    selected_basins = np.array(json.load(f))
select_basin = 6431500
select_idx = (4810-730, 4810)
select_basin_idx = np.where(selected_basins == select_basin)[0][0]

assert eval_dataset is not None, "eval_dataset 不能为空"
x = eval_dataset["xc_nn_norm"][select_idx[0]:select_idx[1], select_basin_idx:select_basin_idx+1, :]
x = torch.tensor(x).float().permute(1, 0, 2)
batch_size, T, F = x.shape
print("模型输入形状:", x.shape)


print("\n正在运行模型以获取预测结果...")
with torch.no_grad():
    test_y = nn_model.predict_timevar_parametersv2(x)  # shape (1, 730, 48)
B, N, O = test_y.shape[0], test_y.shape[1], test_y.shape[2]

print(f"模型输出形状: {test_y.shape}")
print(f"输入时间步 T={T}, 输出时间步 T_output={N}, 输入特征维度 F={F}")
print(f"输出参数维度: P={O}")
print("\n开始计算 Temporal Integrated Gradients (Multi-Step)...")
print("这可能需要一些时间，请耐心等待...\n")

# 使用自定义的多步解释器
explainer = MultiStepTemporalIntegratedGradients(
    nn_model.predict_timevar_parametersv2
)
all_out_attr = []


for o in range(O):
    all_time_attr = []

    attributions_tuple = explainer.attribute(
        x,
        baselines=0,
        n_steps=50,
        method="gausslegendre",
        temporal_target=False,
        target=(-1, o),
        return_temporal_attributions=True,  # 返回所有时间步的归因结果
        show_progress=True,
    )

    full_attr = attributions_tuple[0]  # 取出 tensor
    print(f"Feature {o} attribution shape: {full_attr.shape}")

    # full_attr 应该是 [Batch, 730, 730, Features]
    all_out_attr.append(full_attr)


total_out_attr = (
    torch.stack(all_out_attr, dim=-1)
    .squeeze()
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)

save_path = Path(__file__).parent / "results" / model_name / str(select_basin)
if not save_path.exists():
    save_path.mkdir(parents=True)
np.savez_compressed(save_path / "tig_contribs.npz", data=total_out_attr)
