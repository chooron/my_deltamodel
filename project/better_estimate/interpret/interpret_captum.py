import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from captum.attr import IntegratedGradients

load_dotenv()
proj_path = os.getenv("PROJ_PATH", ".")
if proj_path:
    sys.path.append(proj_path)
from dmg import ModelHandler  # noqa
from dmg.core.data.loaders import HydroLoader  # noqa
from dmg.core.utils import import_trainer  # noqa
from project.better_estimate import load_config  # noqa

model_name = "lstm"
config = load_config(rf"conf/config_dhbv_{model_name}.yaml")
config["mode"] = "test"
config["test"]["test_epoch"] = 100
with open(os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt"), "r") as f:
    selected_basins = np.array(json.load(f))
select_basin = 6431500
select_idx = (4810-730, 4810)
select_basin_idx = np.where(selected_basins == select_basin)[0][0]
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
assert eval_dataset is not None, "eval_dataset 不能为空"
x = eval_dataset["xc_nn_norm"][:730, select_basin_idx:select_basin_idx+1, :]
x = torch.tensor(x).float().permute(1, 0, 2)
batch_size, T, F = x.shape
print("模型输入形状:", x.shape)

##############################################
# 模型输出 shape 检查
##############################################
print("\n正在运行模型以获取预测结果...")
with torch.no_grad():
    test_y = nn_model.predict_timevar_parametersv2(x)  # shape (1, 730, 48)
B, N, O = test_y.shape[0], test_y.shape[1], test_y.shape[2]

print(f"模型输出形状: {test_y.shape}")
print(f"输入时间步 T={T}, 输出时间步 T_output={N}, 输入特征维度 F={F}")
print(f"输出参数维度: P={O}")
print("\n开始计算 Temporal Integrated Gradients...")
print("这可能需要一些时间，请耐心等待...\n")

all_out_attr = []
for o in range(O):
    all_time_attr = []
    for t in tqdm(range(730)):
        # 每次只解释第 t 个时间步的输出（因果模型天然只能看到前 t 个输入）
        def forward_slice(x):
            return nn_model.predict_timevar_parametersv2(x)[
                :, t, o
            ]  # 如果输出是 [B,730,1] 就 .[:, t, 0]

        ig_t = IntegratedGradients(forward_slice)

        attr_t = ig_t.attribute(
            x, baselines=torch.zeros_like(x), n_steps=50, method="gausslegendre"
        )  # shape: [8, 730, features]

        # 手动强制因果掩码（未来输入贡献归零）
        mask = torch.arange(730, device=x.device) <= t
        attr_t = attr_t * mask[None, :, None]  # 只保留 <=t 的贡献
        all_time_attr.append(attr_t)
    full_attr = torch.stack(
        all_time_attr, dim=1
    )  # → [8, 730_out, 730_in, features]
    all_out_attr.append(full_attr)
total_out_attr = (
    torch.stack(all_out_attr, dim=-1).squeeze().detach().cpu().numpy()
)
save_path = Path(__file__).parent / "results" / model_name / str(select_basin)
if not save_path.exists():
    save_path.mkdir(parents=True)
np.savez_compressed(save_path / "ig_contribs.npz", data=total_out_attr)
