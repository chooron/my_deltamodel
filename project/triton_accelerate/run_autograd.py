"""
使用 autograd 后端运行训练 - 用于对比测试
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv("PROJ_PATH"))
from dmg import ModelHandler
from dmg.core.utils import (
    import_data_loader,
    import_trainer,
    set_randomseed,
)
from project.triton_accelerate import load_config

# 使用 autograd 配置
CONFIG_PATH = r"conf/config_dhbv_ann_autograd.yaml"

config = load_config(CONFIG_PATH)
config["mode"] = "train"
set_randomseed(config["random_seed"])

print(f"Backend: {config['delta_model']['phy_model'].get('backend', 'triton')}")

model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config["data_loader"])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
trainer_cls = import_trainer(config["trainer"])
trainer = trainer_cls(
    config, model, train_dataset=data_loader.train_dataset, verbose=True
)

trainer.train()
print(f"Training complete. Model saved to {config['model_path']}")
