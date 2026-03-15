"""用 anomaly detection 定位 NaN 梯度的精确来源"""
import os, sys
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from dmg.core.data.data import create_training_grid
from project.blend_formula import load_config

CONFIG_PATH = r'conf/config_dblend_v1.yaml'
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
set_randomseed(config['random_seed'])

model = ModelHandler(config, verbose=False)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(config, model, train_dataset=data_loader.train_dataset, verbose=False)

n_samples, n_minibatch, n_timesteps = create_training_grid(
    data_loader.train_dataset["xc_nn_norm"], config
)
sample = trainer.sampler.get_training_sample(
    trainer.train_dataset, n_samples, n_timesteps
)

# 开启 anomaly detection
torch.autograd.set_detect_anomaly(True)

print("Forward pass...")
output = model(sample)

print("Calc loss...")
loss = model.calc_loss(sample)
print(f"Loss: {loss.item():.4f}")

print("Backward pass (with anomaly detection)...")
try:
    loss.backward()
    print("Backward completed without error")
except RuntimeError as e:
    print(f"RuntimeError during backward: {e}")
