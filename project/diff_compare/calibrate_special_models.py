import os
import sys
import yaml
import gc
from dotenv import load_dotenv

load_dotenv()
proj_path = os.getenv("PROJ_PATH")
sys.path.append(proj_path)  # type: ignore
from dmg import ModelHandler  # noqa: E402
from dmg.core.utils import (  # noqa: E402
    import_data_loader,
    import_trainer,
    set_randomseed,
)
from project.diff_compare import load_config  # noqa: E402

# ------------------------------------------#
# Define model settings here.
CONFIG_PATH = r"conf/config_dspecial_calibrate.yml"
# ------------------------------------------#
# model training
config = load_config(CONFIG_PATH)
config["mode"] = "train"
set_randomseed(config["random_seed"])
model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config["data_loader"])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
trainer_cls = import_trainer(config["trainer"])
trainer = trainer_cls(
    config, model, train_dataset=data_loader.train_dataset, verbose=True
)

trainer.train()
print(f"Training complete. Model saved to {config['model_path']}")

# model evaluation
config["mode"] = "test"
config["test"]["test_epoch"] = 50
set_randomseed(config["random_seed"])

model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config["data_loader"])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)

print("Evaluating model...")
config["test"]["start_time"] = "1989/01/01"
config["test"]["end_time"] = "1998/12/31"
trainer_cls = import_trainer(config["trainer"])
trainer = trainer_cls(
    config,
    model,
    eval_dataset=data_loader.train_dataset,
    verbose=True,
)
trainer.evaluate()

print(f"Metrics and predictions saved to {config['out_path']}")

config["test"]["start_time"] = "1999/01/01"
config["test"]["end_time"] = "2009/12/31"
tester = trainer_cls(
    config,
    model,
    eval_dataset=data_loader.eval_dataset,
    verbose=True,
)
tester.evaluate()
print(f"Metrics and predictions saved to {config['out_path']}")