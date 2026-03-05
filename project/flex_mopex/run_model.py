import os
import sys

from dotenv import load_dotenv
import yaml

load_dotenv()

sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore
from dmg import ModelHandler  # noqa: E402
from dmg.core.utils import (  # noqa: E402
    import_data_loader,
    import_trainer,
    set_randomseed,
)
from project.flex_mopex import load_config  # noqa: E402

#------------------------------------------#
# Define model settings here. 3555MiB
CONFIG_PATH = r'conf/config_dmopex_v1.yaml'
#------------------------------------------#


def update_yaml_with_alpha(config_path: str, alpha: float) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)

    config_yaml['loss_function']['aic_alpha'] = alpha
    config_yaml['save_path'] = f'project/flex_mopex/output/flex_mopex_v1/alpha_{alpha:g}'
    config_yaml['trained_model'] = f"{config_yaml['save_path']}/model/"

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_yaml, f, allow_unicode=True, sort_keys=False)


if len(sys.argv) < 2:
    raise ValueError('请传入 alpha 参数，例如: python run_model.py 0.1')

alpha = float(sys.argv[1])
update_yaml_with_alpha(CONFIG_PATH, alpha)

# model training
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
set_randomseed(config['random_seed'])
model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    train_dataset=data_loader.train_dataset,
    verbose=True
)

# trainer.train()
print(f"Training complete. Model saved to \n{config['model_path']}")

# model evaluation
config['mode'] = 'test'
config['test']['test_epoch'] = 50
set_randomseed(config['random_seed'])

model = ModelHandler(config, verbose=True)
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)

trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    eval_dataset=data_loader.eval_dataset,
    verbose=True,
)

print('Evaluating model...')
trainer.evaluate()
print(f"Metrics and predictions saved to \n{config['out_path']}")