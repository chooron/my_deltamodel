#!/usr/bin/env python3
"""
并行率定 special 水文模型（首字母大写，使用 config_dspecial_calibrate.yaml）

用法:
    python calibrate_special_parallel.py --model Gr4j
"""

import argparse
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dmg import ModelHandler  # noqa: E402
from dmg.core.utils import (  # noqa: E402
    import_data_loader,
    import_trainer,
    initialize_config,
    set_randomseed,
)

CONF_FILE = project_root / "project/diff_compare/conf/config_dspecial_calibrate.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="并行率定 special 水文模型")
    parser.add_argument("--model", type=str, required=True, help="模型名称 (例如: Gr4j, Flexi)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    # 读取 yaml，修改 model[0]，不写回文件
    with open(CONF_FILE, "r") as f:
        config = yaml.safe_load(f)

    # 手动合并 Hydra defaults 中的 observations（yaml.safe_load 不处理 defaults）
    obs_name = next(
        (d["observations"] for d in config.get("defaults", []) if isinstance(d, dict) and "observations" in d),
        None,
    )
    if obs_name:
        obs_file = CONF_FILE.parent / "observations" / f"{obs_name}.yaml"
        with open(obs_file, "r") as f:
            config["observations"] = yaml.safe_load(f)

    config["delta_model"]["phy_model"]["model"][0] = model_name
    config["device"] = "cuda"
    config["gpu_id"] = 0

    # 直接调用 initialize_config，补全 dtype/device/日期等字段
    config = initialize_config(config)

    loss_name = config["loss_function"]["model"]
    base_output = Path(config["save_path"])
    config["save_path"] = str(base_output / f"{model_name}_{loss_name}")

    print(f"\n{'='*60}")
    print(f"开始率定模型: {model_name}")
    print(f"配置文件: {CONF_FILE.name}")
    print(f"损失函数: {loss_name}")
    print(f"输出路径: {config['save_path']}")
    print(f"{'='*60}\n")

    # 训练
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

    # # 评估
    # config["mode"] = "test"
    # config["test"]["test_epoch"] = 100
    # set_randomseed(config["random_seed"])
    # model = ModelHandler(config, verbose=True)
    # data_loader_cls = import_data_loader(config["data_loader"])
    # data_loader = data_loader_cls(config, test_split=True, overwrite=False)
    # trainer_cls = import_trainer(config["trainer"])
    # trainer = trainer_cls(
    #     config,
    #     model,
    #     train_dataset=data_loader.train_dataset,
    #     eval_dataset=data_loader.eval_dataset,
    #     verbose=True,
    # )
    # trainer.evaluate()

    # print(f"\n{'='*60}")
    # print(f"模型 {model_name} 率定完成!")
    # print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
