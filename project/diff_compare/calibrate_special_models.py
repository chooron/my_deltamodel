import os
import sys
import yaml
import gc
import torch
from pathlib import Path
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


CONFIG_PATH = r"conf/config_dspecial_calibrate.yml"

# 需要遍历的模型（首字母大写）
SPECIAL_MODELS = [
    "Flexi",
    "Flexb",
    "Flexis",
    "Gr4j",
    "Hillslope",
    "Ihacres",
    "Mopex4",
    "Mopex5",
    "Newzealand2",
    "Plateau",
    "Smar",
]


def run_one(model_name: str) -> None:
    """Train + evaluate a single model."""
    # 重新读取配置，避免跨模型污染
    with open(
        f"{proj_path}/project/diff_compare/conf/config_dspecial_calibrate.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)
    config["delta_model"]["phy_model"]["model"][0] = model_name

    with open(
        f"{proj_path}/project/diff_compare/conf/config_dspecial_calibrate.yaml", "w"
    ) as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    set_randomseed(config["random_seed"])
    config = load_config(CONFIG_PATH)

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

    # 评估
    config["mode"] = "test"
    config["test"]["test_epoch"] = 100
    set_randomseed(config["random_seed"])

    model = ModelHandler(config, verbose=True)
    data_loader_cls = import_data_loader(config["data_loader"])
    data_loader = data_loader_cls(config, test_split=True, overwrite=False)

    print("Evaluating model...")
    config["test"]["start_time"] = "1989/01/01"
    config["test"]["end_time"] = "1998/12/31"
    base_outpath = Path(config["out_path"]).parents[0]
    config["out_path"] = base_outpath / "train1989-1998_Ep100"
    config["out_path"].mkdir(parents=True, exist_ok=True)
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
    config["out_path"] = base_outpath / "test1999-2009_Ep100"
    config["out_path"].mkdir(parents=True, exist_ok=True)
    tester = trainer_cls(
        config,
        model,
        eval_dataset=data_loader.eval_dataset,
        verbose=True,
    )
    tester.evaluate()
    print(f"Metrics and predictions saved to {config['out_path']}")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    import traceback

    failed = []
    print(f"{'='*20} Special Models Calibration Start {'='*20}")

    for name in ["Gr4j"]:
        print(f"\n[Processing] Calibrating {name}...")
        try:
            run_one(name)
            print(f"[Success] {name} finished.")
        except Exception as e:
            failed.append((name, str(e), traceback.format_exc()))
            print(f"!! [ERROR] Failed to calibrate {name}. Reason: {e}")
        finally:
            clear_memory()

    print("\n" + "="*50)
    print("             FINAL EXECUTION REPORT")
    print("="*50)
    if not failed:
        print("✅ All special models calibrated successfully!")
    else:
        print(f"⚠️  Completed with {len(failed)} failures. List:")
        for idx, (name, err, _) in enumerate(failed, 1):
            print(f"  {idx}. {name} -> {err}")
    print("="*50)