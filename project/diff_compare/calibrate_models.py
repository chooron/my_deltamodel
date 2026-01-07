import os
import sys
import yaml
import gc
import torch
# torch.autograd.set_detect_anomaly(True)
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
from dmg.models.hydromodel import STFN_INFO

AVAILABLE_MODELs = list(STFN_INFO.keys())
SPECIAL_MODELS = [
    "hbv96",
    "xinanjiang",
    "alpine1",
    "alpine2",
    "australia",
    "collie1",
    "collie2",
    "collie3",
    "gsfb",
    "hymod",
    "modhydrolog", # TODO 训练有问题，需要重跑
    "mopex1",
    "mopex2",
    "mopex3",
    "newzealand1",
    "tcm", # TODO 训练有问题，精度下不去
    
    "flexi",
    "flexb",
    "flexis",
    "gr4j",
    "hillslope",
    "ihacres", # TODO 模型率定过程无异常，但是在预测结果中有显著的偏差
    "mopex4",
    "mopex5",
    "newzealand2", # TODO 模型率定过程无异常，但是在预测结果中有显著的偏差
    "plateau",
    "smar",
]

def main(model_name):
    # ------------------------------------------#
    # Define model settings here.
    CONFIG_PATH = r"conf/config_dhbv_calibrate.yaml"
    with open(
        f"{proj_path}/project/diff_compare/conf/config_dhbv_calibrate.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)
    config["delta_model"]["phy_model"]["model_name"] = model_name

    with open(
        f"{proj_path}/project/diff_compare/conf/config_dhbv_calibrate.yaml", "w"
    ) as f:
        yaml.safe_dump(config, f, default_flow_style=False)
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

# 建议封装一个清理函数
def clear_memory():
    # 1. 强制进行垃圾回收，释放 Python 层面的对象引用
    gc.collect()
    
    # 2. 清理 PyTorch 的 CUDA 缓存分配器
    # 注意：这不会把显存还给操作系统，而是让 PyTorch 内部管理器知道这些显存可以重新分配
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # 某些多进程场景下可能需要

if __name__=='__main__':
    # for nm in AVAILABLE_MODELs:
    #     if nm not in SPECIAL_MODELS:
    #         print(f"calibrate {nm} model")
    #         main(nm)
    for model in ['wetland']:
        main(model)