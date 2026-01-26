#! 主要是测试不同初始参数组下的模型消融实验
import os
import sys
import yaml
import gc
import torch
from dotenv import load_dotenv
from pathlib import Path

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

AVAILABLE_MODELs = ["collie1", "hymod", "hbv96",  "xinanjiang"] # 
NMUL_GROUPS = [16, 32, 64, 128, 256]


def _is_oom_error(err: Exception) -> bool:
    """Return True if the error looks like a CUDA OOM."""
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg and "cuda" in msg


def main(model_name, nmul):
    # ------------------------------------------#
    # Define model settings here.
    CONFIG_PATH = r"conf/config_ablation_nmul.yaml"
    with open(
        f"{proj_path}/project/diff_compare/conf/config_ablation_nmul.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)

    # 修改模型名称
    config["delta_model"]["phy_model"]["model_name"] = model_name

    # 仅修改 nmul，其他保持默认配置
    config["delta_model"]["phy_model"]["nmul"] = nmul

    with open(
        f"{proj_path}/project/diff_compare/conf/config_ablation_nmul.yaml", "w"
    ) as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    config = load_config(CONFIG_PATH)
    # ------------------------------------------#
    # model training
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
    
    # for e in list(range(5,105,5)):
    for e in [1]:
        print("Evaluating model...")
        config["mode"] = "test"
        config["test"]["test_epoch"] = e
        base_outpath = Path(config["out_path"]).parents[0]

        train_out = base_outpath / f"train1989-1998_Ep{e}"
        test_out = base_outpath / f"test1999-2009_Ep{e}"
        train_metrics = train_out / "metrics.json"
        test_metrics = test_out / "metrics.json"

        if train_metrics.exists() and test_metrics.exists():
            print(f"[Skip] metrics found for epoch {e}, skipping evaluation.")
            clear_memory()
            continue

        config["test"]["start_time"] = "1989/01/01"
        config["test"]["end_time"] = "1998/12/31"
        config["out_path"] = train_out
        config["out_path"].mkdir(parents=True, exist_ok=True)
        trainer_cls = import_trainer(config["trainer"])
        model = ModelHandler(config, verbose=True)
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
        config["out_path"] = test_out
        config["out_path"].mkdir(parents=True, exist_ok=True)
        tester = trainer_cls(
            config,
            model,
            eval_dataset=data_loader.eval_dataset,
            verbose=True,
        )
        tester.evaluate()
        print(f"Metrics and predictions saved to {config['out_path']}")

        clear_memory()


# 建议封装一个清理函数
def clear_memory():
    # 1. 强制进行垃圾回收，释放 Python 层面的对象引用
    gc.collect()
    
    # 2. 清理 PyTorch 的 CUDA 缓存分配器
    # 注意：这不会把显存还给操作系统，而是让 PyTorch 内部管理器知道这些显存可以重新分配
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # 某些多进程场景下可能需要

if __name__ == '__main__':
    import traceback  # 用于获取详细报错信息

    failed_experiments = []  # 用于存储异常记录

    print(f"{'='*20} nmul Ablation Start {'='*20}")

    for model_name in AVAILABLE_MODELs:
        for nmul in NMUL_GROUPS:
            experiment_name = f"{model_name}_nmul{nmul}"
            print(
                f"\n[Processing] Model: {model_name}, nmul: {nmul}"
            )

            try:
                # 运行训练和评估
                main(model_name, nmul)
                print(f"[Success] {experiment_name} finished.")
            except Exception as e:
                # 捕获任何异常
                error_message = str(e)
                error_trace = traceback.format_exc()
                error_type = "OOM" if _is_oom_error(e) else "ERROR"

                print(
                    f"!! [{error_type}] Failed to run {experiment_name}. Skipping to next."
                )
                print(f"!! Reason: {error_message}")

                # 记录错误
                failed_experiments.append({
                    "model": model_name,
                    "nmul": nmul,
                    "error": error_message,
                    "error_type": error_type,
                    "traceback": error_trace,
                })
            finally:
                # 清理显存后继续
                clear_memory()
                continue

    # ==========================================================
    # 生成总结报告
    # ==========================================================
    print("\n" + "=" * 50)
    print("        BATCH SIZE ABLATION STUDY REPORT")
    print("=" * 50)

    total_experiments = len(AVAILABLE_MODELs) * len(NMUL_GROUPS)
    successful_experiments = total_experiments - len(failed_experiments)

    print(f"Total Experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {len(failed_experiments)}")

    if len(failed_experiments) == 0:
        print("\n✅ All experiments completed successfully!")
    else:
        print(f"\n⚠️  Completed with {len(failed_experiments)} failures.\n")
        print("Failed Experiments List:")
        for i, item in enumerate(failed_experiments, 1):
            print(
                f"{i}. Model: {item['model']}, Num Start: {item['nmul']} ({item['error_type']})"
            )
            print(f"   Error: {item['error']}")
            print("-" * 30)

    print("=" * 50)