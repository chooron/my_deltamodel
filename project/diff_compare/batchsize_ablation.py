#! 主要是测试不同batch下模型单次训练所需的时间
import os
import sys
import yaml
import gc
import torch
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

AVAILABLE_MODELs = "hbv96" # ["collie1", "hymod", "hbv96", "xinanjiang"] # , 
BATCH_SIZES = [200] # [10, 50, 100, 200]
PRED_LEN_SHORT = [365] # [365, 730, 365 * 4] # 
PRED_LEN_LONG = [365 * 8]


def _is_oom_error(err: Exception) -> bool:
    """Return True if the error looks like a CUDA OOM."""
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg and "cuda" in msg


def main(model_name, batch_size, pred_len):
    # ------------------------------------------#
    # Define model settings here.
    CONFIG_PATH = r"conf/config_ablation_batchsize.yaml"
    with open(
        f"{proj_path}/project/diff_compare/conf/config_ablation_batchsize.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)

    # 修改模型名称
    config["delta_model"]["phy_model"]["model_name"] = model_name

    # 修改batch size配置
    config["delta_model"]["nn_model"]["batch_size"] = batch_size
    config["train"]["batch_size"] = batch_size
    config["test"]["batch_size"] = batch_size

    # 修改rho
    config["delta_model"]["rho"] = pred_len

    with open(
        f"{proj_path}/project/diff_compare/conf/config_ablation_batchsize.yaml", "w"
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

    print(f"{'='*20} Batch Size & rho Ablation Start {'='*20}")

    # 短rho先跑，最长rho最后跑，减小OOM风险
    # phase_pred_lens = [PRED_LEN_SHORT, PRED_LEN_LONG]
    phase_pred_lens = [PRED_LEN_SHORT]

    for pred_lens in phase_pred_lens:
        for batch_size in BATCH_SIZES:
            for model_name in AVAILABLE_MODELs:
                for pred_len in pred_lens:
                    experiment_name = f"{model_name}_batch{batch_size}_rho{pred_len}"
                    print(
                        f"\n[Processing] Model: {model_name}, Batch Size: {batch_size}, rho: {pred_len}"
                    )

                    try:
                        # 运行训练和评估
                        main(model_name, batch_size, pred_len)
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
                            "batch_size": batch_size,
                            "pred_len": pred_len,
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

    total_pred = len(PRED_LEN_SHORT) + len(PRED_LEN_LONG)
    total_experiments = len(BATCH_SIZES) * len(AVAILABLE_MODELs) * total_pred
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
                f"{i}. Model: {item['model']}, Batch Size: {item['batch_size']}, rho: {item['pred_len']} ({item['error_type']})"
            )
            print(f"   Error: {item['error']}")
            print("-" * 30)

    print("=" * 50)