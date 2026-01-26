import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ==========================================
# 1. 配置路径 (Configuration)
# ==========================================
# 基础路径（含 nXX 之前的固定部分）
BASE_PATH = Path(
    r"/workspace/my_deltamodel/project/diff_compare/ablation/nmul/camels_559/train1989-1998/no_multi"
)

# 模型名称（可改）
MODEL_NAME = "hbv96"

# 读取的 loss 文件夹名
LOSS_NAME = "KgeInverseLoss"

METRIC_NAME = 'inv_kge'

# 读取的 metrics 文件名
METRICS_FILE = "metrics.json"

# nmul 列表（可改）
NMUL_LIST = [16, 32, 64, 128, 256]

# 读取的 epoch 列表（5, 10, 15, ..., 100）
EPOCH_LIST = list(range(5, 101, 5))
EPOCH_LIST.insert(0, 1)

# metrics.json 的成员数（可改）
N_MEMBERS = 128

# 流域数量（可改）
N_BASINS = 559

TRAIN_RANGE = "train1989-1998"
TEST_RANGE = "test1999-2009"
TIME_DICT = {
    "train": TRAIN_RANGE,
    "test": TEST_RANGE,
}

# ==========================================
# 2. 核心读取函数
# ==========================================
def _load_json_file(
    filepath: Path, key: str, n_basins: int, n_members: int
) -> np.ndarray:
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")

    content = filepath.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    data = json.loads(content)
    if isinstance(data, str):
        data = json.loads(data)

    arr = np.array(data.get(key, []), dtype=float)
    if arr.size != n_basins * n_members:
        raise ValueError(
            f"Unexpected size for '{key}' in {filepath}: {arr.size}"
        )

    return arr.reshape(n_members, n_basins).T


def _build_run_dir(nmul: int) -> Path:
    return BASE_PATH / f"Calibrate_E100_R365_B200_n{nmul}_noLn_noWU_42"


def _build_metrics_path(
    nmul: int, epoch: int, model_name: str, ds_type: str
) -> Path:
    run_dir = _build_run_dir(nmul)
    return (
        run_dir
        / model_name
        / LOSS_NAME
        / "stat"
        # / f"train1989-1998_Ep{epoch}"
        / f"{TIME_DICT[ds_type]}_Ep{epoch}"
        / METRICS_FILE
    )


def build_best_med_dataframe(
    *,
    nmul_list: Iterable[int],
    epochs: Iterable[int],
    model_name: str,
    ds_type: str,
) -> pd.DataFrame:
    """返回结构: index=epoch, columns=nmul, value=best_med."""
    data_map: dict[str, dict[int, float]] = {}
    for nmul in nmul_list:
        series_data: dict[int, float] = {}
        for ep in epochs:
            metric_path = _build_metrics_path(nmul, ep, model_name, ds_type)
            data = _load_json_file(metric_path, METRIC_NAME, N_BASINS, nmul)
            best_idx = np.nanargmax(data, axis=1)
            best_per_basin = data[np.arange(N_BASINS), best_idx]
            series_data[ep] = float(np.nanmedian(best_per_basin))
        data_map[str(nmul)] = series_data

    df = pd.DataFrame(data_map).sort_index()
    df.index.name = "epoch"
    return df


def main() -> None:
    output_dir = Path(
        "/workspace/my_deltamodel/project/diff_compare/plots/csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_type in ("train", "test"):
        df = build_best_med_dataframe(
            nmul_list=NMUL_LIST,
            epochs=EPOCH_LIST,
            model_name=MODEL_NAME,
            ds_type=ds_type,
        )
        output_path = output_dir / f"{MODEL_NAME}-{ds_type}-{METRIC_NAME}-loss.csv"
        df.to_csv(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
