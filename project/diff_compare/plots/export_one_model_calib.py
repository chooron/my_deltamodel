import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 配置路径
# ==========================================
RESULT_PATH = (
    "/workspace/my_deltamodel/project/diff_compare/output/camels_559/"
    "train1989-1998/no_multi/Calibrate_E100_R365_B100_n20_noLn_noWU_42"
)
OUTPUT_TEST_CSV_NAME  = f"{os.path.dirname(os.path.abspath(__file__))}/csv/dif_test_kge.csv"
OUTPUT_TRAIN_CSV_NAME = f"{os.path.dirname(os.path.abspath(__file__))}/csv/dif_train_kge.csv"

DATA_PATH         = os.getenv("DATA_PATH", ".")
BASIN_ID_FILENAME = "559sub_id.txt"

LOSS_NAME  = "KgeLoss"
METRIC_KEY = "kge"
TRAIN_DIR  = "train1989-1998"
TEST_DIR   = "test1999-2009"
N_BASINS   = 559

# ==========================================
# 指定模型列表（None 表示自动扫描全部）
# ==========================================
TARGET_MODELS = ["Gr4j"]  # 例如：["hbv96", "gr4j"] 或 None 表示全部


# ==========================================
# 2. 自动推断 n_members
# ==========================================
def _infer_n_members(filepath: str, key: str, n_basins: int) -> int:
    """
    从 JSON 文件中自动推断 n_members。
    n_members = arr.size / n_basins
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().lstrip("\ufeff").strip()

    data = json.loads(content)
    if isinstance(data, str):
        data = json.loads(data)

    arr = np.array(data.get(key, []), dtype=float)

    if arr.size % n_basins != 0:
        print(f"  [Warning] arr.size={arr.size} 不能被 n_basins={n_basins} 整除，无法推断 n_members。")
        return None

    n_members = arr.size // n_basins
    return n_members


# ==========================================
# 3. 核心读取函数
# ==========================================
def _load_json_file(
    filepath: str,
    key: str,
    n_basins: int,
    n_members: int,
) -> np.ndarray:
    """
    读取 metrics.json，返回 shape=(n_basins, n_members) 的矩阵。
    JSON 数组为 basin-major 顺序：arr[b * n_members + m] ↔ basin b, member m
    """
    if not os.path.exists(filepath):
        print(f"  [Missing] {filepath}")
        return np.full((n_basins, n_members), np.nan)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().lstrip("\ufeff").strip()

    data = json.loads(content)
    if isinstance(data, str):
        data = json.loads(data)

    arr = np.array(data.get(key, []), dtype=float)

    expected = n_basins * n_members
    if arr.size != expected:
        print(
            f"  [SizeMismatch] {filepath}: "
            f"expected {n_basins}×{n_members}={expected}, got {arr.size}"
        )
        return np.full((n_basins, n_members), np.nan)

    # basin-major：直接 reshape，不转置
    return arr.reshape(n_basins, n_members)


def get_model_results(model_name: str):
    """
    提取单个模型的最优 KGE。
    n_members 从训练集 JSON 自动推断，无需手动配置。
    """
    def _build_path(period_dir: str) -> str:
        return os.path.join(
            RESULT_PATH, model_name, LOSS_NAME,
            "stat", f"{period_dir}_Ep100", "metrics.json",
        )

    train_path = _build_path(TRAIN_DIR)
    test_path  = _build_path(TEST_DIR)

    # 自动推断 n_members
    n_members = _infer_n_members(train_path, METRIC_KEY, N_BASINS)
    if n_members is None:
        print(f"  [Skip] {model_name}: 无法推断 n_members。")
        return None, None

    print(f"  {model_name:20s} | n_members={n_members}")

    train_data = _load_json_file(train_path, METRIC_KEY, N_BASINS, n_members)
    test_data  = _load_json_file(test_path,  METRIC_KEY, N_BASINS, n_members)

    if np.isnan(train_data).all():
        print(f"  [Skip] {model_name}: all-NaN train data.")
        return None, None

    # 每个流域在训练集上选最优成员，测试集取同一索引
    best_idx = np.nanargmax(train_data, axis=1)   # shape=(n_basins,)
    row_idx  = np.arange(N_BASINS)

    train_results = train_data[row_idx, best_idx]  # shape=(n_basins,)
    test_results  = test_data[row_idx, best_idx]   # shape=(n_basins,)

    return train_results, test_results


def load_basin_ids(n_check: int = N_BASINS):
    """读取流域 ID 列表，失败时返回 None。"""
    subset_path = os.path.join(DATA_PATH, BASIN_ID_FILENAME)
    print(f"Loading basin IDs from: {subset_path}")

    if not os.path.exists(subset_path):
        print(f"  [Warning] Not found: {subset_path}. Using numeric index.")
        return None

    with open(subset_path, "r", encoding="utf-8") as f:
        ids = json.load(f)

    if len(ids) != n_check:
        print(f"  [Warning] Count mismatch: {len(ids)} vs {n_check}. Using numeric index.")
        return None

    return ids


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"Scanning: {RESULT_PATH}\n")

    # 获取模型列表：指定 or 全部扫描
    if TARGET_MODELS is not None:
        model_list = [m for m in TARGET_MODELS
                      if os.path.isdir(os.path.join(RESULT_PATH, m))]
        missing = [m for m in TARGET_MODELS if m not in model_list]
        if missing:
            print(f"[Warning] 以下模型目录不存在，已跳过: {missing}")
        print(f"指定模型: {model_list}\n")
    else:
        model_list = sorted(
            e for e in os.listdir(RESULT_PATH)
            if os.path.isdir(os.path.join(RESULT_PATH, e))
        )
        print(f"Found {len(model_list)} model folders.\n")

    train_dict, test_dict = {}, {}
    for model in model_list:
        tr, te = get_model_results(model)
        if tr is not None:
            train_dict[model.lower()] = tr
            test_dict[model.lower()]  = te
            print(f"    → test median KGE: {np.median(te):.4f}")

    print(f"\nValid models extracted: {len(test_dict)}")
    if not test_dict:
        print("Error: No valid data.")
        return

    df_train = pd.DataFrame(train_dict)  # shape=(n_basins, n_models)
    df_test  = pd.DataFrame(test_dict)

    basin_ids = load_basin_ids()
    if basin_ids is not None:
        df_train.index = basin_ids
        df_test.index  = basin_ids
        index_label = "basin_id"
        print("\n>> Basin IDs applied as index.")
    else:
        index_label = "basin_index"

    os.makedirs(os.path.dirname(OUTPUT_TRAIN_CSV_NAME), exist_ok=True)
    df_train.to_csv(OUTPUT_TRAIN_CSV_NAME, index_label=index_label)
    df_test.to_csv(OUTPUT_TEST_CSV_NAME,   index_label=index_label)

    print("-" * 60)
    print(f"Train CSV → {os.path.abspath(OUTPUT_TRAIN_CSV_NAME)}")
    print(f"Test  CSV → {os.path.abspath(OUTPUT_TEST_CSV_NAME)}")
    print(f"Shape     : {df_train.shape}  (rows=basins, cols=models)")
    print("-" * 60)

    # 汇总统计：按测试集中位数排序
    summary = pd.DataFrame({
        "train_median": df_train.median(),
        "test_median":  df_test.median(),
        "test_mean":    df_test.mean(),
        "test_std":     df_test.std(),
    }).sort_values("test_median", ascending=False)

    print("\n===== 各模型测试集 KGE 汇总（按中位数降序）=====")
    print(summary.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()