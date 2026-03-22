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
    "train1989-1998/no_multi/Calibrate_E200_R365_B100_n20_noLn_noWU_42"
)
OUTPUT_TEST_CSV_NAME  = f"{os.path.dirname(os.path.abspath(__file__))}/csv/dif_test_kge2.csv"
OUTPUT_TRAIN_CSV_NAME = f"{os.path.dirname(os.path.abspath(__file__))}/csv/dif_train_kge2.csv"

DATA_PATH         = os.getenv("DATA_PATH", ".")
BASIN_ID_FILENAME = "559sub_id.txt"

LOSS_NAME  = "KgeLoss"
METRIC_KEY = "kge"
TRAIN_DIR  = "train1989-1998"
TEST_DIR   = "test1999-2009"
N_BASINS   = 559


# ==========================================
# 2. 自动推断 n_members
# ==========================================
def _infer_n_members(filepath: str, key: str, n_basins: int) -> int:
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

    return arr.size // n_basins


# ==========================================
# 3. 核心读取函数
# ==========================================
def _load_json_file(filepath: str, key: str, n_basins: int, n_members: int) -> np.ndarray:
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

    if arr.size != n_basins * n_members:
        print(f"  [SizeMismatch] {filepath}: expected {n_basins}×{n_members}={n_basins*n_members}, got {arr.size}")
        return np.full((n_basins, n_members), np.nan)

    # ✅ 唯一修正：basin-major，直接 reshape，不需要转置
    return arr.reshape(n_basins, n_members)


def get_model_results(model_name: str):
    """训练集选最优成员索引，测试集用同一索引取值。n_members 自动推断。"""
    def _path(period_dir):
        return os.path.join(
            RESULT_PATH, model_name, LOSS_NAME,
            "stat", f"{period_dir}_Ep200", "metrics.json"
        )

    train_path = _path(TRAIN_DIR)
    test_path  = _path(TEST_DIR)

    n_members = _infer_n_members(train_path, METRIC_KEY, N_BASINS)
    if n_members is None:
        print(f"  [Skip] {model_name}: 无法推断 n_members。")
        return None, None

    print(f"  {model_name:20s} | n_members={n_members}")

    train_data = _load_json_file(train_path, METRIC_KEY, N_BASINS, n_members)
    test_data  = _load_json_file(test_path,  METRIC_KEY, N_BASINS, n_members)

    if np.isnan(train_data).all():
        print(f"  [Skip] {model_name}: all-NaN.")
        return None, None

    # np.nanargmax 会在任意一行全 NaN 时抛 ValueError，这里按行容错。
    valid_row_mask = ~np.isnan(train_data).all(axis=1)
    invalid_count = int((~valid_row_mask).sum())
    if invalid_count > 0:
        print(f"  [Warn] {model_name}: {invalid_count}/{N_BASINS} basins are all-NaN in train metrics.")

    row_idx = np.arange(N_BASINS)
    train_results = np.full(N_BASINS, np.nan, dtype=float)
    test_results = np.full(N_BASINS, np.nan, dtype=float)

    if valid_row_mask.any():
        valid_rows = row_idx[valid_row_mask]
        best_idx_valid = np.nanargmax(train_data[valid_row_mask], axis=1)
        train_results[valid_rows] = train_data[valid_rows, best_idx_valid]
        test_results[valid_rows] = test_data[valid_rows, best_idx_valid]
    else:
        print(f"  [Skip] {model_name}: no valid basin rows after filtering all-NaN rows.")
        return None, None

    return train_results, test_results


def load_basin_ids(n_check: int = N_BASINS):
    subset_path = os.path.join(DATA_PATH, BASIN_ID_FILENAME)
    print(f"Loading basin IDs from: {subset_path}")

    if not os.path.exists(subset_path):
        print(f"  [Warning] Not found. Using numeric index.")
        return None

    with open(subset_path, "r", encoding="utf-8") as f:
        ids = json.load(f)

    if len(ids) != n_check:
        print(f"  [Warning] Count mismatch: {len(ids)} vs {n_check}. Using numeric index.")
        return None

    return ids


# ==========================================
# 3. 主程序
# ==========================================
COLUMN_ORDER = [
    "collie1", "wetland", "collie2", "newzealand1", "ihacres",
    "alpine1", "gr4j", "us1", "susannah1", "susannah2",
    "collie3", "alpine2", "hillslope", "topmodel", "plateau",
    "newzealand2", "penman", "simhyd", "australia", "gsfb",
    "flexb", "vic", "mopex1", "tcm", "flexi",
    "tank", "xinanjiang", "hymod", "mopex2", "mopex3",
    "mopex4", "flexis", "mopex5", "modhydrolog", "hbv96", "smar",
]


def main():
    print(f"Scanning: {RESULT_PATH}\n")

    model_list = sorted(
        e for e in os.listdir(RESULT_PATH)
        if os.path.isdir(os.path.join(RESULT_PATH, e))
    )
    print(f"Found {len(model_list)} model folders.\n")

    train_dict, test_dict = {}, {}
    failed_models = []
    for model in model_list:
        try:
            tr, te = get_model_results(model)
            if tr is not None:
                train_dict[model.lower()] = tr
                test_dict[model.lower()]  = te
        except Exception as e:
            failed_models.append(model)
            print(f"  [Error] {model}: {e}")

    print(f"\nValid models extracted: {len(test_dict)}")
    if failed_models:
        print(f"  [Summary] Failed models ({len(failed_models)}): {failed_models}")
    if not test_dict:
        print("Error: No valid data.")
        return

    # 按指定顺序排列列，缺失的列跳过，多余的列追加到末尾
    ordered_cols = [c for c in COLUMN_ORDER if c in test_dict]
    extra_cols   = [c for c in test_dict if c not in COLUMN_ORDER]
    if extra_cols:
        print(f"  [Info] Extra models not in COLUMN_ORDER: {extra_cols}")
    final_cols = ordered_cols + extra_cols

    df_train = pd.DataFrame(train_dict)[final_cols]
    df_test  = pd.DataFrame(test_dict)[final_cols]

    basin_ids = load_basin_ids()
    if basin_ids is not None:
        df_train.index = basin_ids
        df_test.index  = basin_ids
        index_label = "basin_id"
        print(">> Basin IDs applied.")
    else:
        index_label = "basin_index"

    os.makedirs(os.path.dirname(OUTPUT_TRAIN_CSV_NAME), exist_ok=True)
    df_train.to_csv(OUTPUT_TRAIN_CSV_NAME, index_label=index_label)
    df_test.to_csv(OUTPUT_TEST_CSV_NAME,   index_label=index_label)

    print("-" * 50)
    print(f"Train → {os.path.abspath(OUTPUT_TRAIN_CSV_NAME)}")
    print(f"Test  → {os.path.abspath(OUTPUT_TEST_CSV_NAME)}")
    print(f"Shape : {df_train.shape}  (rows=basins, cols=models)")
    print("-" * 50)
    print(df_train.head())
    print(df_test.head())


if __name__ == "__main__":
    main()