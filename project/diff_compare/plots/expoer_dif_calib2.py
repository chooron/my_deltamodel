import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 配置路径 (Configuration)
# ==========================================
# 模型结果路径
RESULT_PATH = (
    "/workspace/my_deltamodel/project/diff_compare/output/camels_559/"
    + "train1989-1998/no_multi/Calibrate_E50_R365_B100_n16_noLn_noWU_42"
)
# 输出 CSV 路径
OUTPUT_TEST_CSV_NAME = (
    f"{os.path.dirname(os.path.abspath(__file__))}/csv/dif_test_hybridkge.csv"
)
OUTPUT_TRAIN_CSV_NAME = (
    f"{os.path.dirname(os.path.abspath(__file__))}/csv/dif_train_hybridkge.csv"
)

# 流域 ID 文件路径配置
# 优先读取环境变量 DATA_PATH，如果没有则默认为当前目录 "."
DATA_PATH = os.getenv("DATA_PATH", ".")
BASIN_ID_FILENAME = "559sub_id.txt"

# 默认参数
LOSS_NAME = "KgeInverseLoss"  # 读取 KgeLoss 文件夹
METRIC_KEYS = ("inv_kge", "kge")  # 同时提取 inv_kge 和 kge
TRAIN_DIR = "test1989-1998"
TEST_DIR = "test1999-2009"
N_BASINS = 559  # 固定流域数量


# ==========================================
# 2. 核心读取函数
# ==========================================
def _load_json_file(
    filepath: str, key: str, n_basins: int, n_members: int
) -> np.ndarray:
    if not os.path.exists(filepath):
        return np.full((n_basins, n_members), np.nan)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().lstrip("\ufeff").strip()
        data = json.loads(content)
        if isinstance(data, str):
            data = json.loads(data)

        arr = np.array(data.get(key, []), dtype=float)
        if arr.size != n_basins * n_members:
            return np.full((n_basins, n_members), np.nan)

        return arr.reshape(n_members, n_basins).T
    except Exception:
        return np.full((n_basins, n_members), np.nan)


def _compute_hybrid_metric(metric_a: np.ndarray, metric_b: np.ndarray) -> np.ndarray:
    """Return simple average of two metric matrices."""
    if metric_a.shape != metric_b.shape:
        return np.full(metric_a.shape, np.nan)
    return (metric_a + metric_b) / 2.0


def get_model_results(model_name: str):
    """自动拼接路径并提取训练/测试集最优结果"""
    train_path = os.path.join(
        RESULT_PATH,
        model_name,
        LOSS_NAME,
        "stat",
        f"{TRAIN_DIR}_Ep50",
        "metrics.json",
    )
    test_path = os.path.join(
        RESULT_PATH,
        model_name,
        LOSS_NAME,
        "stat",
        f"{TEST_DIR}_Ep50",
        "metrics.json",
    )

    # 读取 (559, 16)
    train_inv = _load_json_file(train_path, METRIC_KEYS[0], N_BASINS, 16)
    train_kge = _load_json_file(train_path, METRIC_KEYS[1], N_BASINS, 16)
    test_inv = _load_json_file(test_path, METRIC_KEYS[0], N_BASINS, 16)
    test_kge = _load_json_file(test_path, METRIC_KEYS[1], N_BASINS, 16)

    train_data = _compute_hybrid_metric(train_inv, train_kge)
    test_data = _compute_hybrid_metric(test_inv, test_kge)

    if np.isnan(train_data).all():
        print(f"Skipping {model_name}: No valid data found.")
        return None, None

    # 训练集选优 -> 测试集取值
    best_idx = np.nanargmax(train_data, axis=1)
    train_results = train_data[np.arange(N_BASINS), best_idx]
    test_results = test_data[np.arange(N_BASINS), best_idx]

    return train_results, test_results


def load_basin_ids(n_check: int = 559):
    """读取流域 ID 列表"""
    subset_path = os.path.join(DATA_PATH, BASIN_ID_FILENAME)

    print(f"Loading basin IDs from: {subset_path}")

    if not os.path.exists(subset_path):
        print(
            f"[Warning] Basin ID file not found at {subset_path}. Will use numeric index."
        )
        return None

    try:
        with open(subset_path, "r", encoding="utf-8") as f:
            selected_basins = json.load(f)

        if len(selected_basins) != n_check:
            print(
                f"[Warning] ID count mismatch! Found {len(selected_basins)}, expected {n_check}. Using numeric index."
            )
            return None

        return selected_basins
    except Exception as e:
        print(f"[Error] Failed to load basin IDs: {e}")
        return None


# ==========================================
# 3. 主程序 (Main Execution)
# ==========================================
def main():
    print(f"Scanning directory: {RESULT_PATH}")

    # 1. 获取模型列表
    all_entries = os.listdir(RESULT_PATH)
    model_list = [
        entry
        for entry in all_entries
        if os.path.isdir(os.path.join(RESULT_PATH, entry))
    ]
    model_list.sort()

    print(f"Found {len(model_list)} folders. Starting extraction...")

    train_results_dict = {}
    test_results_dict = {}

    # 2. 遍历提取数据
    for model in model_list:
        train_values, test_values = get_model_results(model)
        if test_values is not None:
            train_results_dict[model.lower()] = train_values
            test_results_dict[model.lower()] = test_values

    print("\nExtraction finished.")

    if not test_results_dict:
        print("Error: No valid data extracted from any model.")
        return

    # 3. 转换为 DataFrame
    df_train = pd.DataFrame(train_results_dict)
    df_test = pd.DataFrame(test_results_dict)

    # 4. === 修改部分：尝试加载流域 ID 并设置为 Index ===
    basin_ids = load_basin_ids(n_check=N_BASINS)

    if basin_ids is not None:
        df_train.index = basin_ids
        df_test.index = basin_ids
        index_label_name = "basin_id"
        print(">> Basin IDs successfully loaded and applied as index.")
    else:
        index_label_name = "Basin_Index"
        print(">> Using default numeric index.")

    # 5. 保存为 CSV
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_TEST_CSV_NAME), exist_ok=True)

    df_train.to_csv(OUTPUT_TRAIN_CSV_NAME, index_label=index_label_name)
    df_test.to_csv(OUTPUT_TEST_CSV_NAME, index_label=index_label_name)

    print("-" * 40)
    print(f"Train CSV saved to: {os.path.abspath(OUTPUT_TRAIN_CSV_NAME)}")
    print(f"Test CSV saved to: {os.path.abspath(OUTPUT_TEST_CSV_NAME)}")
    print(
        f"Train/Test Matrix Shape: {df_train.shape}/{df_test.shape} (Rows: Basins, Cols: Models)"
    )
    print("-" * 40)
    print("Preview of the first 5 rows:")
    print("Train:")
    print(df_train.head())
    print("Test:")
    print(df_test.head())


if __name__ == "__main__":
    main()
