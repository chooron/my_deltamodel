import json
import numpy as np

model_name = "Flexb"
loss = "KgeLoss"
train_dir = "test1989-1998"
test_dir = "test1999-2009"
base_dir = (
    "/workspace/my_deltamodel/project/diff_compare/output/camels_559/"
    + "train1989-1998/no_multi/Calibrate_E50_R365_B100_n16_noLn_noWU_42"
)
train_path = (
    rf"{base_dir}/{model_name}/{loss}/stat/{train_dir}_Ep50/metrics.json"
)
test_path = rf"{base_dir}/{model_name}/{loss}/stat/{test_dir}_Ep50/metrics.json"


def load_metric_array(
    json_path: str, key: str = "kge", n_basins: int = 559, n_members: int = 16
) -> np.ndarray:
    """Load JSON metrics and reshape to (n_basins, n_members)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raw_clean = raw.lstrip("\ufeff").strip()
        data = json.loads(raw_clean)

    if isinstance(data, str):
        data = json.loads(data)

    arr = np.array(data[key], dtype=float)
    expected = n_basins * n_members
    if arr.size != expected:
        raise ValueError(
            f"数据长度为 {arr.size}，无法 reshape 为 {n_basins}×{n_members}"
        )
    return arr.reshape(n_members, n_basins).T


def main():
    train_arr = load_metric_array(train_path)
    test_arr = load_metric_array(test_path)

    # 每个流域在训练集上最优的成员索引
    best_idx = train_arr.argmax(axis=1)
    best_train_values = train_arr[np.arange(train_arr.shape[0]), best_idx]

    # 在测试集上取出相同索引位置的值
    selected_test_values = test_arr[np.arange(test_arr.shape[0]), best_idx]

    print(f"Train best mean = {best_train_values.mean():.6f}")
    print(f"Train best median = {np.median(best_train_values):.6f}")
    print(f"Test selected mean = {selected_test_values.mean():.6f}")
    print(f"Test selected median = {np.median(selected_test_values):.6f}")


if __name__ == "__main__":
    main()
