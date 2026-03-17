#!/usr/bin/env python3
"""
临时测试脚本：读取单个模型的 metrics.json，验证 reshape 逻辑
"""

import json
import numpy as np

# 配置
JSON_PATH = (
    "/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B100_n20_noLn_noWU_42/collie1/KgeLoss/stat/train1989-1998_Ep100/metrics.json"
)
METRIC_KEY = "kge"
N_BASINS = 559
N_MEMBERS = 128


def load_and_analyze():
    """读取 JSON，reshape 为 (559, 128)，提取每个流域的最优值"""

    # 1. 读取 JSON
    print(f"读取文件: {JSON_PATH}\n")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        content = f.read().lstrip("\ufeff").strip()

    data = json.loads(content)
    if isinstance(data, str):
        data = json.loads(data)

    arr = np.array(data.get(METRIC_KEY, []), dtype=float)
    print(f"原始数组长度: {arr.size} (期望: {N_BASINS * N_MEMBERS})")

    if arr.size != N_BASINS * N_MEMBERS:
        print(f"❌ 长度不匹配!")
        return

    # 2. Reshape: basin-major 顺序
    # arr[b * N_MEMBERS + m] ↔ basin b, member m
    kge_matrix = arr.reshape(N_BASINS, N_MEMBERS)
    print(f"Reshape 后形状: {kge_matrix.shape} (basins, members)\n")

    # 3. 每个流域选最优成员
    best_idx = np.nanargmax(kge_matrix, axis=1)  # (559,)
    best_kge = kge_matrix[np.arange(N_BASINS), best_idx]  # (559,)

    # 4. 统计
    median_kge = np.nanmedian(best_kge)
    mean_kge = np.nanmean(best_kge)

    print("=" * 60)
    print("每个流域的最优 KGE 统计:")
    print("=" * 60)
    print(f"中位数 (Median): {median_kge:.6f}")
    print(f"平均值 (Mean):   {mean_kge:.6f}")
    print(f"最小值 (Min):    {np.nanmin(best_kge):.6f}")
    print(f"最大值 (Max):    {np.nanmax(best_kge):.6f}")
    print(f"标准差 (Std):    {np.nanstd(best_kge):.6f}")
    print("=" * 60)

    # 5. 展示前 10 个流域的最优成员索引和 KGE
    print("\n前 10 个流域的最优成员:")
    print(f"{'Basin':<8} {'Best Member':<15} {'Best KGE':<12}")
    print("-" * 40)
    for i in range(min(10, N_BASINS)):
        print(f"{i:<8} {best_idx[i]:<15} {best_kge[i]:<12.6f}")

    # 6. 验证：打印第一个流域的所有 128 个成员的 KGE
    print(f"\n流域 0 的所有 {N_MEMBERS} 个成员 KGE (前 10 个):")
    print(kge_matrix[0, :10])
    print(f"最优成员索引: {best_idx[0]}, 最优 KGE: {best_kge[0]:.6f}")


if __name__ == "__main__":
    load_and_analyze()
