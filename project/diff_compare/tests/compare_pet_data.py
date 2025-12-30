import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- 1. 加载数据 ---
path_old = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_dataset"
path_new = (
    r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_forcing_v2.pkl"
)

print("正在加载数据...")
with open(path_old, "rb") as f:
    # forcing shape: (671, 12418, 3) -> [P, T, PET]
    forcing_old, target_old, attr_old = pickle.load(f)

with open(path_new, "rb") as f:
    data_new = pickle.load(f)
    # 假设你的 v2 是之前代码生成的字典，如果是数组直接赋值即可
    if isinstance(data_new, dict):
        forcing_new = data_new["forcing"]  # (671, T, 3)
    else:
        forcing_new = data_new

# --- 2. 提取 PET 列 ---
# 假设 PET 都在第 3 列 (索引 2)
pet_old = forcing_old[:, :, 2]
pet_new = forcing_new[:, :, 2]

print(f"旧数据 PET 形状: {pet_old.shape}")
print(f"新数据 PET 形状: {pet_new.shape}")

# --- 3. 时间对齐截断 ---
# 取两个数据时间步长的最小值，防止长度不一致报错
min_len = min(pet_old.shape[1], pet_new.shape[1])
pet_old = pet_old[:, :min_len]
pet_new = pet_new[:, :min_len]

print(f"对齐后形状: {pet_old.shape} (用于比较)")

# --- 4. 统计指标计算 ---
correlations = []
nses = []
biases = []  # 偏差

print("正在计算每个流域的统计指标...")
for i in range(pet_old.shape[0]):
    series_old = pet_old[i, :]
    series_new = pet_new[i, :]

    # 去除 NaN (如果有的话)
    mask = ~np.isnan(series_old) & ~np.isnan(series_new)

    if np.sum(mask) > 100:  # 只有有效数据足够多才计算
        obs = series_old[mask]
        sim = series_new[mask]

        # 1. 相关系数 (R)
        r, _ = pearsonr(obs, sim)
        correlations.append(r)

        # 2. NSE (以旧数据为基准)
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        nse = 1 - (numerator / (denominator + 1e-6))
        nses.append(nse)

        # 3. 相对偏差 (Bias)
        bias = np.mean(sim) - np.mean(obs)
        biases.append(bias)

# --- 5. 打印结果摘要 ---
correlations = np.array(correlations)
nses = np.array(nses)
biases = np.array(biases)

print("\n" + "=" * 30)
print("   PET 数据一致性分析报告")
print("=" * 30)
print(
    f"平均相关系数 (Correlation): {np.nanmean(correlations):.4f} (理想值 > 0.95)"
)
print(f"平均 NSE 效率系数:          {np.nanmean(nses):.4f} (理想值 > 0.90)")
print(f"平均绝对偏差 (Bias):        {np.nanmean(np.abs(biases)):.4f} mm/day")
print("-" * 30)

# --- 6. 绘图验证 (随机选一个流域) ---
idx = np.random.randint(0, 671)  # 随机选个流域
plt.figure(figsize=(12, 5))

# 子图 1: 时间序列对比 (只画前 365 天看细节)
plt.subplot(1, 2, 1)
plt.plot(pet_old[idx, :365], label="Old Dataset (Target)", alpha=0.7)
plt.plot(
    pet_new[idx, :365],
    label="New Dataset (Generated)",
    alpha=0.7,
    linestyle="--",
)
plt.title(f"Basin {idx} PET Time Series (First Year)")
plt.ylabel("PET (mm/day)")
plt.legend()

# 子图 2: 散点图
plt.subplot(1, 2, 2)
plt.scatter(pet_old[idx, :], pet_new[idx, :], alpha=0.1, s=1)
plt.plot([0, 10], [0, 10], "r--")  # 1:1 线
plt.xlabel("Old PET")
plt.ylabel("New PET")
plt.title(f"Correlation: {correlations[idx]:.4f}")

# --- 7. 额外检查 v2 版本的 NaN 问题 (特别是 1989-2010) ---
if isinstance(data_new, dict) and "dates" in data_new:
    print("\n" + "=" * 30)
    print("   v2 数据 NaN 检查 (1989-2010)")
    print("=" * 30)

    dates = np.array(data_new["dates"])
    # 假设 dates 是 datetime 对象或包含年份的字符串
    # 如果是 datetime 对象，直接取 .year；如果是字符串，根据格式转换
    try:
        years = np.array([d.year for d in dates])
    except AttributeError:
        # 如果是字符串，尝试解析年份 (假设前4位是年份)
        years = np.array([int(str(d)[:4]) for d in dates])

    mask_89_10 = (years >= 1989) & (years <= 2010)
    indices_89_10 = np.where(mask_89_10)[0]

    if len(indices_89_10) > 0:
        forcing_slice = forcing_new[:, indices_89_10, :]
        nan_count = np.isnan(forcing_slice).sum()
        total_elements = forcing_slice.size

        print(
            f"检查时段: {years[indices_89_10[0]]} 至 {years[indices_89_10[-1]]}"
        )
        print(f"时间步数: {len(indices_89_10)}")
        print(f"总数据量: {total_elements}")
        print(f"NaN 总数:  {nan_count}")

        if nan_count > 0:
            var_names = data_new.get("variable_names", ["P", "T", "PET"])
            for feat_idx in range(forcing_slice.shape[2]):
                feat_nan = np.isnan(forcing_slice[:, :, feat_idx]).sum()
                var_name = (
                    var_names[feat_idx]
                    if feat_idx < len(var_names)
                    else f"Feat_{feat_idx}"
                )
                print(f"  - {var_name}: {feat_nan} NaNs")

            # 检查哪些流域有 NaN
            basins_with_nan = np.where(
                np.isnan(forcing_slice).any(axis=(1, 2))
            )[0]
            print(f"含有 NaN 的流域数量: {len(basins_with_nan)}")
            if len(basins_with_nan) > 0:
                print(
                    f"前 5 个含 NaN 的流域索引: {basins_with_nan[:5].tolist()}"
                )
        else:
            print("恭喜！1989-2010 时段未发现 NaN。")
    else:
        print("未找到 1989-2010 时段的数据。")
else:
    print("\n无法进行 NaN 检查：v2 数据格式不包含 'dates' 信息。")

plt.tight_layout()
plt.show()
