import pickle
import numpy as np
import os
import json

# Paths (using relative logic similar to loader)
data_dir = r"e:\PaperCode\dpl-project\generic_deltamodel\data"
path_v2 = os.path.join(data_dir, "camels_forcing_v2.pkl")
path_gage_id = os.path.join(data_dir, "gage_id.npy")
path_559_ids = os.path.join(data_dir, "559sub_id.txt")

print("Loading gage_id.npy...")
all_gage_ids = np.load(path_gage_id)
# Ensure gage ids are strings and handled correctly (e.g., stripping decimals if any)
all_gage_ids = np.array([str(int(float(gid))).zfill(8) for gid in all_gage_ids])

print("Loading 559sub_id.txt using json.load...")
with open(path_559_ids, "r") as f:
    target_ids = json.load(f)
# Ensure target IDs are strings and 8-digit formatted
target_ids = [str(tid).zfill(8) for tid in target_ids]

print(f"Total basins in gage_id: {len(all_gage_ids)}")
print(f"Target basins in subset: {len(target_ids)}")

# Find indices of target basins
target_indices = []
missing_ids = []
for tid in target_ids:
    indices = np.where(all_gage_ids == tid)[0]
    if len(indices) > 0:
        target_indices.append(indices[0])
    else:
        missing_ids.append(tid)

print(f"Mapped {len(target_indices)} basins.")
if missing_ids:
    print(f"Warning: {len(missing_ids)} IDs not found in gage_id.npy")
    print(f"First 5 missing: {missing_ids[:5]}")

print("\nLoading camels_forcing_v2.pkl...")
with open(path_v2, "rb") as f:
    data_v2 = pickle.load(f)

# forcing shape: (671, T, 3)
forcing = data_v2["forcing"]
dates = np.array(data_v2["dates"])
variable_names = data_v2.get("variable_names", ["P", "T", "PET"])

# Year extraction from dates
try:
    years = np.array([d.year for d in dates])
except AttributeError:
    # If dating is string or other format
    years = np.array([int(str(d)[:4]) for d in dates])

# Filter for 1989-2010
mask_period = (years >= 1989) & (years <= 2010)
period_indices = np.where(mask_period)[0]

if len(period_indices) == 0:
    print("Error: 1989-2010 period not found in dates.")
else:
    start_year = years[period_indices[0]]
    end_year = years[period_indices[-1]]
    print(f"Checking period: {start_year} to {end_year}")
    print(f"Number of days in period: {len(period_indices)}")

    # Slice only the 559 target basins and the specific period
    # Note: target_indices maps subset basins to their position in forcing (671 basins)
    sub_forcing = forcing[target_indices, :, :][:, period_indices, :]

    nan_mask = np.isnan(sub_forcing)
    nan_count = nan_mask.sum()

    print("\n" + "=" * 40)
    print(f"   NaN 检查报告 (559 流域, {start_year}-{end_year})")
    print("=" * 40)
    print(f"总 NaN 数量: {nan_count}")

    if nan_count > 0:
        # Per variable breakdown
        for i, var in enumerate(variable_names):
            var_nans = nan_mask[:, :, i].sum()
            print(f"  - {var}: {var_nans} 个 NaN")

        # Per basin breakdown
        basins_with_nan_indices = np.where(nan_mask.any(axis=(1, 2)))[0]
        print(f"含有 NaN 的流域总数: {len(basins_with_nan_indices)}")

        print("\n详细 NaN 列表 (前 20 个异常流域):")
        for i in basins_with_nan_indices[:20]:
            gid = target_ids[i]  # Current ID in the target list
            original_idx = target_indices[
                i
            ]  # Index in the original 671 forcing

            v_details = []
            for v_idx, var in enumerate(variable_names):
                v_count = nan_mask[i, :, v_idx].sum()
                if v_count > 0:
                    v_details.append(f"{var}({v_count})")

            print(
                f"  流域 ID: {gid} (索引: {original_idx}) -> NaN 情况: {', '.join(v_details)}"
            )

        if len(basins_with_nan_indices) > 20:
            print(
                f"  ... 还有 {len(basins_with_nan_indices) - 20} 个流域未列出。"
            )
    else:
        print("未发现任何 NaN 值，数据在该时段内是完整的。")
