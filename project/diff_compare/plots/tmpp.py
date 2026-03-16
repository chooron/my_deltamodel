import json
import numpy as np

# ==========================================
# 配置
# ==========================================
JSON_PATH = (
    "/workspace/my_deltamodel/project/diff_compare/output/camels_559/"
    "train1989-1998/no_multi/Calibrate_E100_R365_B100_n128_noLn_noWU_42/"
    "hbv96/KgeLoss/stat/train1989-1998_Ep100/metrics.json"
)
METRIC_KEY = "kge"
N_BASINS   = 559
N_MEMBERS  = 128

# ==========================================
# 读取与解析
# ==========================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    content = f.read().lstrip("\ufeff").strip()

data = json.loads(content)
if isinstance(data, str):          # 双重序列化保护
    data = json.loads(data)

arr = np.array(data[METRIC_KEY], dtype=float)
print(f"Raw array size : {arr.size}  (expected {N_BASINS * N_MEMBERS})")

# basin-major reshape → (559, 128)
mat = arr.reshape(N_BASINS, N_MEMBERS)
print(f"Matrix shape   : {mat.shape}  (basins × members)")

# ==========================================
# 每个流域取 128 组中的最优（最大）KGE
# ==========================================
best_kge = np.nanmax(mat, axis=1)   # (559,)
best_idx = np.nanargmax(mat, axis=1)

print(f"\n===== hbv96 | train1989-1998 | best-of-128 KGE =====")
print(f"  Median : {np.nanmedian(best_kge):.4f}")
print(f"  Mean   : {np.nanmean(best_kge):.4f}")
print(f"  Std    : {np.nanstd(best_kge):.4f}")
print(f"  Min    : {np.nanmin(best_kge):.4f}")
print(f"  Max    : {np.nanmax(best_kge):.4f}")
print(f"  NaN    : {np.isnan(best_kge).sum()} basins")