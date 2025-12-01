import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg.core.data import load_json
# from dmg.core.post.plot_statbox import plot_boxplots # 移除绘图模块
from project.better_estimate import load_config

# 加载配置
lstm_config = load_config(r"conf/config_dhbv_pub_lstm.yaml")
hopev1_config = load_config(r"conf/config_dhbv_pub_hopev1.yaml")

lstm_criteria_path = os.path.join(lstm_config["out_path"], "metrics.json")
hopev1_criteria_path = os.path.join(hopev1_config["out_path"], "metrics.json")

lstm_df_list = []
hopev1_df_list = []

# 定义 Basin IDs
basin_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]

print("正在加载数据...")
for i in basin_ids:
    # 替换路径中的 basin 编号
    new_lstm_criteria_path = re.sub(
        r"pub_basin_\d+", f"pub_basin_{i}", lstm_criteria_path
    )
    new_hopev1_criteria_path = re.sub(
        r"pub_basin_\d+", f"pub_basin_{i}", hopev1_criteria_path
    )
    
    # 加载 JSON 并转换为 DataFrame
    lstm_criteria_df = pd.DataFrame(load_json(new_lstm_criteria_path))
    hopev1_criteria_df = pd.DataFrame(load_json(new_hopev1_criteria_path))
    
    lstm_df_list.append(lstm_criteria_df)
    hopev1_df_list.append(hopev1_criteria_df)

# ==========================================
# 修改部分：统计分析
# ==========================================

# 1. 获取前 10 个 Basin 的数据 (对应列表中的[:10])
target_lstm_list = lstm_df_list[:10]
target_hope_list = hopev1_df_list[:10]

# 2. 合并 DataFrame
# ignore_index=True 确保索引重置，合并为一个长表
combined_lstm_df = pd.concat(target_lstm_list, ignore_index=True)
combined_hope_df = pd.concat(target_hope_list, ignore_index=True)

# 3. 指定需要统计的指标
target_metric = "kge"  # 你可以在这里修改为 'nse', 'rmse' 等其他指标

# 确保指标存在于数据中
if target_metric in combined_lstm_df.columns and target_metric in combined_hope_df.columns:
    # 4. 计算中位数
    lstm_median = combined_lstm_df[target_metric].median()
    hope_median = combined_hope_df[target_metric].median()

    print("-" * 30)
    print(f"统计指标: {target_metric}")
    print(f"数据范围: 前 10 个 Basin (PUB-1 to PUB-10)")
    print("-" * 30)
    print(f"LSTM 模型中位数 (Median): {lstm_median:.4f}")
    print(f"S4D (HOPEv1) 模型中位数 (Median): {hope_median:.4f}")
    
    # 计算差异
    diff = hope_median - lstm_median
    print(f"差异 (S4D - LSTM): {diff:.4f}")
    print("-" * 30)
else:
    print(f"错误: 指标 '{target_metric}' 在数据集中未找到。可用列: {list(combined_lstm_df.columns)}")