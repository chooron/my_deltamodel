import pandas as pd
import os
import glob

# ==========================================
# 1. 配置路径
# ==========================================
DATA_PATH = "/workspace/my_deltamodel/project/diff_compare/data/marrmot"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CAL_NAME = os.path.join(OUTPUT_DIR, "marrmot_train_hybridkge.csv")
OUTPUT_EVAL_NAME = os.path.join(OUTPUT_DIR, "marrmot_test_hybridkge.csv")

FILTER_KEY = "obj3"

# ==========================================
# 2. 核心处理逻辑
# ==========================================
def clean_model_name(full_name):
    """
    提取简化名称。
    例如: m_01_collie1_1p_1s -> collie1
    逻辑: 按下划线拆分，取第3个元素 (索引2)
    """
    parts = full_name.split('_')
    # 确保长度足够，防止越界
    if len(parts) >= 3:
        return parts[2]  # 返回 collie1, wetland 等
    return full_name  # 如果格式不对，返回原名

def main():
    print(f"Scanning directory: {DATA_PATH}")
    
    # 1. 查找文件
    all_files = glob.glob(os.path.join(DATA_PATH, f"*{FILTER_KEY}*.csv"))
    
    # 先按原始文件名排序，确保处理顺序一致
    # 这样 m_01 会排在 m_02 前面
    all_files.sort()
    
    file_list = []
    for f in all_files:
        basename = os.path.basename(f)
        # 提取完整模型名用于去后缀: m_01_collie1_1p_1s_obj1_params.csv -> m_01_collie1_1p_1s
        full_model_name = basename.split('_obj1')[0]
        
        # === 修改点 1: 提取简短列名 ===
        short_name = clean_model_name(full_model_name)
        
        file_list.append((short_name, f))
    
    print(f"Found {len(file_list)} files. Starting extraction...")

    cal_dict = {}
    eval_dict = {}
    
    for short_name, filepath in file_list:
        try:
            # 2. 读取 CSV
            df_temp = pd.read_csv(filepath)
            
            # 清理列名
            df_temp.columns = [c.strip().lower() for c in df_temp.columns]
            
            # 3. 确定 gauge_id 列
            if 'gauge_id' in df_temp.columns:
                id_col = 'gauge_id'
            else:
                id_col = df_temp.columns[0]
            
            # === 修改点 2: 强制统一 ID 格式 ===
            # 逻辑：先转数字(处理字符串)，再转int(去掉小数点)，最后转str(作为索引)
            # 这会将 "1013500.0" 和 "1013500" 统一变为 "1013500"
            df_temp[id_col] = pd.to_numeric(df_temp[id_col], errors='coerce').fillna(0).astype(int).astype(str)
            
            # === 修改点 3: 去重 ===
            # 因为统一了 ID 格式，"1013500" 和 "1013500.0" 现在变成了相同的 ID，
            # 必须再次去重，否则 set_index 会报错
            df_temp.drop_duplicates(subset=[id_col], keep='first', inplace=True)
            
            # 设置索引
            df_temp.set_index(id_col, inplace=True)
            
            # 4. 提取数据
            cols = df_temp.columns
            col_cal = next((c for c in cols if 'cal' in c), cols[0])
            col_eval = next((c for c in cols if 'eval' in c), cols[1])
            
            # 存入字典，使用简化后的名称 (short_name)
            cal_dict[short_name] = df_temp[col_cal]
            eval_dict[short_name] = df_temp[col_eval]
            
        except Exception as e:
            print(f"[Error] Failed to process {short_name}: {e}")

    # 5. 合并与保存
    if not cal_dict:
        print("No valid data extracted.")
        return

    print("Merging dataframes...")
    
    df_cal_all = pd.DataFrame(cal_dict)
    df_eval_all = pd.DataFrame(eval_dict)
    
    # 保存
    df_cal_all.to_csv(OUTPUT_CAL_NAME, index_label="basin_id")
    df_eval_all.to_csv(OUTPUT_EVAL_NAME, index_label="basin_id")
    
    print("-" * 40)
    print(f"Saved Cal Matrix:  {OUTPUT_CAL_NAME}")
    print(f"Saved Eval Matrix: {OUTPUT_EVAL_NAME}")
    print("-" * 40)
    print("Preview (Cal) with unified IDs and Short Names:")
    print(df_cal_all.iloc[:5, :5]) # 显示前5行，前5列

if __name__ == "__main__":
    main()