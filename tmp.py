import pickle
import pandas as pd
import numpy as np

# 读取pickle文件
data_path = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_dataset"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 读取gage_id
gage_id_path = r"E:\PaperCode\dpl-project\generic_deltamodel\data\gage_id.npy"
gage_id = np.load(gage_id_path)

# 提取最后一个输出（第三个值）
attributes = data[2]

# 定义列名
column_names = [
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "frac_snow",
    "aridity",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "carbonate_rocks_frac",
    "geol_porosity",
    "geol_permeability",
]

# 构建DataFrame
df = pd.DataFrame(attributes, columns=column_names)

# 插入gauge_id作为第一列
df.insert(0, 'gauge_id', gage_id)

# 保存为txt文件
output_path = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_attributes.txt"
df.to_csv(output_path, sep=';', index=False)

print(f"数据形状: {attributes.shape}")
print(f"已保存到: {output_path}")
print(df.head())
