import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point
from dotenv import load_dotenv
import sys
import os
import pandas as pd # 引入 pandas 以便处理数据列

# --- 1. 准备环境 (保持不变) ---
load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore

from dmg.core.post.plot_geo import fetch_geo_data  # noqa
from project.better_estimate import load_config  # noqa

# --- 2. 数据处理 ---
config = load_config(r"conf/config_dhbv_lstm.yaml")
gdf = fetch_geo_data(config)

# 确保坐标系为 EPSG:4326
if gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# 提取经纬度 (如果 fetch_geo_data 没有直接返回 lat/lon 列，需要这步)
# 既然你的 print 结果里有 lat/lon，这里假设它们已经存在。
# 如果不存在，请取消下面两行的注释：
# gdf["lon"] = gdf.geometry.centroid.x
# gdf["lat"] = gdf.geometry.centroid.y

# -------------------------------------------------------------------------
# [核心修改]：设置高亮逻辑
# -------------------------------------------------------------------------

# 1. 定义需要高亮的 ID 列表 (注意：将其转换为字符串以确保匹配，因为 gage_id 通常是字符串)
target_ids = ['1466500', '4105700', '6431500']
target_ids = ['10234500', '2069700', '13331500']

# 2. 确保 gdf 中的 gage_id 也是字符串格式 (防止 int 和 str 比较失败)
gdf['gage_id'] = gdf['gage_id'].astype(str)

# 3. 创建一个新的分类列 'status'
# 如果 ID 在目标列表中，标记为 'Target'，否则标记为 'Other'
gdf['status'] = gdf['gage_id'].apply(lambda x: 'Target' if x in target_ids else 'Other')

# 4. 创建一个新的大小列 'plot_size'
# 目标点大小设为 15，普通点设为 5 (你可以根据需要调整这些数值)
gdf['plot_size'] = gdf['status'].apply(lambda x: 15 if x == 'Target' else 5)

# 为了让图层顺序更好（让高亮的点浮在上面），我们先把普通点排前面，高亮点排后面
gdf = gdf.sort_values(by='status', ascending=True) 

print("\n处理后的 GDF (带高亮状态):\n", gdf[['gage_id', 'status', 'plot_size']].head())

# --- 3. 创建 Plotly 可交互地图 ---

fig = px.scatter_mapbox(
    gdf,
    lat="lat",
    lon="lon",
    hover_name="gage_id",
    
    # [修改]：将颜色映射到我们在上面创建的 'status' 列
    color="status",
    
    # [修改]：将点的大小映射到 'plot_size' 列
    size="plot_size",
    
    # [修改]：手动指定颜色。Target 为红色，Other 为蓝色(或默认色)
    color_discrete_map={
        "Target": "red",     # 高亮点的颜色
        "Other": "royalblue" # 普通点的颜色
    },
    
    # [修改]：设置最大点的大小，防止 Plotly 自动缩放得太小
    size_max=15, 
    
    hover_data={
        "gage_id": True,
        "lat": True,
        "lon": True,
        "status": False,     # 悬停时不显示 status 字段
        "plot_size": False   # 悬停时不显示 plot_size 字段
    },
    zoom=3,
    height=600
)

# 设置地图样式
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})

# --- 4. 保存 ---
output_filename = "interactive_gage_map_highlighted.html"
fig.write_html(output_filename)

print(f"\n成功! 地图已保存为: {output_filename}")

# fig.show()