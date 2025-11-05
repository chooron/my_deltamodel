import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point
from dotenv import load_dotenv
import sys
import os
# --- 1. 准备示例数据 (您可以跳过这步，使用您自己的 gdf) ---
# 创建一个模拟的 GeoDataFrame
load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))  # type: ignore

from dmg.core.post.plot_geo import fetch_geo_data  # noqa
from project.better_estimate import load_config  # noqa
# --- 2. 确保 GDF 是正确的格式 ---
config = load_config(r"conf/config_dhbv_lstm.yaml")
gdf = fetch_geo_data(config)
# 确保坐标系为 EPSG:4326 (Plotly 需要经纬度)
# 如果您的数据不是这个坐标系，请取消下一行注释
# gdf = gdf.to_crs(epsg=4326)

# 从 'geometry' 列中提取经纬度到新列
# Plotly Express 需要明确的 'lat' 和 'lon' 列

print("\n处理后的 GDF (带 lat/lon):\n", gdf)

# --- 3. 创建 Plotly 可交互地图 ---

# 这是核心步骤
fig = px.scatter_mapbox(
    gdf,                  # 您的 GeoDataFrame
    lat="lat",            # 指定纬度列
    lon="lon",            # 指定经度列
    hover_name="gage_id", # 当鼠标悬停时，这将作为粗体标题
    hover_data={          # 定义悬停时显示哪些额外信息
        "gage_id": True,  # 明确显示 'gage_id'
        "lat": True,      # 显示纬度
        "lon": True       # 显示经度
    },
    zoom=3,               # (可选) 初始缩放级别
    height=600            # (可选) 图表高度
)

# 设置地图的底图样式
# 'open-street-map' 是一个免费且常用的选项
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0}) # 占满空间

# --- 4. 保存为可交互的 HTML 文件 ---

output_filename = "interactive_gage_map.html"
fig.write_html(output_filename)

print(f"\n成功! 地图已保存为: {output_filename}")

# (可选) 如果您在 Jupyter Notebook 中，可以直接显示图表
# fig.show()