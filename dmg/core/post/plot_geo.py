import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ..data import txt_to_array

def fetch_geo_data(config):
    DATA_PATH = os.getenv("DATA_PATH")
    GAGE_ID_PATH = os.path.join(DATA_PATH, "gage_id.npy")
    GAGE_ID_531_PATH = os.path.join(DATA_PATH, "531sub_id.txt")
    SHAPEFILE_PATH = os.path.join(DATA_PATH, "camels_loc", "camels_671_loc.shp")
    # ------------------------------------------#

    # 1. Load gage ids + basin shapefile with geocoordinates (lat, long) for every gage.
    gage_ids = np.load(GAGE_ID_PATH, allow_pickle=True)
    gage_ids_531 = txt_to_array(GAGE_ID_531_PATH)
    coords = gpd.read_file(SHAPEFILE_PATH)

    # 2. Format geocoords for 531- and 671-basin CAMELS sets.
    coords_531 = coords[coords["gage_id"].isin(list(gage_ids_531))].copy()

    coords["gage_id"] = pd.Categorical(
        coords["gage_id"], categories=list(gage_ids), ordered=True
    )
    coords_531["gage_id"] = pd.Categorical(
        coords_531["gage_id"], categories=list(gage_ids_531), ordered=True
    )

    coords = coords.sort_values("gage_id")  # Sort to match order of metrics.

    # 4. Add the evaluation metrics to the basin shapefile.
    if config["observations"]["name"] == "camels_671":
        full_data = coords
    elif config["observations"]["name"] == "camels_531":
        full_data = coords_531
    else:
        raise ValueError(
            f"Observation data supported: 'camels_671' or 'camels_531'. Got: {config['observations']}"
        )
    return full_data

def geoplot_single_metric(
        gdf: gpd.GeoDataFrame,
        metric_name: str,
        title: str = None,
        cbar_title: str = None,
        map_color: bool = False,
        draw_rivers: bool = False,
        dynamic_colorbar: bool = False,
        dpi: int = 100,
        marker='o',
        marker_size: int = 50,
        ax=None,
        cax_pos=[0.8, 0.07, 0.16, 0.03],
        cmap="RdBu",
        fontsize=16,
        alpha=0.8,
        labelsize=14,
        vmin=0,
        vmax=1,
        highlight_basin_ids: list = None,  # 需要高亮的 basin_id 列表
        basin_id_col: str = 'gage_id'     # gdf 中对应的列名
):
    """Geographically map a single model performance metric using Cartopy."""
    
    # --- Input Validation ---
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        raise ValueError("GeoDataFrame must include 'lat' and 'lon' columns.")
    if metric_name not in gdf.columns:
        raise ValueError(f"GeoDataFrame does not contain the column '{metric_name}'.")

    # --- Setup Figure and Axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi, subplot_kw={'projection': ccrs.Mercator()})
    else:
        fig = ax.figure

    # --- Setup Map Extent ---
    min_lat, max_lat = gdf['lat'].min() - 2.5, gdf['lat'].max() + 2.5
    min_lon, max_lon = gdf['lon'].min() - 2.5, gdf['lon'].max() + 2.5
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # --- Add Map Features ---
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1, linestyle=':', zorder=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, zorder=2)
    if map_color:
        ax.add_feature(cfeature.LAND, zorder=0)
        ax.add_feature(cfeature.OCEAN, zorder=0)
    else:
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    if draw_rivers:
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.3, edgecolor='blue', zorder=1)
    
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # --- Plot the Metric Data ---
    
    # 1. 确定统一的 vmin 和 vmax
    if dynamic_colorbar:
        plot_vmin = gdf[metric_name].min()
        plot_vmax = gdf[metric_name].max()
    else:
        plot_vmin = vmin
        plot_vmax = vmax

    # 2. 判断是否需要高亮
    scatter_for_cbar = None # 用于传递给colorbar的对象

    if highlight_basin_ids and basin_id_col in gdf.columns:
        mask_highlight = gdf[basin_id_col].isin(highlight_basin_ids)
        gdf_highlight = gdf[mask_highlight]
        gdf_normal = gdf[~mask_highlight]

        # 2.1 画普通点 (底层)
        scatter_for_cbar = ax.scatter(
            gdf_normal['lon'], gdf_normal['lat'],
            c=gdf_normal[metric_name],
            s=marker_size,
            marker=marker,
            cmap=cmap,
            alpha=alpha,
            edgecolor='none',
            transform=ccrs.PlateCarree(),
            vmin=plot_vmin,
            vmax=plot_vmax,
            zorder=3
        )

        # 2.2 画高亮点 (顶层 + 黑色边框)
        if not gdf_highlight.empty:
            ax.scatter(
                gdf_highlight['lon'], gdf_highlight['lat'],
                c=gdf_highlight[metric_name],
                s=marker_size, 
                marker=marker,
                cmap=cmap,
                alpha=alpha, # 如果希望高亮点完全不透明，可设为 1.0
                edgecolor='black', # 黑色边框
                linewidth=1.2,     # 边框稍粗一点更明显
                transform=ccrs.PlateCarree(),
                vmin=plot_vmin,
                vmax=plot_vmax,
                zorder=4
            )

            # --- [核心新增] 添加标注 (ID Label) ---
            for idx, row in gdf_highlight.iterrows():
                # 获取ID文本
                label_text = str(row[basin_id_col])
                
                # 使用 annotate 添加文本
                # xy: 点的经纬度
                # xytext: 文本相对于点的偏移量 (x, y)，单位是 point (像素点)
                # textcoords: 指定偏移量的参考系
                txt = ax.annotate(
                    text=label_text,
                    xy=(row['lon'], row['lat']),
                    xytext=(5, 5),  # 向右上方偏移 5 个点，避免遮挡数据点
                    textcoords='offset points',
                    transform=ccrs.PlateCarree(), # 这一步至关重要，告诉matplotlib坐标是经纬度
                    fontsize=fontsize - 2, # 字体稍微比标题小一点
                    fontweight='bold',
                    color='black',
                    zorder=5 # 文字必须在最上层
                )
                
                # [可选] 给文字加白色描边，防止地图背景太黑看不清文字
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

    else:
        # 无高亮逻辑，全量绘制
        scatter_for_cbar = ax.scatter(
            gdf['lon'], gdf['lat'],
            c=gdf[metric_name],
            s=marker_size,
            marker=marker,
            cmap=cmap,
            alpha=alpha,
            edgecolor='none',
            transform=ccrs.PlateCarree(),
            vmin=plot_vmin,
            vmax=plot_vmax,
            zorder=3
        )

    # --- Colorbar setup ---
    cax = ax.inset_axes(cax_pos)
    # 无论上面走了哪个分支，scatter_for_cbar 都是包含了正确 cmap/norm 的对象
    cbar = fig.colorbar(scatter_for_cbar, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=labelsize)
    
    if cbar_title:
        cbar.ax.set_title(cbar_title, fontsize=fontsize, pad=8) 

    return fig, ax