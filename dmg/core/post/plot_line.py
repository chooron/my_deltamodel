import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
def plot_time_series(
        dates: pd.DatetimeIndex,
        data: Dict[str, np.ndarray],
        time_range: Optional[Tuple[str, str]] = None,
        target_name: str = 'target',
        colors: Optional[Dict[str, str]] = None, # 注意：这里的key需要和data的key一致
        title: str = '',
        ylabel: str = 'Streamflow (mm/day)',
        title_fontsize: int = 16,
        ylabel_fontsize: int = 14,
        tick_fontsize: int = 12,
        legend_fontsize: int = 12,
        axis_linewidth: float = 1.5,
        show_legend: bool = True,
        subplot_label: Optional[str] = None, 
        ax: Optional[plt.Axes] = None
):
    # ... (数据校验和截取逻辑同上，省略以节省空间) ...
    # --- 数据截取逻辑重复之前的即可 ---
    # ...
    y_len = len(next(iter(data.values())))
    if len(dates) != y_len:
        raise ValueError(f"Length mismatch.")
    
    if time_range:
        start_date, end_date = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        start_idx = dates.searchsorted(start_date, side='left')
        end_idx = dates.searchsorted(end_date, side='right')
        plot_dates = dates[start_idx:end_idx]
        plot_data = {name: arr[start_idx:end_idx] for name, arr in data.items()}
    else:
        start_date, end_date = dates[0], dates[-1]
        plot_dates = dates
        plot_data = data

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12), dpi=150) # 调整为竖长图
    else:
        fig = ax.figure

    # 白底
    ax.set_facecolor('white')
    if fig: fig.patch.set_facecolor('white')

    if colors is None: colors = {}

    # --- 绘图 ---
    for name, y_values in plot_data.items():
        if name == target_name:
            ax.plot(plot_dates, y_values, color='#333333', linestyle='-', 
                    label="Observation", linewidth=2.0, zorder=10, alpha=1.0)
        else:
            # 获取颜色，如果没有指定则默认红色
            c = colors.get(name, 'red')
            ax.plot(plot_dates, y_values, color=c, linestyle='--', 
                    label=name, linewidth=2.5, alpha=0.9, zorder=5)

    # --- 格式化 ---
    ax.set_xlim(start_date, end_date)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=10)
    ax.set_xlabel('')
    
    # 序号 (a) (b) (c)
    if subplot_label:
        ax.text(0.94, 0.96, subplot_label, transform=ax.transAxes, 
                fontsize=title_fontsize + 2, fontweight='bold', 
                va='top', ha='left', zorder=20)

    # --- 图例修改关键点 ---
    if show_legend:
        # loc='upper center': 位于上方居中
        # frameon=False: 去掉图例边框
        # ncol=3: 横向排列
        leg = ax.legend(
            fontsize=legend_fontsize, 
            loc='upper left', 
            ncol=1,
            frameon=True,           # 必须开启边框 (原来是 False)
            fancybox=True,          # 开启圆角 (对应 boxstyle="round")
            framealpha=0.7,         # 背景透明度 (对应 alpha=0.7)
            facecolor='white',      # 背景颜色 (对应 fc="white")
            edgecolor='black',      # 边框颜色 (对应 ec="black")
        )
        leg.get_frame().set_linewidth(1.0)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)
        spine.set_color('black')
        
    ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)

    return fig, ax