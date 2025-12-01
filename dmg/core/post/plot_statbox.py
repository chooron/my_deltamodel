from typing import List, Optional, Tuple
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

def plot_boxplots(
    list1: List[pd.DataFrame],
    list2: List[pd.DataFrame],
    column_name: str,
    group_labels: List[str],
    ax: plt.Axes,
    list1_color: str = "#3C5488",
    list2_color: str = "#E64B35",
    ylabel: str = "Value",
    ylim: Optional[Tuple[float, float]] = None,
    legend_labels: List[str] = ["LSTM", "S4D"],
    notch: bool = True,
    show_grid: bool = True,
    # --- 新增功能参数 ---
    fontsize_label: int = 14,  # Y轴标签字体大小
    fontsize_tick: int = 12,  # 刻度标签字体大小
    fontsize_legend: int = 12,  # 图例字体大小
    subplot_label: Optional[str] = None,  # 子图标签 (如 "(a)", "A")
    subplot_label_fontsize: int = 14,  # 子图标签字体大小
    show_legend: bool = True,  # 是否显示图例
    legend_loc: str = "best",  # 图例位置锚点
    legend_bbox: Optional[
        Tuple[float, float]
    ] = None,  # 图例精确定位 (用于放到图外)
):
    # --- 1. 输入检查 ---
    if len(list1) != len(list2) or len(list1) != len(group_labels):
        raise ValueError("Input lists and labels must have the same length.")

    # --- 2. 准备数据 ---
    data_to_plot = []
    positions = []
    colors = [list1_color, list2_color] * len(list1)

    group_spacing = 2.5
    box_width = 0.6
    offset = 0.4

    for i, (df1, df2) in enumerate(zip(list1, list2)):
        data1 = df1[column_name].dropna() if column_name in df1.columns else []
        data2 = df2[column_name].dropna() if column_name in df2.columns else []
        data_to_plot.extend([data1, data2])
        center = i * group_spacing
        positions.extend([center - offset, center + offset])

    if not data_to_plot:
        return ax

    # --- 3. 网格 ---
    if show_grid:
        ax.yaxis.grid(
            True, linestyle="--", which="major", color="grey", alpha=0.3
        )
        ax.set_axisbelow(True)

    # --- 4. 样式 ---
    boxprops = dict(linewidth=1.2, edgecolor="black")
    medianprops = dict(linewidth=1.5, color="black", alpha=0.8)
    whiskerprops = dict(linewidth=1.2, linestyle="-")
    capprops = dict(linewidth=1.2)
    flierprops = dict(
        marker="o",
        markerfacecolor="none",
        markeredgecolor="gray",
        markersize=4,
        linestyle="none",
        alpha=0.6,
    )

    # --- 5. 绘图 ---
    bp = ax.boxplot(
        data_to_plot,
        notch=notch,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.85)

    # --- 6. 坐标轴与字体大小控制 ---
    tick_centers = [i * group_spacing for i in range(len(group_labels))]
    ax.set_xticks(tick_centers)
    ax.set_xticklabels(group_labels, fontsize=fontsize_tick)  # 设置X轴刻度字体

    # 设置Y轴刻度字体
    ax.tick_params(axis="y", labelsize=fontsize_tick)

    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(
        axis="both", which="major", direction="in", length=5, width=1
    )
    ax.tick_params(axis="y", which="minor", direction="in", length=3, width=0.8)

    # 垂直分隔线
    for i in range(len(group_labels) - 1):
        mid_point = (tick_centers[i] + tick_centers[i + 1]) / 2
        ax.axvline(
            mid_point, color="gray", linestyle=":", linewidth=0.8, alpha=0.5
        )

    # 轴标签字体
    ax.set_ylabel(ylabel, fontsize=fontsize_label)

    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(min(positions) - 1.0, max(positions) + 1.0)

    # --- 7. 子图标签 (右上角) ---
    if subplot_label:
        # (0.96, 0.96) 表示 axes 坐标系右上角，根据需要微调
        ax.text(
            0.98,
            0.1,
            subplot_label,
            transform=ax.transAxes,
            fontsize=subplot_label_fontsize,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # --- 8. 图例控制 (支持外部显示) ---
    if show_legend:
        legend_patches = [
            mpatches.Patch(
                facecolor=list1_color,
                label=legend_labels[0],
                edgecolor="black",
                alpha=0.85,
            ),
            mpatches.Patch(
                facecolor=list2_color,
                label=legend_labels[1],
                edgecolor="black",
                alpha=0.85,
            ),
        ]

        # ax.legend(
        #     handles=legend_patches,
        #     loc=legend_loc,           # 锚点位置
        #     bbox_to_anchor=legend_bbox, # 坐标偏移 (用于放到外部)
        #     fontsize=fontsize_legend, # 图例字体
        #     frameon=True,
        #     fancybox=False,
        #     edgecolor='black',
        #     facecolor='white',
        #     framealpha=1.0,
        #     ncol=len(legend_labels)   # 设为水平排列 (可选，看你喜好)
        # ).get_frame().set_linewidth(0.8)
        ax.legend(
            handles=legend_patches,
            loc=legend_loc,  # 锚点位置
            bbox_to_anchor=legend_bbox,  # 坐标偏移
            fontsize=fontsize_legend+2,  # 图例字体
            frameon=False,  # <--- 【关键修改】关闭边框
            ncol=len(legend_labels),  # 水平排列
        )

    # --- 9. 边框 ---
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    return ax


def plot_ensemble_spread_boxplot(
    data_list,
    axes,
    model_names=None,
    titles=None,
    colors=None,
    title_fontsize=14,
    label_fontsize=14,
    tick_fontsize=12,
    show_outliers=False,  # 是否显示异常值点，默认不显示以保持整洁
    box_width=0.6,
):
    """
    绘制系综离散度（标准差）的箱线图对比。
    
    参数:
    data_list (list of np.ndarray): 模型数据列表。
                                    每个数据的形状应为 (T, K, N)
                                    T: 时间步 (730)
                                    K: 变量类型数量 (3)
                                    N: 系综成员数量 (16)
    axes (list of matplotlib.axes.Axes): 用于绘图的轴列表，长度应等于 K。
    model_names (list of str): 模型名称列表，用于x轴标签。
    titles (list of str): 每个子图的标题（对应 K 个变量）。
    colors (list of str): 每个模型的颜色代码。
    """
    
    # --- 1. 基础配置与检查 ---
    if len(data_list) < 1:
        raise ValueError("data_list must contain at least one dataset.")
    
    # 检查维度
    T, K, N = data_list[0].shape
    
    if len(axes) != K:
        raise ValueError(f"Provided {len(axes)} axes, but data has {K} variable types.")
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(data_list))]
        
    if colors is None:
        # 默认备用颜色
        colors = ["#5B84B1FF", "#D94F4FFF"] 
    
    # 确保颜色数量足够
    if len(colors) < len(data_list):
        colors = colors * len(data_list)

    # --- 2. 循环绘制每一列 (每个变量类型) ---
    for k in range(K):
        ax = axes[k]
        
        # 准备绘图数据：计算每个时间步的系综标准差 (Spread)
        # spread_data_per_model 结构: [ (T,), (T,), ... ] 对应不同模型
        spread_data_per_model = []
        
        for model_idx, data in enumerate(data_list):
            # 取出第 k 个变量的所有数据: (T, N)
            variable_data = data[:, k, :]
            
            # 计算系综标准差 (Ensemble Std Dev)作为不确定性指标
            # 结果形状: (T,)
            spread = np.std(variable_data, axis=1)
            spread_data_per_model.append(spread)
            
        # --- 3. 绘制箱线图 ---
        # patch_artist=True 允许填充颜色
        bplot = ax.boxplot(
            spread_data_per_model,
            patch_artist=True,
            widths=box_width,
            showfliers=show_outliers,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2, color="gray"),
            capprops=dict(linewidth=1.2, color="gray"),
            sym='.' # 异常值样式
        )
        
        # --- 4. 填充颜色 ---
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8) # 稍微透明一点，更有质感
            patch.set_edgecolor(color) # 边框同色，或者设为 'black'
            
        # --- 5. 样式美化 (保持与 plot_parameters 一致) ---
        
        # 设置标题
        if titles and k < len(titles):
            ax.set_title(titles[k], fontsize=title_fontsize, pad=10)
            
        # 设置 X 轴标签 (模型名称)
        ax.set_xticklabels(model_names, fontsize=tick_fontsize, fontweight='medium')
        
        # 设置 Y 轴标签
        # 仅在第一列显示ylabel，或者根据需要调整
        if k == 0:
            ax.set_ylabel("Ensemble Spread (Std Dev)", fontsize=label_fontsize)
            
        # 设置刻度字体
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        
        # 网格线
        ax.grid(True, linestyle="--", alpha=0.5, axis='y') # 仅显示Y轴网格
        
        # 去除顶部和右侧的边框 (Spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 科学计数法 (如果数值很小)
        # 自动检测是否需要科学计数法
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
        
def plot_temporal_volatility_boxplot(
    data_list,
    axes,
    model_names=None,
    var_names=None,
    titles=None,
    colors=None,
    title_fontsize=14,
    label_fontsize=14,
    tick_fontsize=12,
    show_outliers=False,
    box_width=0.3,
):
    """
    绘制【时间波动性】(Temporal Volatility) 的箱线图对比。
    指标定义：|X_t - X_{t-1}| (相邻时间步的变化幅度绝对值)
    物理含义：反映曲线的"锯齿程度"或"粗糙度"。
    """
    
    # --- 1. 基础配置 ---
    if len(data_list) < 1:
        raise ValueError("data_list must contain at least one dataset.")
    
    T, K, N = data_list[0].shape
    
    if len(axes) != K:
        raise ValueError(f"Provided {len(axes)} axes, but data has {K} variable types.")
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(data_list))]
        
    if colors is None:
        colors = ["#5B84B1FF", "#D94F4FFF"] 
    
    if len(colors) < len(data_list):
        colors = colors * len(data_list)

    # --- 2. 循环绘制每一列 (每个变量类型) ---
    for k in range(K):
        ax = axes[k]
        
        # 准备绘图数据
        volatility_data_per_model = []
        
        for model_idx, data in enumerate(data_list):
            # 取出第 k 个变量的所有数据: (T, N)
            variable_data = data[:, k, :]
            
            # 【核心修改】计算时间波动性 (Temporal Volatility)
            # 1. 计算一阶差分: X_t - X_{t-1} -> 形状 (T-1, N)
            diffs = np.diff(variable_data, axis=0)
            
            # 2. 取绝对值: |X_t - X_{t-1}|
            abs_diffs = np.abs(diffs)
            
            # 3. 展平 (Flatten): 把所有时间步、所有系综成员的跳变幅度混在一起统计
            # 这样箱线图反映的是：这个模型产生的线条，平均每一步跳多大？
            volatility_data_per_model.append(abs_diffs.flatten())
            
        # --- 3. 绘制箱线图 ---
        bplot = ax.boxplot(
            volatility_data_per_model,
            patch_artist=True,
            widths=box_width,
            showfliers=show_outliers,
            medianprops=dict(color="black", linewidth=2.0),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5, color="gray"),
            capprops=dict(linewidth=1.5, color="gray"),
            sym='.' 
        )
        ax.spines['top'].set_visible(True)   # 改为 True
        ax.spines['right'].set_visible(True) # 改为 True
        
        # --- 4. 填充颜色 ---
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor(color)
            
        # --- 5. 样式美化 ---
        if titles and k < len(titles):
            # 可以在标题里加上 (Roughness) 字样提示读者
            ax.set_title(f"{titles[k]}", fontsize=title_fontsize, pad=10)
        if k == K:    
            ax.set_xticklabels(model_names, fontsize=tick_fontsize, fontweight='medium')
        
        # Y轴标签改为 Temporal Volatility
        if var_names:
            ax.set_ylabel(r"$|\Delta$ " + f"{var_names[k]}" + r"$|$", fontsize=label_fontsize)
        else:
            ax.set_ylabel("Temporal Volatility", fontsize=label_fontsize)
            
        ax.tick_params(axis="y", labelsize=tick_fontsize, direction='out')
        ax.grid(True, linestyle="--", alpha=0.5, axis='y')
        
        # 科学计数法
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))