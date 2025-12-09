import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker


def set_nature_style():
    """
    Sets matplotlib params to mimic Nature Communications style.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],  # 优先使用 Arial
            "font.size": 8,  # 正文基础字号 (Points)
            "axes.labelsize": 9,  # 轴标签字号
            "axes.titlesize": 10,  # 标题字号
            "xtick.labelsize": 8,  # X轴刻度字号
            "ytick.labelsize": 8,  # Y轴刻度字号
            "legend.fontsize": 8,  # 图例字号
            "axes.linewidth": 1.0,  # 边框粗细
            "xtick.major.width": 0.8,  # 刻度线粗细
            "ytick.major.width": 0.8,
            "xtick.direction": "out",  # 刻度朝外
            "ytick.direction": "out",
            "lines.linewidth": 1.5,  # 线条粗细
            "figure.dpi": 300,  # 高分辨率用于打印
            "savefig.dpi": 300,
        }
    )


def plot_distribution_curve(
    ax,
    data,
    data_name="Value",
    bins=50,
    line1=-0.05,
    line2=0.05,
    show_counts=True,
):
    """
    Generates a Nature-style histogram and KDE curve.
    """

    # --- Color Palette (Nature Publishing Group style) ---
    color_hist = "#8491B4"  # 柔和的灰蓝色
    color_kde = "#3C5488"  # 深蓝色 (NPG Blue)
    color_line1 = "#E64B35"  # 朱红色 (NPG Red)
    color_line2 = "#00A087"  # 蓝绿色 (NPG Green)

    # --- 1. Histogram ---
    # 使用 density=True 确保和 KDE 同一尺度
    # edgecolor='white' 可以让柱状图更清晰，减少视觉粘连
    ax.hist(
        data,
        bins=bins,
        density=True,
        color=color_hist,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
        label="Observed Dist.",
        zorder=1,
    )

    # --- 2. KDE Curve ---
    kde = gaussian_kde(data)
    # 动态计算范围，稍微留出余量
    x_min, x_max = data.min(), data.max()
    margin = (x_max - x_min) * 0.1
    x_range = np.linspace(x_min - margin, x_max + margin, 1000)

    ax.plot(
        x_range,
        kde(x_range),
        color=color_kde,
        linewidth=2,
        label="KDE Fit",
        zorder=2,
    )

    # --- 3. Vertical Lines ---
    # 使用 ylim 确保线条贯穿整个高度，但不手动设置数值，保持自动缩放
    ax.axvline(
        x=line1,
        color=color_line1,
        linestyle="--",
        linewidth=1.2,
        label=f"$x = {line1}$",
        zorder=3,
    )
    ax.axvline(
        x=line2,
        color=color_line2,
        linestyle="--",
        linewidth=1.2,
        label=f"$x = {line2}$",
        zorder=3,
    )

    # --- 4. Annotations (Counts) ---
    # 优化：不再将文字散落在图上（容易遮挡），而是集中在图表左上角或右上角显示统计信息
    # 这种方式更符合学术图表的严谨性
    if show_counts:
        count_less = np.sum(data < line1)
        count_greater = np.sum(data > line2)
        count_between = len(data) - count_less - count_greater
        total = len(data)

        # 格式化文本块
        stats_text = (
            f"Total $N$ = {total}\n"
            f"x < {line1}: {count_less} ({count_less / total:.1%})\n"
            f"{line1} $\leq$ x $\leq$ {line2}: {count_between} ({count_between / total:.1%})\n"
            f"x > {line2}: {count_greater} ({count_greater / total:.1%})"
        )

        # 使用 transform=ax.transAxes 相对坐标定位，确保文字始终在图内
        # 0.05, 0.95 表示左上角
        ax.text(
            0.95,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",   # 关键：改为底部对齐，文字向上生长
            horizontalalignment="right",  # 关键：改为右对齐，文字向左生长
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.9,
                edgecolor="#dddddd",
            ),
        )

    # --- 5. Aesthetics & Cleanup ---
    ax.set_xlabel(data_name)
    ax.set_ylabel("Probability Density")

    # 去除顶部和右侧的边框 (Nature 风格经典做法)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    
    # 调整刻度显示
    ax.legend(
        loc="upper right",
        frameon=True,        # 开启边框
        facecolor="white",   # 背景颜色：白色
        framealpha=0.8,      # 透明度：0.8 (0=全透, 1=不透)，保证不完全遮挡数据
        edgecolor="#dddddd",   # 边框颜色：黑色
        fancybox=False       # 边框样式：False为直角，True为圆角 (Nature风格常用直角)
    )

    # 可选：如果非要网格，使用非常淡的灰色，且置于底层
    ax.grid(True, which="major", axis='y', linestyle=":", linewidth=0.5, color="#d3d3d3", alpha=0.5, zorder=0)


# --- 示例用法 ---
if __name__ == "__main__":
    # 1. 应用样式
    set_nature_style()

    # 2. 生成模拟数据
    np.random.seed(42)
    mock_data = np.concatenate(
        [
            np.random.normal(0, 0.1, 800),  # 尖峰
            np.random.normal(0.2, 0.3, 200),  # 长尾
        ]
    )

    # 3. 绘图
    # Nature 单栏宽度约为 89mm (3.5 inches)，双栏约为 183mm (7.2 inches)
    fig, ax = plt.subplots(figsize=(3.5, 3))

    plot_distribution_curve(
        ax,
        mock_data,
        data_name="Relative Error ($\delta$)",
        line1=-0.1,
        line2=0.1,
    )

    plt.tight_layout()
    plt.show()
