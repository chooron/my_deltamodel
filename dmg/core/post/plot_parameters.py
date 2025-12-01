import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from matplotlib.ticker import MaxNLocator
# plot_parameters(
#     timevar_params[:, :, :],
#     titles=titles,
#     median_color="red",
#     smooth_method="ewma",
#     alpha=0.1,
#     span=5,
# )
# plot_parameters(
#     timevar_params[:, :, :],
#     titles=titles,
#     median_color="red",
#     smooth_method="savgol",
#     window_length=51,
#     polyorder=2,
# )
# plot_parameters(
#     timevar_params[:, :, :],
#     titles=titles,
#     median_color="red",
#     smooth_method="lowpass",
#     cutoff_freq=0.1,
#     order=3,
# )


def smooth_sequences(data, method="moving_average", **kwargs):
    """
    对序列数据应用平滑处理，支持多种平滑方法。

    参数:
    data (np.array): 输入数据，期望 shape 为 (T, K, N)
                     T: 序列长度
                     K: 参数/类型数量
                     N: 成分数量
    method (str): 平滑方法，可选值:
                  - 'moving_average': 简单移动平均 (默认)
                  - 'ewma': 指数加权移动平均
                  - 'savgol': Savitzky-Golay 滤波
                  - 'lowpass': Butterworth 低通滤波
    **kwargs: 方法特定的参数
              - 对于 'moving_average': window (int, 默认 5)
              - 对于 'ewma': alpha (float, 0-1, 默认 0.1) 或 span (int, 默认 5)
              - 对于 'savgol': window_length (int, 默认 51, 必须奇数), polyorder (int, 默认 2)
              - 对于 'lowpass': cutoff_freq (float, 0-1, 默认 0.1, 归一化频率), order (int, 默认 4)

    返回:
    smoothed_data (np.array): 平滑后的数据，shape 与输入相同
    """
    if data.ndim != 3:
        raise ValueError(
            f"Input data has {data.ndim} dimensions, but expected 3 (T, K, N)."
        )

    T, K, N = data.shape
    smoothed_data = np.zeros_like(data)

    for k in range(K):
        for n in range(N):
            seq = data[:, k, n]

            if method == "moving_average":
                window = kwargs.get("window", 5)
                if window <= 1:
                    smoothed_seq = seq
                else:
                    df = pd.DataFrame(seq)
                    smoothed_seq = (
                        df.rolling(window=window, center=True, min_periods=1)
                        .mean()
                        .values.flatten()
                    )

            elif method == "ewma":
                alpha = kwargs.get("alpha", None)
                span = kwargs.get("span", 5)
                df = pd.DataFrame(seq)
                if alpha is not None:
                    smoothed_seq = (
                        df.ewm(alpha=alpha, adjust=True).mean().values.flatten()
                    )
                else:
                    smoothed_seq = (
                        df.ewm(span=span, adjust=True).mean().values.flatten()
                    )

            elif method == "savgol":
                window_length = kwargs.get("window_length", 51)
                polyorder = kwargs.get("polyorder", 2)
                if window_length % 2 == 0:
                    window_length += 1  # 确保奇数
                smoothed_seq = savgol_filter(
                    seq, window_length=window_length, polyorder=polyorder
                )

            elif method == "lowpass":
                cutoff_freq = kwargs.get(
                    "cutoff_freq", 0.1
                )  # 归一化频率 (Nyquist频率为0.5)
                order = kwargs.get("order", 4)
                b, a = butter(order, cutoff_freq, btype="low", analog=False)
                smoothed_seq = filtfilt(b, a, seq)

            else:
                raise ValueError(
                    f"Unsupported method: {method}. Choose from 'moving_average', 'ewma', 'savgol', 'lowpass'."
                )

            smoothed_data[:, k, n] = smoothed_seq

    return smoothed_data


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes


def plot_parameters(
    data,
    titles=None,
    model_name=None,
    gray_color="0.7",
    median_color="blue",
    smooth_method=None,
    axes=None,
    fig=None,
    ts=None,
    title_fontsize=14,
    label_fontsize=16,
    tick_fontsize=14,
    legend_fontsize=16,
    show_ylabel=True,
    # --- [新增参数] ---
    legend_pos=None,  # 可选: 'top-left', 'top-right', None
    **smooth_kwargs,
):
    """
    绘制参数序列图，突出显示中位数成分，并可选择应用平滑。

    新增参数:
    legend_pos (str): 控制第一个子图图例的位置。
                      'top-left': 图外左上方
                      'top-right': 图外右上方
                      None: 不显示图例 (默认)
    """

    # 检查数据维度
    if data.ndim != 3:
        raise ValueError(
            f"Input data has {data.ndim} dimensions, but expected 3 (T, K, N)."
        )

    T, K, N = data.shape

    # 如果没有提供标题，创建默认标题
    if titles is None:
        titles = [f"Parameter Type {k + 1}" for k in range(K)]
    elif len(titles) != K:
        print(
            f"Warning: Provided {len(titles)} titles, but expected {K}. Using default titles."
        )
        titles = [f"Parameter Type {k + 1}" for k in range(K)]

    # 应用平滑（如果指定）
    # (假设 smooth_sequences 函数在外部定义，此处省略具体实现)
    if smooth_method is not None:
        # data = smooth_sequences(data, method=smooth_method, **smooth_kwargs)
        pass

    # 处理axes参数
    if axes is None:
        if fig is None:
            fig, axes_temp = plt.subplots(
                K, 1, figsize=(14, 5 * K), sharex=True, sharey=True
            )
            axes = [axes_temp] if K == 1 else list(axes_temp)
        else:
            axes = [fig.add_subplot(K, 1, k + 1) for k in range(K)]
    elif isinstance(axes, matplotlib.axes.Axes):
        if K != 1:
            raise ValueError(
                f"Provided a single Axes, but data has K={K} parameter types."
            )
        axes = [axes]
    else:
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        if len(axes) < K:
            raise ValueError(
                f"Provided axes has length {len(axes)}, but expected at least {K}."
            )
        axes = list(axes)[:K]

    if ts is None:
        ts = np.arange(T)

    # 遍历 K 个参数类型
    import matplotlib.dates as mdates

    for k in range(K):
        ax = axes[k]

        param_data = data[:, k, :]
        temporal_averages = np.nanmean(param_data, axis=0)
        median_avg_value = np.median(temporal_averages)
        median_component_index = np.argmin(
            np.abs(temporal_averages - median_avg_value)
        )
        median_sequence = param_data[:, median_component_index]

        # 5. 绘制所有 N 个成分的灰色线条
        for i in range(N):
            ax.plot(
                ts, param_data[:, i], color=gray_color, alpha=0.4, linewidth=1.0
            )

        # 6. 绘制中位数成分的彩色线条
        ax.plot(
            ts,
            median_sequence,
            color=median_color,
            linewidth=2.0,
            label=model_name,
        )

        # x轴日期格式化
        if hasattr(ts[0], "year") and hasattr(ts[0], "month"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        # 7. 设置图表样式
        if show_ylabel:
            ax.set_ylabel(titles[k], fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        if k != K - 1:
            ax.tick_params(bottom=False, labelbottom=False)
        ax.grid(True, linestyle="--", alpha=0.5)

    if legend_pos is None:
        for ax in axes:
            ax.legend(
                fontsize=legend_fontsize,
                frameon=True,  # 通常图外的图例不画边框比较好看，如需边框改为True
                borderaxespad=0,
                edgecolor="none",
                facecolor="white",             # 白色背景
                framealpha=0.5,                # 半透明（0~1，越低越透明）
                **dict(loc="lower right", bbox_to_anchor=(0.97, 0.8)),
            )

    # --- [核心修改] 处理图例位置 ---
    if legend_pos and len(axes) > 0:
        first_ax = axes[0]

        # 定义位置参数
        if legend_pos == "top-left":
            # loc='lower left' 指图例框的左下角
            # bbox_to_anchor=(0, 1.02) 指轴坐标系的 (0, 1.02) 位置，即左上角上方一点点
            loc_params = dict(loc="lower left", bbox_to_anchor=(0, 1.02))
        elif legend_pos == "top-right":
            # loc='lower right' 指图例框的右下角
            # bbox_to_anchor=(1, 1.02) 指轴坐标系的 (1, 1.02) 位置，即右上角上方一点点
            loc_params = dict(loc="lower right", bbox_to_anchor=(1, 1.02))
        else:
            loc_params = dict(loc="best")

        # 绘制图例
        # borderaxespad=0 让图例紧贴着 bbox 定义的位置
        first_ax.legend(
            fontsize=legend_fontsize,
            frameon=False,  # 通常图外的图例不画边框比较好看，如需边框改为True
            borderaxespad=0,
            **loc_params,
        )

    if fig is not None:
        # 使用 constrained_layout=True 初始化 figure 通常比 tight_layout 更好，
        # 但如果必须用 tight_layout，这行代码有助于调整
        # plt.tight_layout()
        pass

    return fig, axes


def plot_parameter_distributions(
    data, titles=None, bins=30, median_color="blue"
):
    """
    绘制不同参数类型在时间维度上的分布直方图。

    参数:
    data (np.array): shape = (T, K, N)
                     T: 时间长度
                     K: 参数类型数量
                     N: 成分数量
    titles (list of str, optional): K个子图的标题。
    bins (int): 直方图分箱数。
    median_color (str): 中位数竖线的颜色。
    """

    # --- 数据检查 ---
    if data.ndim != 3:
        raise ValueError(
            f"Expected data with shape (T, K, N), but got {data.shape}"
        )

    T, K, N = data.shape

    # --- 处理标题 ---
    if titles is None:
        titles = [f"Parameter Type {k + 1}" for k in range(K)]
    elif len(titles) != K:
        print(
            f"Warning: Provided {len(titles)} titles, expected {K}. Using default titles."
        )
        titles = [f"Parameter Type {k + 1}" for k in range(K)]

    # --- 创建图像 ---
    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K), squeeze=False)

    for k in range(K):
        ax = axes[k, 0]

        # 提取当前参数类型所有时间和成分的值
        flat_data = data[:, k, :].flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # 去除 NaN

        # 绘制直方图
        ax.hist(
            flat_data, bins=bins, color="gray", alpha=0.7, edgecolor="black"
        )

        # 绘制中位数竖线
        median_val = np.median(flat_data)
        ax.axvline(
            median_val,
            color=median_color,
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {median_val:.3f}",
        )

        # 设置标题和样式
        ax.set_title(f"{titles[k]} - Value Distribution", fontsize=13)
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig, axes
