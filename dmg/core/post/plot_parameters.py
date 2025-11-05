import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

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


def plot_parameters(
    data,
    titles=None,
    gray_color="0.7",
    median_color="blue",
    smooth_method=None,
    axes=None,
    fig=None,
    ts=None,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    show_ylabel=True,
    **smooth_kwargs,
):
    """
    绘制参数序列图，突出显示中位数成分，并可选择应用平滑。

    参数:
    data (np.array): 输入数据，期望 shape 为 (T, K, N)
                     T: 序列长度
                     K: 参数/类型数量
                     N: 成分数量
    titles (list of str): K个子图的标题列表，用于legend。
    gray_color (str): N个成分的默认颜色。
    median_color (str): 中位数成分的突出显示颜色。
    smooth_method (str, optional): 平滑方法，传递给 smooth_sequences。
                                   如果为 None，则不进行平滑。
    axes (matplotlib.axes.Axes or list or np.ndarray, optional): 预先提供的单个Axes对象（当K=1时）、Axes列表（长度为K）或Axes数组（将展平后取前K个）。
                                                         例如 ax、[ax0, ax1, ...] 或 axes[0,:]（对于2x3数组的某一行）。
                                                         如果为 None，则创建新的。
    fig (matplotlib.figure.Figure, optional): 预先提供的figure对象。
                                              如果axes为None，则需要创建新的fig。
    title_fontsize (int): 标题字体大小 (默认 14)。(已弃用，因为title被移除)
    label_fontsize (int): x/y标签字体大小 (默认 12)。
    tick_fontsize (int): 刻度标签字体大小 (默认 10)。
    legend_fontsize (int): 图例字体大小 (默认 10)。
    show_ylabel (bool): 是否展示 y 轴 label (默认 True，仅在第一个子图显示，如果共享 y 轴)。
    **smooth_kwargs: 平滑方法特定的参数，传递给 smooth_sequences。
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
    if smooth_method is not None:
        data = smooth_sequences(data, method=smooth_method, **smooth_kwargs)

    # 处理axes参数
    import matplotlib.axes

    if axes is None:
        # 如果没有提供axes，创建新的fig和axes，并共享x和y轴
        if fig is None:
            fig, axes_temp = plt.subplots(
                K, 1, figsize=(14, 5 * K), sharex=True, sharey=True
            )
            axes = [axes_temp] if K == 1 else list(axes_temp)
        else:
            axes = [
                fig.add_subplot(K, 1, k + 1) for k in range(K)
            ]  # 注意: add_subplot 不直接支持共享，但用户可预设
    elif isinstance(axes, matplotlib.axes.Axes):
        # 如果传入的是单个Axes对象，假设K=1
        if K != 1:
            raise ValueError(
                f"Provided a single Axes, but data has K={K} parameter types. Provide a list or array of {K} Axes."
            )
        axes = [axes]
    else:
        # 假设是列表或numpy数组
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()  # 展平多维数组，例如处理axes[0,:]
        if len(axes) < K:
            raise ValueError(
                f"Provided axes has length {len(axes)}, but expected at least {K}."
            )
        axes = list(axes)[:K]  # 转换为列表并取前K个

    if ts is None:
        ts = np.arange(T)

    # 遍历 K 个参数类型
    for k in range(K):
        ax = axes[k]  # 获取当前子图

        # 提取当前参数的所有成分数据, shape (T, N)
        param_data = data[:, k, :]

        # 1. 计算每个成分的“时间平均值”
        temporal_averages = np.nanmean(param_data, axis=0)  # shape (N,)

        # 2. 找到所有平均值的中位数
        median_avg_value = np.median(temporal_averages)

        # 3. 找到平均值最接近中位数的那个成分的索引
        median_component_index = np.argmin(
            np.abs(temporal_averages - median_avg_value)
        )

        # 4. 提取中位数成分的序列数据
        median_sequence = param_data[:, median_component_index]

        # 5. 绘制所有 N 个成分的灰色线条
        for i in range(N):
            ax.plot(
                ts, param_data[:, i], color=gray_color, alpha=0.4, linewidth=1.
            )

        # 6. 绘制中位数成分的彩色线条，使用titles[k]作为label
        ax.plot(
            ts,
            median_sequence,
            color=median_color,
            linewidth=2.0,
            label=titles[k],
        )

        # 7. 设置图表样式
        # 无title
        if k == K - 1:  # 仅最下方的ax设置x label
            ax.set_xlabel("Year", fontsize=label_fontsize)
        if show_ylabel:  # 仅最左上方的ax设置y label（如果启用）
            ax.set_ylabel(titles[k], fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        # 对于x ticks，仅最下方显示
        if k != K - 1:
            ax.tick_params(bottom=False, labelbottom=False)
        ax.grid(True, linestyle="--", alpha=0.5)

    # 如果创建了新的fig，调整布局
    if fig is not None:
        plt.tight_layout()

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
