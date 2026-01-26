import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import os
from pathlib import Path
# ================= 1. 配置区域 =================

# 基础路径 (请根据实际情况调整或保持不变)
BASE_DIR = '/workspace/my_deltamodel/project/diff_compare/plots/npz/'

# 定义要对比的 N 列表 (从小到大)
N_LIST = [16, 32, 64, 128, 256]

MODEL_NAME = 'hymod'
baseline_file = f"{MODEL_NAME}_runs_iters_params.npy" # 10×10001×5, 重复的次数，epoch的总数，参数个数
BASELINE_COLOR = "#D64045"

# 文件字典
FILES = {
    16:  os.path.join(BASE_DIR, f'param_snapshots_{MODEL_NAME}_n16.npz'),
    32:  os.path.join(BASE_DIR, f'param_snapshots_{MODEL_NAME}_n32.npz'),
    64:  os.path.join(BASE_DIR, f'param_snapshots_{MODEL_NAME}_n64.npz'),
    128: os.path.join(BASE_DIR, f'param_snapshots_{MODEL_NAME}_n128.npz'),
    256: os.path.join(BASE_DIR, f'param_snapshots_{MODEL_NAME}_n256.npz')
}

# 想要展示的 Epoch (Y轴层级，保持原始标注)
TARGET_EPOCHS = [0] + list(range(10, 101, 10))


COLLIE1_PARAMS_BOUNDS = {
    "Smax": [1.0, 2000.0],
}

HYMOD_PARAMS_BOUNDS = {
    "smax": [1.0, 2000.0],
    "b_exp": [0.0, 10.0],
    "a_split": [0.0, 1.0],
    "kf": [0.0, 1.0],
    "ks": [0.0, 1.0],
}

JULIA_HYMOD_PARAMS = ['Smax', 'b_exp', 'a_split', 'kf', 'ks']

XINANJIANG_PARAMS_BOUNDS = {
    "aim": [0.0, 1.0],
    "par_a": [-0.49, 0.49],
    "par_b": [0.0, 10.0],
    "stot": [1.0, 2000.0],
    "fwm": [0.01, 0.99],
    "flm": [0.01, 0.99],
    "par_c": [0.01, 0.99],
    "ex": [0.0, 10.0],
    "ki": [0.0, 1.0],
    "kg": [0.0, 1.0],
    "ci": [0.0, 1.0],
    "cg": [0.0, 1.0],
}

HBV96_PARAMS_BOUNDS = {
    "tt": [-3.0, 5.0],
    "tti": [0.0, 17.0],
    "ttm": [-3.0, 3.0],
    "cfr": [0.0, 1.0],
    "cfmax": [0.0, 20.0],
    "whc": [0.0, 1.0],
    "cflux": [0.0, 4.0],
    "fc": [1.0, 2000.0],
    "lp": [0.05, 0.95],
    "beta": [0.0, 10.0],
    "k0": [0.0, 1.0],
    "alpha": [0.0, 4.0],
    "perc": [0.0, 20.0],
    "k1": [0.0, 1.0],
}

# 参数 Latex 标签
PARAM_LATEX = {
    "collie1": {
        "Smax": r"$S_{max}$",
    },
    "hymod": {
        "smax": r"$S_{max}$",
        "b_exp": r"$b_{exp}$",
        "a_split": r"$a_{split}$",
        "kf": r"$k_{f}$",
        "ks": r"$k_{s}$",
    },
    "xinanjiang": {
        "aim": r"$a_{im}$",
        "par_a": r"$a$",
        "par_b": r"$b$",
        "stot": r"$S_{tot}$",
        "fwm": r"$f_{wm}$",
        "flm": r"$f_{lm}$",
        "par_c": r"$c$",
        "ex": r"$e_{x}$",
        "ki": r"$k_{i}$",
        "kg": r"$k_{g}$",
        "ci": r"$c_{i}$",
        "cg": r"$c_{g}$",
    },
    "hbv96": {
        "tt": r"$T_{t}$",
        "tti": r"$T_{ti}$",
        "ttm": r"$T_{tm}$",
        "cfr": r"$c_{fr}$",
        "cfmax": r"$c_{fmax}$",
        "whc": r"$w_{hc}$",
        "cflux": r"$c_{flux}$",
        "fc": r"$F_{c}$",
        "lp": r"$L_{p}$",
        "beta": r"$\beta$",
        "k0": r"$k_{0}$",
        "alpha": r"$\alpha$",
        "perc": r"$P_{erc}$",
        "k1": r"$k_{1}$",
    },
}


# 参数配置
BASIN_IDX = 7

# ================= 2. 颜色定义 (Blues 渐变，nmul 越大越深) =================
color_levels = np.linspace(0.25, 0.95, len(N_LIST))  # 扩大深浅差异
COLORS = {n: cm.Blues(level) for n, level in zip(sorted(N_LIST), color_levels)}
# 线宽保持适中
LINE_WIDTHS = {n: 1.1 for n in N_LIST}

# 采用 Figure6 的风格
plt.rcParams.update({
    "font.family": "serif",
    'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix',
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})

# ================= 3. 数据加载 =================
def load_all_data():
    cache = {}
    print("正在加载所有数据...")
    for n in N_LIST:
        path = FILES[n]
        if os.path.exists(path):
            try:
                raw = np.load(path)
                # 兼容不同保存格式
                params = raw['params'] if 'params' in raw else raw[raw.files[0]]
                epochs = raw['epochs'] if 'epochs' in raw else np.arange(params.shape[0])
                cache[n] = {"params": params, "epochs": epochs}
                print(f"  -> N={n} 加载成功")
            except Exception as e:
                print(f"  -> N={n} 读取失败: {e}")
        else:
            print(f"  -> N={n} 文件不存在")
    return cache


def load_baseline_data():
    """Load baseline npy (runs x iters x params) for hymod and align param order."""
    if MODEL_NAME != "hymod":
        return None
    path = os.path.join(BASE_DIR, baseline_file)
    if not os.path.exists(path):
        print(f"Baseline file not found: {path}")
        return None
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"Baseline load failed: {exc}")
        return None

    if arr.ndim != 3:
        print(f"Unexpected baseline shape {arr.shape}, expect 3D")
        return None

    # map baseline params to current param order
    baseline_params = JULIA_HYMOD_PARAMS
    target_params = list(HYMOD_PARAMS_BOUNDS.keys())
    idx_map = []
    for name in target_params:
        if name in baseline_params:
            idx_map.append(baseline_params.index(name))
        elif name.lower() in [p.lower() for p in baseline_params]:
            idx_map.append([p.lower() for p in baseline_params].index(name.lower()))
        else:
            idx_map.append(None)

    reordered = np.full((arr.shape[0], arr.shape[1], len(target_params)), np.nan, dtype=arr.dtype)
    for tgt_i, src_i in enumerate(idx_map):
        if src_i is None or src_i >= arr.shape[2]:
            continue
        reordered[:, :, tgt_i] = arr[:, :, src_i]

    baseline_steps = [0] + list(range(1000, min(10000, arr.shape[1] - 1) + 1, 1000))
    return {"values": reordered, "steps": baseline_steps}

# ================= 4. 绘图逻辑 =================

def plot_comparison_ridgeline():
    bounds_map = {
        "collie1": COLLIE1_PARAMS_BOUNDS,
        "hymod": HYMOD_PARAMS_BOUNDS,
        "xinanjiang": XINANJIANG_PARAMS_BOUNDS,
        "hbv96": HBV96_PARAMS_BOUNDS,
    }

    if MODEL_NAME not in bounds_map:
        raise ValueError(f"未知模型: {MODEL_NAME}")

    data_cache = load_all_data()
    if not data_cache:
        print("无数据，终止。"); return

    baseline_cache = load_baseline_data()

    param_bounds = bounds_map[MODEL_NAME]
    param_names = list(param_bounds.keys())
    label_map = PARAM_LATEX.get(MODEL_NAME, {})
    num_params = len(param_names)

    # 为每个参数计算实际取值范围，便于直接展示原始尺度
    param_ranges = {}
    for p_idx in range(num_params):
        vals = []
        for arr in data_cache.values():
            # arr shape: [epochs, basins, params, nmul]
            vals.append(arr["params"][:, :, p_idx, :].reshape(-1))
        if vals:
            stacked = np.concatenate(vals)
            vmin, vmax = stacked.min(), stacked.max()
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                padding = 0.02 * (vmax - vmin)
                param_ranges[p_idx] = (vmin - padding, vmax + padding)
            else:
                param_ranges[p_idx] = (vmin - 0.1, vmin + 0.1)
        else:
            param_ranges[p_idx] = (0.0, 1.0)

    # 按实际可用 epoch 重新确定层级
    available_epochs = sorted({ep for cache in data_cache.values() for ep in cache["epochs"] if ep in TARGET_EPOCHS})
    if not available_epochs:
        print("指定的 epochs 在数据中不存在，终止。")
        return

    # 将 baseline 迭代转为等效 epoch (step/100) 并与可用 epoch 合并为同一层
    baseline_layers = {}
    if baseline_cache is not None:
        for step in baseline_cache["steps"]:
            pseudo_ep = step / 100.0
            if pseudo_ep not in baseline_layers:
                baseline_layers[pseudo_ep] = []
            baseline_layers[pseudo_ep].append(step)

    layer_values = sorted(set(available_epochs) | set(baseline_layers.keys()))
    layer_labels = [("ep", ep) for ep in layer_values]

    # 创建画布: 根据参数数量自适应宽度
    fig_width = num_params * 1.5
    fig_height = 10
    fig, axes = plt.subplots(1, num_params, figsize=(fig_width, fig_height), sharey=True)
    axes = np.atleast_1d(np.array(axes).ravel())

    # 定义边距变量以便复用
    top_margin = 0.92
    bottom_margin = 0.08
    left_margin = 0.06
    right_margin = 0.95

    # 调整间距
    plt.subplots_adjust(wspace=0.1, bottom=bottom_margin, top=top_margin, left=left_margin, right=right_margin)

    # ================== <--- 修改点 1: 添加全局大外框 ==================
    # 创建一个新的 axes，覆盖整个子图区域。
    # [left, bottom, width, height]
    rect_ax = fig.add_axes([left_margin, bottom_margin, right_margin - left_margin, top_margin - bottom_margin], zorder=0.5)
    # 隐藏刻度和背景，只保留边框
    rect_ax.set_xticks([])
    rect_ax.set_yticks([])
    rect_ax.set_facecolor('none')  # 设置背景透明
    # 可以稍微加粗一下外框线以示区别
    for spine in rect_ax.spines.values():
        spine.set_linewidth(1.2)
    # =================================================================

    # 遍历参数 (子图)
    n_layers = len(layer_labels)
    layer_gap = 1.0 / max(n_layers - 1, 1)
    y_min = -0.05 * layer_gap
    y_max = 1.0 + 0.9 * layer_gap  # 给最上层留足空间

    for p_idx in range(num_params):
        ax = axes[p_idx]

        # --- 遍历 Y 轴的每一层 ---
        for layer_i, (layer_kind, layer_val) in enumerate(layer_labels):

            base_y = layer_i * layer_gap  # 将层级映射到 0-1 区间

            ep = layer_val

            # 先画 NMUL 的 KDE 曲线（仅当该 ep 存在于数据中）
            if ep in available_epochs:
                for n_val in N_LIST:
                    if n_val not in data_cache:
                        continue
                    full_data = data_cache[n_val]["params"]
                    epochs_arr = data_cache[n_val]["epochs"]
                    epoch_to_idx = {int(e): i for i, e in enumerate(epochs_arr)}

                    if ep not in epoch_to_idx:
                        continue
                    ep_idx = epoch_to_idx[ep]

                    values = full_data[ep_idx, BASIN_IDX, p_idx, :]

                    try:
                        x_min, x_max = param_ranges.get(p_idx, (values.min(), values.max()))
                        kde = gaussian_kde(values, bw_method='scott')
                        x_grid = np.linspace(x_min, x_max, 200)
                        y_density = kde(x_grid)
                        if y_density.max() > 0:
                            y_density = y_density / y_density.max()
                    except Exception:
                        x_min, x_max = param_ranges.get(p_idx, (0.0, 1.0))
                        x_grid = np.linspace(x_min, x_max, 200)
                        y_density = np.zeros_like(x_grid)

                    final_y = base_y + y_density * (layer_gap * 0.8)

                    ax.plot(
                        x_grid,
                        final_y,
                        color=COLORS[n_val],
                        linewidth=LINE_WIDTHS[n_val],
                        alpha=0.9,
                        label=f"N={n_val}" if (p_idx == 0 and layer_i == 0) else "",
                    )

            # 再画 baseline 迭代柱（如果该层对应 baseline）
            if baseline_cache is not None and ep in baseline_layers:
                for step in baseline_layers[ep]:
                    step_idx = step if step < baseline_cache["values"].shape[1] else None
                    if step_idx is None:
                        continue
                    bars = baseline_cache["values"][:, step_idx, p_idx]
                    bars = bars[np.isfinite(bars)]
                    if bars.size == 0:
                        continue
                    bar_height = layer_gap * 0.8
                    ax.vlines(
                        bars,
                        base_y,
                        base_y + bar_height,
                        color=BASELINE_COLOR,
                        linewidth=0.8,
                        alpha=0.7,
                        label="Baseline" if (p_idx == 0 and layer_i == 0) else None,
                    )
                    if p_idx == 0:
                        ax.text(
                            param_ranges.get(p_idx, (0, 1))[0] - 0.02 * (param_ranges.get(p_idx, (0, 1))[1] - param_ranges.get(p_idx, (0, 1))[0]),
                            base_y + bar_height * 0.5,
                            f"Iter {step} (Ep {ep:g})",
                            ha='right',
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='#333',
                        )

            # 添加一条淡淡的基准线 (地板)
            ax.hlines(base_y, *param_ranges.get(p_idx, (0.0, 1.0)), colors='gray', linestyles='-', lw=0.5, alpha=0.2)

        # 子图装饰
        x_min, x_max = param_ranges.get(p_idx, (0.0, 1.0))
        x_mid = 0.5 * (x_min + x_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([x_mid])
        ax.set_xticklabels([f"{x_mid:.2f}"])
        ax.set_yticks([]) # 隐藏刻度

        # 去边框 (保持不变，内部使用脊线图风格)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        label = label_map.get(param_names[p_idx], param_names[p_idx])
        ax.set_xlabel(label, fontsize=11, fontweight='bold')

    # 添加全局图例 (Legend) - 只在顶部加一个
    # 获取第一个子图的句柄来生成图例
    handles, labels = axes[0].get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l]
    if filtered:
        handles, labels = zip(*filtered)
        # 计算总项数（标题+图例项）用于设置 ncol
        num_items = len(handles) + 1
        title_handle = plt.Line2D([], [], linestyle="none", marker=None, color="none")

        # ================== <--- 修改点 2: 修改图例样式 ==================
        fig.legend(
            [title_handle] + list(handles),
            ["Number of Start:"] + list(labels), # 稍微修改标题加个冒号
            loc='upper center',
            # 稍微向下调整一点 y 坐标，避免紧贴大边框
            bbox_to_anchor=(0.5, top_margin + 0.06),
            ncol=num_items,      # <--- 强制所有项在一行
            frameon=False,
            fontsize=12,
            columnspacing=1.2,   # <--- 调整列间距使一行更紧凑
            handlelength=1.0,    # <--- 缩短图例线的长度 (原为 2.0，或默认值)
            handletextpad=0.5,   # <--- 调整线和文字之间的间距
        )
        # ===============================================================
    cur_dir = Path(__file__).resolve().parent
    figures_dir = cur_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / f"Figure_7_N_Comparison_Ridgeline_{MODEL_NAME}.png"
    # 增加 pad_inches 以确保大边框和图例不被裁切
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"图表生成完毕: {save_path}")
    plt.show()

# 执行
plot_comparison_ridgeline()