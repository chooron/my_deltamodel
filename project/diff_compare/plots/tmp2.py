import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import os

# ================= 1. 配置区域 =================

BASE_DIR = '/workspace/my_deltamodel/project/diff_compare/plots/npz/'
N_LIST = [16, 32, 64, 128, 256]

FILES = {
    16:  os.path.join(BASE_DIR, 'param_snapshots_xinanjiang_n16.npz'),
    32:  os.path.join(BASE_DIR, 'param_snapshots_xinanjiang_n32.npz'),
    64:  os.path.join(BASE_DIR, 'param_snapshots_xinanjiang_n64.npz'),
    128: os.path.join(BASE_DIR, 'param_snapshots_xinanjiang_n128.npz'),
    256: os.path.join(BASE_DIR, 'param_snapshots_xinanjiang_n256.npz')
}

TARGET_EPOCHS = [0, 1] + list(range(5, 101, 5))

NUM_PARAMS = 12 
BASIN_IDX = 100  
# 补全参数名，防止不够用
PARAM_NAMES = [
    r"$K_{sat}$", r"$B$", r"$C$", r"$Ki$", r"$Kg$", 
    r"$CI$", r"$CG$", r"$Im$", r"$Ex$", r"$W$", r"$P_{11}$", r"$P_{12}$"
]

# 颜色定义 (Blues 渐变)
color_levels = np.linspace(0.3, 1.0, len(N_LIST)) 
COLORS = {n: cm.Blues(level) for n, level in zip(sorted(N_LIST), color_levels)}
LINE_WIDTHS = {n: 1.2 for n in N_LIST}

# ================= 2. 核心工具函数 (修正边界效应) =================

def bounded_kde_eval(values, x_grid, bounds):
    """
    镜像反射法：解决参数在边界处变成'斜线'的问题。
    """
    low, high = bounds
    # 1. 只有极小方差（单点堆积）时，返回尖峰
    if np.std(values) < 1e-6:
        y = np.zeros_like(x_grid)
        idx = (np.abs(x_grid - np.mean(values))).argmin()
        y[idx] = 10.0 # 任意高值
        return y

    # 2. 镜像数据 (Reflection Method)
    reflection_low = 2 * low - values
    reflection_high = 2 * high - values
    combined_values = np.concatenate([values, reflection_low, reflection_high])
    
    try:
        # bw_method 稍微调小一点，让曲线更精细
        kde = gaussian_kde(combined_values, bw_method=0.3) 
        return kde(x_grid) * 3  # *3 是因为数据量翻了3倍，保持密度积分为1
    except:
        return np.zeros_like(x_grid)

def load_all_data():
    cache = {}
    print("正在加载所有数据...")
    for n in N_LIST:
        path = FILES[n]
        if os.path.exists(path):
            try:
                raw = np.load(path)
                params = raw['params'] if 'params' in raw else raw[raw.files[0]]
                epochs = raw['epochs'] if 'epochs' in raw else np.arange(params.shape[0])
                cache[n] = {"params": params, "epochs": epochs}
                print(f"  -> N={n} OK")
            except Exception as e:
                print(f"  -> N={n} Error: {e}")
    return cache

# ================= 3. 绘图主逻辑 =================

def plot_comparison_ridgeline():
    data_cache = load_all_data()
    if not data_cache: return

    # 1. 计算每个参数的物理范围 (Global Bounds)
    param_ranges = {}
    print("计算参数范围...")
    for p_idx in range(NUM_PARAMS):
        vals = []
        for arr in data_cache.values():
            # shape: [epochs, basins, params, nmul]
            vals.append(arr["params"][:, BASIN_IDX, p_idx, :].reshape(-1))
        
        if vals:
            stacked = np.concatenate(vals)
            # 这里的 padding 是为了画图好看，不让线顶到框
            vmin, vmax = stacked.min(), stacked.max()
            padding = 0.05 * (vmax - vmin) if vmax > vmin else 0.1
            # 记录用于画图的视窗范围 (Display Range)
            param_ranges[p_idx] = (max(0, vmin - padding), vmax + padding) # 假设下限至少是0
        else:
            param_ranges[p_idx] = (0.0, 1.0)

    # 2. 确定 Epoch 列表
    all_eps = set()
    for c in data_cache.values():
        all_eps.update(c["epochs"])
    available_epochs = sorted([ep for ep in TARGET_EPOCHS if ep in all_eps])
    
    # 3. 预计算每个参数的“全局最大密度” (Global Max Density)
    # 目的：让 N=256 的尖峰看起来比 N=16 高，体现收敛性
    print("预计算最大密度用于缩放...")
    max_density_per_param = {}
    for p_idx in range(NUM_PARAMS):
        local_max = 0
        x_min, x_max = param_ranges[p_idx]
        x_grid = np.linspace(x_min, x_max, 200)

        for n_val in N_LIST:
            if n_val not in data_cache: continue
            # 扫描几个关键 epoch 找最大值
            check_epochs = [available_epochs[-1]] 
            full_data = data_cache[n_val]["params"]
            ep_arr = data_cache[n_val]["epochs"]
            ep_map = {int(e): i for i, e in enumerate(ep_arr)}
            
            for ep in check_epochs:
                if ep in ep_map:
                    idx = ep_map[ep]
                    v = full_data[idx, BASIN_IDX, p_idx, :]
                    # 假设物理边界是数据的极值
                    y = bounded_kde_eval(v, x_grid, (v.min(), v.max()))
                    if y.max() > local_max: local_max = y.max()
        
        max_density_per_param[p_idx] = local_max if local_max > 0 else 1.0

    # 4. 开始绘图
    # 调整画布宽度：12个参数需要更宽，否则很挤
    fig, axes = plt.subplots(1, NUM_PARAMS, figsize=(20, 10), sharey=True)
    if NUM_PARAMS == 1: axes = [axes] # 容错
    
    plt.subplots_adjust(wspace=0.1, bottom=0.08, top=0.92, left=0.06, right=0.98)
    
    n_layers = len(available_epochs)
    # 增大间距系数，防止太挤
    layer_gap = 1.0 / (n_layers + 1)

    for p_idx in range(NUM_PARAMS):
        ax = axes[p_idx]
        x_min, x_max = param_ranges[p_idx]
        x_grid = np.linspace(x_min, x_max, 200)
        
        # 物理边界：这里简单假设是绘图范围的收缩，或者你可以硬编码 (0,1)
        # 这一步决定了“反弹”的位置，非常关键
        phys_bounds = (x_min + 0.01, x_max - 0.01) 

        for layer_i, ep in enumerate(available_epochs):
            base_y = layer_i * layer_gap 

            for n_val in N_LIST:
                if n_val not in data_cache: continue
                full_data = data_cache[n_val]["params"]
                ep_arr = data_cache[n_val]["epochs"]
                ep_map = {int(e): i for i, e in enumerate(ep_arr)}
                
                if ep not in ep_map: continue
                idx = ep_map[ep]
                values = full_data[idx, BASIN_IDX, p_idx, :]

                # === 核心修改：使用 Bounded KDE ===
                # 这里我们假设数据的 min/max 就是物理边界，这样最安全
                current_bounds = (values.min(), values.max())
                y_density = bounded_kde_eval(values, x_grid, current_bounds)
                
                # === 核心修改：使用全局缩放，而非自身缩放 ===
                # 这样 N=256 会比 N=16 高
                global_max = max_density_per_param[p_idx]
                scaled_y = (y_density / global_max) * (layer_gap * 1.5) # 1.5是重叠系数
                
                final_y = base_y + scaled_y
                
                ax.plot(x_grid, final_y, color=COLORS[n_val], lw=LINE_WIDTHS[n_val], alpha=0.9)
            
            # Label
            if p_idx == 0:
                ax.text(x_min, base_y, f"Ep {ep}", ha='right', va='bottom', fontsize=9, fontweight='bold', color='#444')
            
            # Baseline
            ax.hlines(base_y, x_min, x_max, colors='gray', lw=0.5, alpha=0.2)

        # Decorations
        # 补全标题
        title = PARAM_NAMES[p_idx] if p_idx < len(PARAM_NAMES) else f"P{p_idx}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        mid_val = (x_min + x_max) / 2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xticks([mid_val])
        ax.set_xticklabels([f"{mid_val:.2f}"])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)

    # Legend
    handles = [plt.Line2D([0],[0], color=COLORS[n], lw=2) for n in N_LIST]
    fig.legend(handles, [f"N={n}" for n in N_LIST], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=False, fontsize=12)
    
    plt.savefig("Final_Ridgeline_Plot.png", dpi=300, bbox_inches='tight')
    print("Done.")
    plt.show()

plot_comparison_ridgeline()