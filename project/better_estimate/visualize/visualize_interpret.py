import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
font_family = "Times New Roman"
plt.rcParams.update(
    {
        "font.family": font_family,
        "font.serif": [font_family],
        "mathtext.fontset": "custom",
        "mathtext.rm": font_family,
        "mathtext.it": font_family,
        "mathtext.bf": font_family,
        "axes.unicode_minus": False,
    }
)

# --- 1. 配置参数 ---
base_path = Path(__file__).parent.parent / "interpret" / "results"
print(base_path)
basin_id = "1466500"
# 定义两个对比的模型/结果路径
models = [
    {"name": "LSTM", "path": base_path / "lstm" / basin_id / "tint_contribs.npz"},
    {"name": "Transformer", "path": base_path / "hopev1" / basin_id / "tint_contribs.npz"} # 示例路径
]

time_indices = [100, 365, 730]  # 3个时间点
output_dims = 3                 # 每个时间点展示3个输出维度
display_len = 100               # 纵向只看100个时间步
feature_count = 3               # 横向只看3个特征

# --- 2. 数据加载函数 ---
def get_data(file_path):
    try:
        # 尝试加载真实数据
        raw = np.load(file_path)["data"]
        # 假设原始维度: (Sample, Time, Feat, Out, ...) -> 取中位数 -> (Sample, Time, Feat, Out)
        # 如果你的数据维度已经是 (730, 730, 38, 3)，则不需要 reshape
        if raw.ndim == 5:
            return np.median(raw, axis=-1)
        return raw
    except Exception:
        # 如果文件不存在，生成模拟数据用于看效果
        return np.random.randn(730, 730, 38, 3)

# 预加载两个模型的数据
data_list = [get_data(m["path"]) for m in models]

# --- 3. 绘图主逻辑 ---
# 计算总列数 = 时间点数量 * 输出维度数量 = 3 * 3 = 9
total_cols = len(time_indices) * output_dims 
total_rows = len(models)

# 设置一个很宽的画布，适应9列竖长的图
fig, axes = plt.subplots(
    nrows=total_rows, 
    ncols=total_cols, 
    figsize=(18, 8),     # 宽18，高8，保证竖长条不被压扁
    sharex=True,         # 所有子图共享X轴范围
    sharey=True,         # 所有子图共享Y轴范围（重要！方便对比时间步）
    constrained_layout=True # 自动调整间距
)

cmap = matplotlib.cm.coolwarm
cmap.set_bad("white")

output_names = [r"$\beta$", r"$\gamma$", r"$K_0$"]
feature_names = ["P", "T", "PET"]
# 遍历行 (模型)
for row_idx in range(total_rows):
    data = data_list[row_idx]
    model_name = models[row_idx]["name"]
    
    # 遍历列 (时间点 x 输出维度)
    # 我们需要用双重循环来填充这9列
    col_counter = 0 
    
    for t_idx, time_val in enumerate(time_indices):
        for out_idx in range(output_dims):
            # 获取当前的子图对象
            ax = axes[row_idx, col_counter]
            
            # --- 数据切片 ---
            # Sample: time_val - 1
            # Time: :display_len (前100)
            # Feature: :feature_count (前3)
            # Output: out_idx
            heat = data[time_val - 1, :time_val, :feature_count, out_idx]
            
            # --- 噪声处理 ---
            max_val = np.max(np.abs(heat))
            if max_val > 0:
                heat[np.abs(heat) < max_val * 0.1] = np.nan
            
            # --- 绘图 ---
            im = ax.imshow(
                heat,
                aspect="auto",
                cmap=cmap,
                vmin=np.nanmin(heat) * 0.9,
                vmax=np.nanmax(heat) * 0.9,
                interpolation='nearest'
            )
            
            # --- 顶部标题 (仅在第一行显示) ---
            # 逻辑：显示 "T=100 (Out 0)", "T=100 (Out 1)"...
            if row_idx == 0:
                # 为了美观，只在每组时间点的中间那个图(out_idx=1)显示大标题
                if out_idx == 1:
                    ax.set_title(f"Time Point = {time_val}\n{output_names[out_idx]}", fontsize=20)
                else:
                    ax.set_title(output_names[out_idx], fontsize=20)
            else:
                ax.tick_params(axis='y',
                left=False,      # 不显示刻度短横线
                labelleft=False) # 不显示数字
            


            
            # --- 底部标签 (仅在最后一行显示) ---
            if row_idx == total_rows - 1:
                # 1. 设置刻度位置：0, 1, 2...
                ax.set_xticks(range(feature_count))
                
                # 2. 设置刻度文本（特征名称）
                # rotation=45: 旋转45度，防止文字挤在一起
                # ha='right': 对齐方式，让旋转后的文字看起来更整齐
                ax.set_xticklabels(feature_names,  ha='center', fontsize=20)
                
                # 3. 只有中间的图显示总标题 "Feature"，避免重复
                if out_idx == 1:
                    ax.set_xlabel("Input Features", fontsize=20, labelpad=10)
                
                # 确保底部刻度线是可见的
                ax.tick_params(bottom=True)
            
            else:
                # 非最后一行的图，彻底隐藏X轴刻度和标签
                ax.set_xticks([])
                ax.tick_params(bottom=False, left=False)
            
                        # --- 左侧标签 (仅在第一列显示) ---
            if col_counter == 0:
                ax.set_ylabel(f"Time Step", fontsize=20)
                # ax.tick_params(axis='y', labelsize=18)
                ax.tick_params(axis='y', labelsize=18, length=5, width=1.5,
                               direction='out', left=True, labelleft=True)
            
            # (这一行原来的代码删除或注释掉，因为上面已经分别处理了 tick_params)
            # ax.tick_params(left=False, bottom=False) 
            
            col_counter += 1

# 添加颜色条 (放在最右边，共用一个)
cbar = fig.colorbar(im, ax=axes, location='right', aspect=40, pad=0.02)

# 【新增修改】设置色标刻度数值的字体大小
cbar.ax.tick_params(labelsize=18) 

# 【新增修改】设置色标标题 (Label) 的字体大小
cbar.set_label("Contribution", fontsize=20, labelpad=18)

# 保存
save_path = Path(__file__).parent / "figures" / "global_comparison_heatmap.png"
plt.savefig(save_path, dpi=600)
plt.close()
print(f"图表已生成: {save_path}")