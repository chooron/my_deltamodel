import os
import sys
import pickle
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go

# 将项目根目录添加到路径中，以便导入 dmg 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dmg.models.hydromodel.xinanjiang import (
    xinanjiang_step_all,
    create_initial_state,
)  # noqa


def flow_conversion(
    attr,
    target,
):
    """Convert hydraulic flow from ft3/s to mm/day."""
    # attr[:, 11] 是 671 个流域的面积 (km2)，形状为 (671,)
    basin_area = attr[:, 11]
    # 将面积转换为 (671, 1, 1) 形状，以便与 (671, 12418, 1) 的 target 自动对齐
    area = basin_area[:, np.newaxis, np.newaxis]
    return (10**3) * target * 0.0283168 * 3600 * 24 / (area * (10**6))


def run_test():
    # --- 配置信息 ---
    gauge_id_to_test = 2415000
    dataset_path = os.path.join(project_root, "data/camels_dataset")
    gage_id_path = os.path.join(project_root, "data/gage_id.npy")
    params_csv_path = os.path.join(
        project_root,
        "project/diff_compare/tests/params/m_28_xinanjiang_12p_4s_calibrated_params.csv",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据集 (forcing: [流域, 时间, [P, T, PET]])
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        forcing, target, attr = pickle.load(f)
    print(f"Original Forcing shape: {forcing.shape}, Target shape: {target.shape}")

    # --- 时间范围选取 ---
    # 原数据集范围: 1980/10/01 - 2014/09/30
    # 目标范围: 1989/01/01 - 2009/12/31
    base_date = pd.to_datetime("1980/10/01")
    target_start = pd.to_datetime("1989/01/01")
    target_end = pd.to_datetime("2009/12/31")

    start_idx = (target_start - base_date).days
    end_idx = (target_end - base_date).days + 1  # +1 包含 2009/12/31 这一天

    # 截取指定时间段的数据
    forcing = forcing[:, start_idx:end_idx, :]
    target = target[:, start_idx:end_idx, :]
    print(f"SubsetName range: 1989/01/01 to 2009/12/31")
    print(f"Subset Forcing shape: {forcing.shape}, Subset Target shape: {target.shape}")

    target = flow_conversion(attr, target)

    # 2. 获取流域索引
    # 需要根据 gauge_id 找到 forcing/target 矩阵中的对应行
    gage_ids = np.load(gage_id_path)
    try:
        # 有时 ID 是 int，有时是 str 或者是带前导 0 的 str，这里做兼容处理
        basin_idx = np.where(gage_ids.astype(int) == int(gauge_id_to_test))[0][
            0
        ]
    except (ValueError, IndexError):
        basin_idx = np.where(gage_ids.astype(str) == str(gauge_id_to_test))[0][
            0
        ]

    print(f"Testing gauge_id {gauge_id_to_test} at index {basin_idx}")

    # 3. 加载并准备参数
    params_df = pd.read_csv(params_csv_path)
    basin_params = params_df[params_df["gauge_id"] == gauge_id_to_test].iloc[0]

    # 辅助函数：将参数转换为 torch tensor 且形状为 (1, 1)
    def to_tensor(val):
        return torch.tensor([[float(val)]], device=device, dtype=torch.float32)

    # 参数名称顺序与 xinanjiang_step 调用顺序一致:
    # aim, par_a, par_b, stot, fwm, flm, par_c, ex, ki, kg, ci, cg
    p = {
        "aim": to_tensor(basin_params["param_1"]),
        "par_a": to_tensor(basin_params["param_2"]),
        "par_b": to_tensor(basin_params["param_3"]),
        "stot": to_tensor(basin_params["param_4"]),
        "fwm": to_tensor(basin_params["param_5"]),
        "flm": to_tensor(basin_params["param_6"]),
        "par_c": to_tensor(basin_params["param_7"]),
        "ex": to_tensor(basin_params["param_8"]),
        "ki": to_tensor(basin_params["param_9"]),
        "kg": to_tensor(basin_params["param_10"]),
        "ci": to_tensor(basin_params["param_11"]),
        "cg": to_tensor(basin_params["param_12"]),
    }

    # 4. 准备强制力和观测数据 (截取该流域的所有时间步)
    # forcing 索引: 0: P (降水), 1: T (气温), 2: PET (潜在蒸散发)
    b_forcing = torch.tensor(
        forcing[basin_idx], device=device, dtype=torch.float32
    )
    b_target = torch.tensor(
        target[basin_idx], device=device, dtype=torch.float32
    )

    # 5. 初始化模型状态
    # n_grid=1, nmul=1
    S1, S2, S3, S4 = create_initial_state(1, 1, device)

    # 6. 时间步循环模拟
    num_steps = b_forcing.shape[0]
    
    # 准备存储所有结果的字典
    results = {
        "P": [], "T": [], "PET": [], "Q_obs": [],
        "Q_sim": [], "Ea": [], "flux_r": [], "flux_e": [],
        "flux_rs": [], "flux_ri": [], "flux_rg": [],
        "flux_qi": [], "flux_qg": [], "S1": [], "S2": [], "S3": [], "S4": []
    }

    print(f"Simulation in progress for {num_steps} steps...")
    for t in range(num_steps):
        # 提取当前时间步的输入
        P_t = b_forcing[t, 0:1].unsqueeze(0)  # (1, 1)
        T_t = b_forcing[t, 1:2].unsqueeze(0)  # (1, 1)
        PET_t = b_forcing[t, 2:3].unsqueeze(0)  # (1, 1)

        # 调用模型 step 函数
        (
            Qsim, Ea, flux_r, flux_e, flux_rs, flux_ri, flux_rg,
            flux_qi, flux_qg, S1, S2, S3, S4
        ) = xinanjiang_step_all(
            P_t, T_t, PET_t,
            p["aim"], p["par_a"], p["par_b"], p["stot"],
            p["fwm"], p["flm"], p["par_c"], p["ex"],
            p["ki"], p["kg"], p["ci"], p["cg"],
            S1, S2, S3, S4,
        )

        # 记录所有值
        results["P"].append(P_t.item())
        results["T"].append(T_t.item())
        results["PET"].append(PET_t.item())
        results["Q_obs"].append(target[basin_idx, t, 0].item())
        results["Q_sim"].append(Qsim.item())
        results["Ea"].append(Ea.item())
        results["flux_r"].append(flux_r.item())
        results["flux_e"].append(flux_e.item())
        results["flux_rs"].append(flux_rs.item())
        results["flux_ri"].append(flux_ri.item())
        results["flux_rg"].append(flux_rg.item())
        results["flux_qi"].append(flux_qi.item())
        results["flux_qg"].append(flux_qg.item())
        results["S1"].append(S1.item())
        results["S2"].append(S2.item())
        results["S3"].append(S3.item())
        results["S4"].append(S4.item())

    # 导出为 CSV
    results_df = pd.DataFrame(results)
    # 生成时间列
    date_range = pd.date_range(start=target_start, periods=num_steps, freq='D')
    results_df.insert(0, 'Date', date_range)
    
    output_csv = f"xinanjiang_results_gauge_{gauge_id_to_test}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"All simulation results saved to {output_csv}")

    # 7. 结果对比与计算指标
    q_sim = results_df["Q_sim"].values
    q_obs = results_df["Q_obs"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=q_obs, mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(y=q_sim, mode="lines", name="Simulated"))
    fig.update_layout(
        title=f"Runoff Comparison for gauge {gauge_id_to_test}",
        xaxis_title="Time Step",
        yaxis_title="Runoff (mm/day)",
        legend_title="Type",
    )
    fig.write_html("runoff_comparison.html")
    print(f"Interactive plot saved to runoff_comparison.html")

    # 计算 NSE (纳什效率系数)
    numerator = np.sum((q_obs - q_sim) ** 2)
    denominator = np.sum((q_obs - np.mean(q_obs)) ** 2)
    nse = 1 - (numerator / denominator)

    print("-" * 40)
    print(f"Test Result for gauge {gauge_id_to_test}:")
    print(f"NSE (Nash-Sutcliffe Efficiency): {nse:.4f}")

    # 打印前 10 个时间步的数值对比
    print("\nComparison (First 10 steps):")
    print(f"{'Step':<6} | {'Observed':<10} | {'Simulated':<10}")
    for i in range(min(10, num_steps)):
        print(f"{i:<6} | {q_obs[i]:<10.4f} | {q_sim[i]:<10.4f}")


if __name__ == "__main__":
    run_test()
