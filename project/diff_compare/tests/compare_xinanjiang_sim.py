import pandas as pd
import plotly.graph_objects as go

origin_df = pd.read_csv("/workspace/my_deltamodel/project/diff_compare/tests/data/xinanjiang_1013500_states.csv")
our_df = pd.read_csv("/workspace/my_deltamodel/project/diff_compare/tests/data/xinanjiang_results_gauge_1013500.csv")

origin_col = "Sim_q"
our_col = "Q_sim"

# 绘图代码
fig = go.Figure()

# 绘制原始结果
fig.add_trace(go.Scatter(
    x=our_df['Date'], 
    y=origin_df[origin_col], 
    mode='lines', 
    name=f'Original ({origin_col})',
    line=dict(color='blue', width=1.5)
))

# 绘制我们的结果
fig.add_trace(go.Scatter(
    x=our_df['Date'], 
    y=our_df[our_col], 
    mode='lines', 
    name=f'Our ({our_col})',
    line=dict(color='red', width=1.5, dash='dash')
))

fig.update_layout(
    title="Xinanjiang Simulation Comparison",
    xaxis_title="Date",
    yaxis_title="Runoff (mm/day)",
    legend_title="Sources",
    hovermode="x unified",
    template="plotly_white"
)

fig.write_html("comparison_result.html")
print("Comparison Plot saved to comparison_result.html")