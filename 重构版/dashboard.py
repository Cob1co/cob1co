import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pathlib import Path

# ==========================================
# 配置与加载
# ==========================================
st.set_page_config(page_title="LMPC 智能微网控制台", layout="wide", page_icon="⚡")

# 设置项目根目录 (假设 dashboard.py 在项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'LMPC' / 'logs' / 'eval_results_march.pkl'

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # 将字典转换为 DataFrame 方便绘图
    time_idx = data['time']
    
    df = pd.DataFrame({'Time': time_idx})
    df.set_index('Time', inplace=True)
    
    # 提取 Baseline 数据
    df['Base_SOC'] = data['baseline']['soc']
    df['Base_Grid'] = data['baseline']['grid_power']
    df['Base_Cost'] = np.cumsum(data['baseline']['cost']) # 累计成本
    
    # 提取 Phase 3 数据
    df['LMPC_SOC'] = data['phase3']['soc']
    df['LMPC_Grid'] = data['phase3']['grid_power']
    df['LMPC_Cost'] = np.cumsum(data['phase3']['cost']) # 累计成本
    
    # 提取权重 (注意 weights 是 list of lists [soc, grid, cost])
    weights = np.array(data['phase3']['weights'])
    df['Alpha_SOC'] = weights[:, 0]
    df['Alpha_Grid'] = weights[:, 1]
    df['Alpha_Cost'] = weights[:, 2]
    
    return df, data['metrics']

# ==========================================
# 侧边栏控制
# ==========================================
st.sidebar.title("🎮 控制面板")

data_tuple = load_data()

if data_tuple is None:
    st.error(f"未找到数据文件: {DATA_PATH}")
    st.info("请先运行 evaluate_system.py 并确保它保存了结果。")
    st.stop()

df, metrics = data_tuple

# 日期选择器
min_date = df.index.min().date()
max_date = df.index.max().date()
st.sidebar.info(f"数据范围: {min_date} ~ {max_date}")

selected_date = st.sidebar.date_input(
    "选择查看日期",
    min_value=min_date,
    max_value=max_date,
    value=min_date
)

# 过滤当日数据
day_mask = df.index.date == selected_date
df_day = df[day_mask]

# ==========================================
# 主界面：核心 KPI
# ==========================================
st.title("⚡ Phase 3: Learning MPC 性能看板")

# 展示全月总指标
col1, col2, col3, col4 = st.columns(4)

# 解析 metrics 字典中的字符串 (移除 'MW', '%', '¥') 并转为浮点数以便计算
def parse_metric(val_str):
    return float(val_str.replace('%','').replace('MW','').replace('¥',''))

total_cost_base = metrics['总成本']['基线']
total_cost_lmpc = metrics['总成本']['Phase3']
improv_cost = metrics['总成本']['改善']

col1.metric("💰 全月总成本 (Baseline)", f"¥ {float(total_cost_base):,.0f}")
col2.metric("💰 全月总成本 (LMPC)", f"¥ {float(total_cost_lmpc):,.0f}", delta=improv_cost)
col3.metric("📉 Grid 跟踪误差", f"{metrics['电网跟踪误差']['Phase3']} MW", delta=metrics['电网跟踪误差']['改善'], delta_color="inverse")
col4.metric("🔋 SOC 跟踪误差", metrics['SOC跟踪误差']['Phase3'], delta=metrics['SOC跟踪误差']['改善'])

st.markdown("---")

if len(df_day) == 0:
    st.warning("所选日期没有数据。")
else:
    # ==========================================
    # 图表 1: SOC 轨迹对比
    # ==========================================
    st.subheader(f"🔋 SOC 轨迹对比 ({selected_date})")
    
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=df_day.index, y=df_day['Base_SOC'], name="专家基线 (Baseline)", 
                                 line=dict(color='gray', width=2, dash='dash')))
    fig_soc.add_trace(go.Scatter(x=df_day.index, y=df_day['LMPC_SOC'], name="LMPC 实际 (Phase 3)", 
                                 line=dict(color='#00CC96', width=3)))
    
    # 添加 SOC 限制线
    fig_soc.add_hline(y=0.9, line_dash="dot", annotation_text="Max SOC", annotation_position="bottom right")
    fig_soc.add_hline(y=0.1, line_dash="dot", annotation_text="Min SOC", annotation_position="bottom right")
    
    fig_soc.update_layout(height=400, hovermode="x unified", yaxis_title="SOC (0-1)")
    st.plotly_chart(fig_soc, use_container_width=True)

# ... (在 fig_soc 绘制代码之后)

    # ==========================================
    # 新增图表 1.5: SOC 差异放大镜 (Delta SOC)
    # ==========================================
    st.subheader("🔍 SOC 差异放大镜 (LMPC - Baseline)")
    delta_soc = df_day['LMPC_SOC'] - df_day['Base_SOC']
    
    # 如果差异全是 0，说明真的有问题；如果有波动，说明只是微调
    if delta_soc.abs().max() < 1e-6:
        st.error("⚠️ 警告：两条 SOC 曲线完全重合，数据可能未正确加载！")
    else:
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(
            x=df_day.index, 
            y=delta_soc, 
            fill='tozeroy',
            name="SOC 差异",
            line=dict(color='orange', width=1)
        ))
        fig_delta.update_layout(height=200, yaxis_title="SOC 差值", hovermode="x unified")
        st.plotly_chart(fig_delta, use_container_width=True)
    # ==========================================
    # 图表 2: 动态权重透视 (Transformer 大脑)
    # ==========================================
    st.subheader("🧠 Transformer 动态权重分析")
    st.caption("观察 Alpha 权重如何随时间变化：Alpha 越高代表 MPC 越'听话'，越低代表越'自由'。")
    
    fig_weights = go.Figure()
    fig_weights.add_trace(go.Scatter(x=df_day.index, y=df_day['Alpha_SOC'], name="α_SOC (跟踪电量)", line=dict(color='blue')))
    fig_weights.add_trace(go.Scatter(x=df_day.index, y=df_day['Alpha_Grid'], name="α_Grid (跟踪功率)", line=dict(color='orange')))
    fig_weights.add_trace(go.Scatter(x=df_day.index, y=df_day['Alpha_Cost'], name="α_Cost (省钱权重)", line=dict(color='red')))
    
    fig_weights.update_layout(height=350, hovermode="x unified", yaxis_title="权重值")
    st.plotly_chart(fig_weights, use_container_width=True)

    # ==========================================
    # 图表 3: 电网交互功率与成本累积
    # ==========================================
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("🔌 电网交互功率 (MW)")
        fig_grid = go.Figure()
        fig_grid.add_trace(go.Scatter(x=df_day.index, y=df_day['Base_Grid'], name="Baseline", line=dict(color='gray', width=1)))
        fig_grid.add_trace(go.Scatter(x=df_day.index, y=df_day['LMPC_Grid'], name="LMPC", line=dict(color='purple', width=2)))
        st.plotly_chart(fig_grid, use_container_width=True)
        
    with col_g2:
        st.subheader("💸 当日累计成本 (CNY)")
        # 计算当日的累计成本（减去当日0时刻的累计值）
        day_start_cost_base = df_day['Base_Cost'].iloc[0]
        day_start_cost_lmpc = df_day['LMPC_Cost'].iloc[0]
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=df_day.index, y=df_day['Base_Cost'] - day_start_cost_base, 
                                      name="Baseline 累积", fill='tozeroy', line=dict(color='gray')))
        fig_cost.add_trace(go.Scatter(x=df_day.index, y=df_day['LMPC_Cost'] - day_start_cost_lmpc, 
                                      name="LMPC 累积", fill='tozeroy', line=dict(color='green')))
        st.plotly_chart(fig_cost, use_container_width=True)
        st.markdown("---")
    st.subheader("📋 原始数据核对 (前 10 行)")
    check_cols = ['Base_SOC', 'LMPC_SOC', 'Alpha_SOC', 'Alpha_Grid', 'Base_Cost', 'LMPC_Cost']
    st.dataframe(df_day[check_cols].head(10).style.format("{:.4f}"))