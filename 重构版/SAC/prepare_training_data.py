"""数据预处理脚本：为第二阶段训练准备数据

功能：
1. 读取原始天气数据（data/data2023.csv）
2. 调用第一阶段的WindFarm/PVPlant模型计算风光出力
3. 使用scenario_manager进行聚类，为每天打上Day_Label标签
4. 保存为训练数据（PPO/clustered_training_data.csv）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import WindFarm, PVPlant
from models.params import load_model_config

# SA-MBPS文件夹名包含连字符，需要动态导入
import importlib.util
sa_mbps_path = project_root / "SA-MBPS" / "scenario_manager.py"
spec = importlib.util.spec_from_file_location("scenario_manager", sa_mbps_path)
scenario_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenario_manager)
split_days = scenario_manager.split_days
build_scenarios = scenario_manager.build_scenarios


def load_capacity_from_phase1(phase1_result_path: Path = None):
    """从第一阶段结果加载容量配置
    
    TODO: 根据实际情况修改加载方式
    方式1：从保存的文件读取
    方式2：直接手动输入第一阶段的输出
    """
    # 临时方案：手动设置（运行第一阶段后填入）
    capacity = {
        "wind_mw": 20.0,      # 替换为第一阶段输出的值
        "pv_mw": 35.0,        # 替换为第一阶段输出的值
        "ts_mwh": 200.0,      # 替换为第一阶段输出的值
        "eh_mw_th": 25.0,     # 替换为第一阶段输出的值
        "st_mw_e": 15.0,      # 替换为第一阶段输出的值
    }
    
    print("使用第一阶段容量配置：")
    for k, v in capacity.items():
        print(f"  {k}: {v:.2f}")
    
    return capacity


def build_renewable_models(capacity: dict):
    """构建风电和光伏模型（复用第一阶段）"""
    cfg = load_model_config()
    assets = cfg.get("assets", {})
    
    # 风电模型
    wind_cfg = assets.get("wind", {})
    power_curve = [
        (0.0, 0.0),
        (wind_cfg.get("v_in_mps", 3.0), 0.0),
        (wind_cfg.get("v_rated_mps", 12.0), 1.0),
        (wind_cfg.get("v_out_mps", 25.0), 0.0),
    ]
    wind = WindFarm(capacity_mw=capacity["wind_mw"], power_curve=power_curve)
    
    # 光伏模型
    pv_cfg = assets.get("pv", {})
    pv = PVPlant(
        capacity_mw=capacity["pv_mw"],
        ghi_ref=pv_cfg.get("ghi_ref_w_per_m2", 1000.0),
        temp_coeff=pv_cfg.get("temp_coeff_per_c", -0.004),
        t_ref_c=pv_cfg.get("t_ref_c", 25.0),
    )
    
    return wind, pv


def compute_renewable_output(df: pd.DataFrame, wind: WindFarm, pv: PVPlant):
    """逐时计算风光出力
    
    参数：
        df: 原始数据，包含 Temperature_C, Solar_W_m2, Wind_Speed_m_s
        wind: 风电模型
        pv: 光伏模型
    
    返回：
        添加了 Wind_Gen_MW 和 PV_Gen_MW 列的DataFrame
    """
    print("\n开始计算风光出力...")
    
    wind_gen_list = []
    pv_gen_list = []
    
    for idx, row in df.iterrows():
        # 风电出力
        wind.reset()
        wind_out = wind.step(
            {"wind_speed": row["Wind_Speed_m_s"]},
            dt_hours=1.0
        )
        wind_gen_list.append(wind_out["p_mw"])
        
        # 光伏出力
        pv.reset()
        pv_out = pv.step(
            {"ghi": row["Solar_W_m2"], "temp": row["Temperature_C"]},
            dt_hours=1.0
        )
        pv_gen_list.append(pv_out["p_mw"])
        
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 条数据")
    
    df["Wind_Gen_MW"] = wind_gen_list
    df["PV_Gen_MW"] = pv_gen_list
    df["REN_Gen_MW"] = df["Wind_Gen_MW"] + df["PV_Gen_MW"]
    
    print(f"✓ 风光出力计算完成")
    print(f"  风电年发电量: {df['Wind_Gen_MW'].sum():.2f} MWh")
    print(f"  光伏年发电量: {df['PV_Gen_MW'].sum():.2f} MWh")
    print(f"  总可再生发电: {df['REN_Gen_MW'].sum():.2f} MWh")
    
    return df


def cluster_days_by_weather(df: pd.DataFrame, k_typical: int = 5):
    """使用第一阶段的聚类方法为每天打标签
    
    参数：
        df: 包含逐时数据的DataFrame
        k_typical: 典型日数量（默认5）
    
    返回：
        添加了 Day_Index 和 Day_Label 列的DataFrame
    """
    print(f"\n开始天气聚类（k={k_typical}）...")
    
    # 按天切分
    daily_dfs = split_days(df)
    print(f"  共 {len(daily_dfs)} 天数据")
    
    # 调用第一阶段的聚类方法
    scenarios_cfg = {
        "k_typical": k_typical,
        "random_state": 42
    }
    scenarios = build_scenarios(df, scenarios_cfg)
    
    # 为每一天找到其所属的簇
    # 这里需要重新做聚类以获取所有天的标签
    from sklearn.cluster import KMeans
    
    # 构造特征（10维，与Phase 3保持一致）
    features = []
    for day_df in daily_dfs:
        # 均值特征（基础）
        mean_solar = day_df["Solar_W_m2"].mean()
        mean_wind = day_df["Wind_Speed_m_s"].mean()
        mean_load = day_df["Load_kW"].mean()
        mean_price = day_df["Price_CNY_kWh"].mean()
        
        # 标准差特征（波动性）
        std_solar = day_df["Solar_W_m2"].std()
        std_wind = day_df["Wind_Speed_m_s"].std()
        std_load = day_df["Load_kW"].std()
        
        # 峰值特征（极端情况）
        max_solar = day_df["Solar_W_m2"].max()
        max_wind = day_df["Wind_Speed_m_s"].max()
        max_load = day_df["Load_kW"].max()
        
        features.append([
            mean_solar, mean_wind, mean_load,      # 均值 (1-3)
            std_solar, std_wind, std_load,          # 标准差 (4-6)
            max_solar, max_wind, max_load,          # 峰值 (7-9)
            mean_price                               # 电价 (10)
        ])
    
    features_array = np.array(features)
    kmeans = KMeans(n_clusters=k_typical, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(features_array)
    
    # 为原始DataFrame添加标签
    day_labels = []
    day_indices = []
    for day_idx, (day_df, label) in enumerate(zip(daily_dfs, labels)):
        day_labels.extend([label] * len(day_df))
        day_indices.extend([day_idx] * len(day_df))
    
    df["Day_Index"] = day_indices
    df["Day_Label"] = day_labels
    
    # 统计各类别天数
    label_counts = pd.Series(labels).value_counts().sort_index()
    print(f"✓ 聚类完成，各类别天数分布：")
    for label, count in label_counts.items():
        print(f"  类别 {label}: {count} 天 ({count/len(daily_dfs)*100:.1f}%)")
    
    return df


def add_hour_column(df: pd.DataFrame):
    """添加小时列（0-23）"""
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df["Hour"] = df["Time"].dt.hour
    else:
        # 如果没有Time列，根据顺序生成
        df["Hour"] = df.index % 24
    return df


def main():
    """主流程"""
    print("=" * 60)
    print("第二阶段数据预处理脚本")
    print("=" * 60)
    
    # 1. 加载第一阶段容量配置
    print("\n[步骤 1/5] 加载第一阶段容量配置")
    capacity = load_capacity_from_phase1()
    
    # 2. 读取原始数据
    print("\n[步骤 2/5] 读取原始数据")
    data_path = Path(__file__).parent.parent / "data" / "data2023.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ 读取数据: {len(df)} 条记录")
    print(f"  列名: {df.columns.tolist()}")
    
    # 筛选2023年的数据（如果数据超过一年）
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        # 只保留2023年的数据
        df = df[df["Time"].dt.year == 2023].copy()
        print(f"✓ 筛选2023年数据: {len(df)} 条记录 ({len(df)//24} 天)")
    
    # 如果数据量不是一年的，截取前8760条（365天）
    if len(df) > 8760:
        print(f"  警告：数据超过一年，截取前8760条（365天）")
        df = df.iloc[:8760].copy()
    elif len(df) < 8760:
        raise ValueError(f"数据不足一年：只有 {len(df)} 条记录（需要至少8760条）")
    
    # 3. 计算风光出力
    print("\n[步骤 3/5] 计算风光出力")
    wind, pv = build_renewable_models(capacity)
    df = compute_renewable_output(df, wind, pv)
    
    # 4. 天气聚类
    print("\n[步骤 4/5] 天气聚类")
    df = cluster_days_by_weather(df, k_typical=5)
    
    # 5. 添加辅助列
    print("\n[步骤 5/5] 添加辅助列")
    df = add_hour_column(df)
    
    # 转换负荷单位（kW -> MW）
    if "Load_kW" in df.columns:
        df["Load_MW"] = df["Load_kW"] / 1000.0
    
    # 6. 保存结果
    output_path = Path(__file__).parent / "clustered_training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ 数据保存成功: {output_path}")
    print(f"  注意：如果文件夹已重命名为Phase2-SAC，请手动移动此文件")
    
    # 7. 输出统计信息
    print("\n" + "=" * 60)
    print("数据统计信息")
    print("=" * 60)
    print(f"总记录数: {len(df)}")
    print(f"总天数: {df['Day_Index'].max() + 1}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n数据预览:")
    print(df.head(10))
    
    print("\n✓ 预处理完成！可以开始训练第二阶段的专家策略。")
    
    # 保存容量配置到phase2_config.yaml
    phase2_config_path = Path(__file__).parent / "phase2_config.yaml"
    if phase2_config_path.exists():
        with open(phase2_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    if "capacity" not in config:
        config["capacity"] = {}
    config["capacity"].update(capacity)
    
    with open(phase2_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"✓ 容量配置已更新到: {phase2_config_path}")


if __name__ == "__main__":
    main()
