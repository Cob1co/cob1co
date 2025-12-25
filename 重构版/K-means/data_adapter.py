"""场景聚类适配器。

功能：
1. 读取 data/data2023.csv 全年数据。
2. 对 365 天的气象与负荷特征进行 K-Means 聚类 (k=5)。
3. 输出带标签的数据文件 clustered_data.csv，供 PPO 训练使用。
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

# 添加项目根目录到 path 以便导入 SA-MBPS 下的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 复用 SA-MBPS 中的特征提取逻辑
try:
    from SA_MBPS.scenario_manager import split_days
except ImportError:
    # 如果导入失败，提供简单的回退实现
    def split_days(df):
        if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
            df["Time"] = pd.to_datetime(df["Time"])
        df["Date"] = df["Time"].dt.date
        return [g.reset_index(drop=True) for _, g in df.groupby("Date")]

def _extract_daily_features(day_df: pd.DataFrame) -> np.ndarray:
    """
    提取单日特征向量（改进版）
    
    特征组成（10维）：
    1-3. 均值：平均光照, 平均风速, 平均负荷
    4-6. 标准差：光照波动, 风速波动, 负荷波动
    7-9. 峰值：最大光照, 最大风速, 最大负荷
    10. 平均电价
    """
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
    
    return np.array([
        mean_solar, mean_wind, mean_load,      # 均值 (1-3)
        std_solar, std_wind, std_load,          # 标准差 (4-6)
        max_solar, max_wind, max_load,          # 峰值 (7-9)
        mean_price                               # 电价 (10)
    ])

def run_clustering(data_path: str, k: int = 5, output_file: str = "clustered_data.csv"):
    print(f"[K-Means] 读取数据: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 1. 切分天
    days = split_days(df)
    print(f"[K-Means] 全年共 {len(days)} 天。")
    
    # 2. 提取特征
    features = []
    valid_indices = []
    
    for i, day_df in enumerate(days):
        if len(day_df) == 24: # 确保一天是完整的24小时
            feat = _extract_daily_features(day_df)
            features.append(feat)
            valid_indices.append(i)
            
    X = np.array(features)
    
    # 3. 聚类
    print(f"[K-Means] 开始聚类 (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # 4. 整理输出
    print("[K-Means] 正在生成带标签数据...")
    output_list = []
    for idx, label in zip(valid_indices, labels):
        day_df = days[idx].copy()
        day_df["Day_Label"] = label  # 0~4
        day_df["Day_Index"] = idx
        output_list.append(day_df)
        
    final_df = pd.concat(output_list, ignore_index=True)
    
    # 保存数据
    out_path = os.path.join(os.path.dirname(__file__), output_file)
    final_df.to_csv(out_path, index=False)
    print(f"[K-Means] 成功！带标签数据已保存至: {out_path}")
    
    # ⭐ 保存K-means模型（供Phase 3使用）
    import pickle
    model_path = os.path.join(os.path.dirname(__file__), "kmeans_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"[K-Means] K-means模型已保存至: {model_path}")
    
    # 统计信息
    counts = pd.Series(labels).value_counts().sort_index()
    print("\n=== 各类典型日统计 ===")
    print(counts)
    print("\n=== 聚类中心 (10维特征) ===")
    print("特征: [均值(光/风/负荷), 标准差(光/风/负荷), 峰值(光/风/负荷), 电价]")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"\n专家{i}:")
        print(f"  均值: 光照={center[0]:.1f} 风速={center[1]:.1f} 负荷={center[2]:.1f}")
        print(f"  标准差: 光照={center[3]:.1f} 风速={center[4]:.1f} 负荷={center[5]:.1f}")
        print(f"  峰值: 光照={center[6]:.1f} 风速={center[7]:.1f} 负荷={center[8]:.1f}")
        print(f"  电价: {center[9]:.3f}")
    print("======================")

if __name__ == "__main__":
    # 假设数据在项目根目录的 data/data2023.csv
    raw_data = os.path.join(PROJECT_ROOT, "data", "data2023.csv")
    run_clustering(raw_data)
