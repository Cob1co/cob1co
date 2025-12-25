"""场景聚合模块：基于逐时数据构造典型日场景。

说明：
- 输入：包含 Time,Temperature_C,Solar_W_m2,Wind_Speed_m_s,Load_kW,Price_CNY_kWh 的全年逐时数据。
- 输出：若干典型日及其在全年中的权重，用于第一阶段容量配置评估。

注意：这里只做聚类与权重计算，不涉及经济性。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.cluster import KMeans


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """确保 Time 列为 pandas 的 datetime 类型。"""
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df = df.copy()
        df["Time"] = pd.to_datetime(df["Time"])
    return df


def split_days(df: pd.DataFrame) -> List[pd.DataFrame]:
    """按自然日切分数据，返回每天的数据切片列表。"""
    df = _ensure_datetime(df)
    df["Date"] = df["Time"].dt.date
    days: List[pd.DataFrame] = []
    for _, g in df.groupby("Date"):
        days.append(g.reset_index(drop=True))
    return days


def _build_daily_features(day_df: pd.DataFrame) -> List[float]:
    """为单个日数据构造聚类特征向量。

    当前特征：日平均光照、日平均风速、日平均温度、峰谷电价差。
    """
    ghi_mean = float(day_df["Solar_W_m2"].mean())
    wind_mean = float(day_df["Wind_Speed_m_s"].mean())
    temp_mean = float(day_df["Temperature_C"].mean())

    price = day_df["Price_CNY_kWh"]
    if len(price) > 0:
        price_peak = float(price.max())
        price_valley = float(price.min())
        price_diff = price_peak - price_valley
    else:
        price_diff = 0.0

    return [ghi_mean, wind_mean, temp_mean, price_diff]


def build_scenarios(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[Tuple[pd.DataFrame, float]]:
    """基于全年数据构造典型日场景。

    参数:
    - df: 全年逐时数据 DataFrame。
    - cfg: 配置字典，至少包含 k_typical, random_state 两个键。

    返回:
    - 列表 [(day_df, weight), ...]，其中 weight 为该典型日代表的年度天数占比。
    """
    days = split_days(df)
    if not days:
        return []

    k = int(cfg.get("k_typical", 5))
    random_state = int(cfg.get("random_state", 0))

    # 构造所有日的特征矩阵
    features = [
        _build_daily_features(day_df) for day_df in days
    ]
    feat_df = pd.DataFrame(features, columns=["ghi", "wind", "temp", "price_diff"])

    # KMeans 聚类
    if k > len(days):
        k = len(days)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(feat_df)

    # 为每个簇选择一个代表日（离质心最近）并计算权重
    scenarios: List[Tuple[pd.DataFrame, float]] = []
    total_days = len(days)
    for cluster_id in range(k):
        idx_in_cluster = feat_df.index[labels == cluster_id].tolist()
        if not idx_in_cluster:
            continue
        # 计算与质心的欧氏距离，选最近的那一天
        center = km.cluster_centers_[cluster_id]
        best_idx = idx_in_cluster[0]
        best_dist = float("inf")
        for idx in idx_in_cluster:
            vec = feat_df.loc[idx].values
            dist = float(((vec - center) ** 2).sum())
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        rep_day_df = days[best_idx]
        weight = len(idx_in_cluster) / float(total_days)
        scenarios.append((rep_day_df, weight))

    return scenarios
