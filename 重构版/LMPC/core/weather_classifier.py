"""天气分类器

基于 K-means 聚类结果，对给定一天/一段时间的气象与负荷数据
提取 10 维特征并进行聚类预测，输出专家 ID。

特征设计与 K-means/data_adapter.py 中保持一致：
1-3: 光照/风速/负荷 均值
4-6: 光照/风速/负荷 标准差
7-9: 光照/风速/负荷 峰值
10: 电价均值
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# 项目根目录（重构版/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WeatherClassifier:
    """天气分类器：加载 K-means 模型并提供简单接口。

    说明：
    - 训练阶段：由 K-means/data_adapter.py 生成 kmeans_model.pkl；
    - 运行阶段：本类只负责加载模型并做一次 predict。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._load_kmeans_model()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_kmeans_model(self):
        """按以下优先级加载 K-means 模型：
        1. phase3_config.yaml.models.kmeans_model 指定路径；
        2. 默认：PROJECT_ROOT / "K-means" / "kmeans_model.pkl"。
        """
        model_paths = []

        # 1) 配置文件中的相对路径
        try:
            models_cfg = self.config.get("models", {})
            rel_path = models_cfg.get("kmeans_model")
            if rel_path:
                model_paths.append(PROJECT_ROOT / rel_path)
        except Exception:
            pass

        # 2) 默认路径
        model_paths.append(PROJECT_ROOT / "K-means" / "kmeans_model.pkl")

        for path in model_paths:
            if path is not None and path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)

        raise FileNotFoundError(
            "未找到 K-means 聚类模型 kmeans_model.pkl，请先在 K-means 目录运行 data_adapter.py 生成。"
        )

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mean(series: Optional[pd.Series]) -> float:
        """对均值做安全处理：空列或全 NaN 时返回 0.0。"""
        if series is None:
            return 0.0
        val = series.mean()
        return float(val) if pd.notna(val) else 0.0

    @staticmethod
    def _safe_std(series: Optional[pd.Series]) -> float:
        """对标准差做安全处理：空列或全 NaN 时返回 0.0。"""
        if series is None:
            return 0.0
        val = series.std()
        return float(val) if pd.notna(val) else 0.0

    @staticmethod
    def _safe_max(series: Optional[pd.Series]) -> float:
        """对最大值做安全处理：空列或全 NaN 时返回 0.0。"""
        if series is None:
            return 0.0
        val = series.max()
        return float(val) if pd.notna(val) else 0.0

    def _extract_features_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """从一个时间序列 DataFrame 中提取 10 维日特征。

        注意：这里不强制必须是 24 小时整天，只要列名一致即可。
        """
        solar = df.get("Solar_W_m2")
        wind = df.get("Wind_Speed_m_s")
        load = df.get("Load_kW")
        price = df.get("Price_CNY_kWh")

        mean_solar = self._safe_mean(solar)
        mean_wind = self._safe_mean(wind)
        mean_load = self._safe_mean(load)
        mean_price = self._safe_mean(price)

        std_solar = self._safe_std(solar)
        std_wind = self._safe_std(wind)
        std_load = self._safe_std(load)

        max_solar = self._safe_max(solar)
        max_wind = self._safe_max(wind)
        max_load = self._safe_max(load)

        feat = np.array(
            [
                mean_solar,
                mean_wind,
                mean_load,
                std_solar,
                std_wind,
                std_load,
                max_solar,
                max_wind,
                max_load,
                mean_price,
            ],
            dtype=np.float32,
        )
        return feat

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def classify_from_history(self, history_df: pd.DataFrame) -> int:
        """基于历史数据进行天气分类。

        参数：
            history_df: 包含 Solar_W_m2 / Wind_Speed_m_s / Load_kW / Price_CNY_kWh 列的时间序列。
        返回：
            专家 ID (0~k-1)。
        """
        if history_df is None or history_df.empty:
            # 没有数据时，默认返回 0 号专家
            return 0

        feat = self._extract_features_from_df(history_df)
        # 确保特征 dtype 与模型内部 dtype 一致，避免 sklearn Buffer dtype mismatch
        target_dtype = getattr(self.model, "cluster_centers_", None)
        if target_dtype is not None:
            feat = feat.astype(self.model.cluster_centers_.dtype, copy=False)
        label = int(self.model.predict(feat.reshape(1, -1))[0])
        return label

    def classify_from_forecast(self, forecast_df: pd.DataFrame) -> int:
        """基于未来预测数据进行天气分类。

        这里的实现与 classify_from_history 完全相同，只是语义不同。
        """
        if forecast_df is None or forecast_df.empty:
            return 0

        feat = self._extract_features_from_df(forecast_df)
        target_dtype = getattr(self.model, "cluster_centers_", None)
        if target_dtype is not None:
            feat = feat.astype(self.model.cluster_centers_.dtype, copy=False)
        label = int(self.model.predict(feat.reshape(1, -1))[0])
        return label
