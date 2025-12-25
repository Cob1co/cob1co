"""Phase 3 特征提取器

负责从历史状态序列 + 专家参考计划 + 预测误差中构造 12 维特征，
用于 Transformer 权重控制器的输入。

特征设计参考文档：
1. 系统状态 (4)
   - soc
   - grid_power (归一化)
   - soc_deviation
   - grid_deviation (归一化)
2. 预测质量 (3)
   - load_error
   - pv_error
   - wind_error
3. 时间特征 (2)
   - hour
   - is_peak
4. 专家信息 (3)
   - expert_id
   - expert_confidence
   - time_since_switch
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class FeatureExtractor:
    """特征提取器

    说明：
    - 只做非常轻量的数值处理，不涉及任何 MPC 或专家逻辑；
    - 预测误差已经由外部模块计算好，这里直接使用；
    - 归一化尺度尽量从配置中读取，避免魔法数字。
    """

    def __init__(self, config: Dict[str, Any]):
        transformer_cfg = config.get("transformer", {})
        train_cfg = transformer_cfg.get("training", {})
        feat_cfg = train_cfg.get("features", {})

        # 预测误差窗口/历史长度配置目前只在 DataCollector 中使用，这里做记录
        self.window = int(feat_cfg.get("forecast_error_window", 4))
        self.history_len = int(feat_cfg.get("history_length", 24))

        # 电网功率归一化尺度：用装机容量近似
        cap_cfg = config.get("capacity", {})
        wind_mw = float(cap_cfg.get("wind_mw", 20.0))
        pv_mw = float(cap_cfg.get("pv_mw", 35.0))
        st_mw = float(cap_cfg.get("st_mw_e", 15.0))
        # 最多风+光+汽轮机同时输出，作为一个合理上界
        self.max_grid = max(1.0, wind_mw + pv_mw + st_mw)

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def extract_features(
        self,
        history_states: List[Dict[str, Any]],
        expert_plan: Dict[str, np.ndarray],
        forecast_errors: Dict[str, float],
        expert_id: int,
        expert_switch_time: int,
    ) -> np.ndarray:
        """提取单步 12 维特征。

        参数：
            history_states: 历史状态列表，每个元素至少包含
                - 'soc': float
                - 'grid_power': float (MW)
                - 'time': pandas.Timestamp 或 datetime 或字符串
            expert_plan: 当前 8 小时专家参考计划 {'soc': (T,), 'grid_power': (T,)}
            forecast_errors: {'load': float, 'pv': float, 'wind': float}
            expert_id: 当前专家编号 0~4
            expert_switch_time: 距离上一次专家切换经过的小时数
        """
        if not history_states:
            # 没有历史数据时，返回全零向量，避免崩溃
            return np.zeros(12, dtype=np.float32)

        # 当前时刻状态：取最后一个
        current = history_states[-1]
        soc = float(current.get("soc", 0.5))
        grid_power = float(current.get("grid_power", 0.0))

        # 专家参考的当前值
        if expert_plan and "soc" in expert_plan and len(expert_plan["soc"]) > 0:
            soc_ref = float(expert_plan["soc"][0])
        else:
            soc_ref = soc

        if expert_plan and "grid_power" in expert_plan and len(expert_plan["grid_power"]) > 0:
            grid_ref = float(expert_plan["grid_power"][0])
        else:
            grid_ref = grid_power

        # 电网功率归一化
        grid_norm = grid_power / (self.max_grid + 1e-6)
        grid_dev_norm = abs(grid_power - grid_ref) / (self.max_grid + 1e-6)

        # 预测误差（由外部给出）
        load_err = float(forecast_errors.get("load", 0.0))
        pv_err = float(forecast_errors.get("pv", 0.0))
        wind_err = float(forecast_errors.get("wind", 0.0))

        # 时间特征
        t = current.get("time", None)
        hour = 0
        if t is not None:
            if hasattr(t, "hour"):
                hour = int(t.hour)
            else:
                # 兼容字符串等类型
                try:
                    dt = pd.to_datetime(t)
                    hour = int(dt.hour)
                except Exception:
                    hour = 0
        hour_norm = hour / 24.0
        is_peak = 1.0 if (8 <= hour < 11 or 17 <= hour < 21) else 0.0

        # 专家信息
        expert_id_norm = float(expert_id) / 4.0  # 0~1 之间
        expert_conf = 1.0  # 暂时统一认为专家置信度为 1，将来可接入更复杂逻辑
        time_since_switch = float(expert_switch_time) / 24.0

        feat = np.array(
            [
                soc,                 # 0
                grid_norm,           # 1
                abs(soc - soc_ref),  # 2
                grid_dev_norm,       # 3
                load_err,            # 4
                pv_err,              # 5
                wind_err,            # 6
                hour_norm,           # 7
                is_peak,             # 8
                expert_id_norm,      # 9
                expert_conf,         # 10
                time_since_switch,   # 11
            ],
            dtype=np.float32,
        )
        return feat
