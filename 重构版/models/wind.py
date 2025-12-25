"""风电场模型。"""

from typing import Any, Dict, List, Tuple

from .base import Component


class WindFarm(Component):
    """简单的风电场模型。

    使用离散功率曲线 (风速, 归一化功率) 做线性插值。
    """

    def __init__(self, capacity_mw: float, power_curve: List[Tuple[float, float]]):
        # 额定装机容量（MW）
        self.capacity_mw = float(capacity_mw)
        # 按风速升序排序的功率曲线
        self.curve = sorted(power_curve, key=lambda x: x[0])

    def reset(self, **kwargs: Any) -> None:
        # 当前模型无内部状态，重置时不做处理
        return None

    def _norm_power(self, wind_speed: float) -> float:
        """根据风速计算归一化功率 (0~1)。"""
        if not self.curve:
            return 0.0
        # 低于最小风速
        if wind_speed <= self.curve[0][0]:
            return max(0.0, min(1.0, float(self.curve[0][1])))
        # 区间线性插值
        for i in range(1, len(self.curve)):
            v0, p0 = self.curve[i - 1]
            v1, p1 = self.curve[i]
            if wind_speed <= v1:
                if v1 == v0:
                    return max(0.0, min(1.0, float(p1)))
                r = (wind_speed - v0) / (v1 - v0)
                p = p0 + r * (p1 - p0)
                return max(0.0, min(1.0, float(p)))
        # 高于最大风速
        return max(0.0, min(1.0, float(self.curve[-1][1])))

    def step(self, inputs: Dict[str, Any], dt_hours: float = 1.0) -> Dict[str, Any]:
        """根据当前风速计算风电出力。"""
        ws = float(inputs.get("wind_speed", 0.0))
        p_norm = self._norm_power(ws)
        p_mw = self.capacity_mw * p_norm
        return {"p_mw": p_mw, "p_norm": p_norm}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], power_curve: List[Tuple[float, float]]):
        """从配置字典构造风电场。

        cfg 通常来源于 models.config.yaml 中 assets.wind。
        power_curve 建议由上层代码根据 cfg["power_curve_csv"] 读取。
        """
        cap = float(cfg.get("capacity_mw", 0.0))
        return cls(capacity_mw=cap, power_curve=power_curve)
