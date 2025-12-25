"""光伏电站模型。"""

from typing import Any, Dict

from .base import Component


class PVPlant(Component):
    """简单的光伏模型。

    使用 GHI 线性比例 + 温度线性折减。
    """

    def __init__(
        self,
        capacity_mw: float,
        ghi_ref: float = 1000.0,
        temp_coeff: float = -0.004,
        t_ref_c: float = 25.0,
    ) -> None:
        self.capacity_mw = float(capacity_mw)
        self.ghi_ref = float(ghi_ref)
        self.temp_coeff = float(temp_coeff)
        self.t_ref = float(t_ref_c)

    def reset(self, **kwargs: Any) -> None:
        # 当前模型无内部状态
        return None

    def step(self, inputs: Dict[str, Any], dt_hours: float = 1.0) -> Dict[str, Any]:
        """根据 GHI 和环境温度计算光伏出力。"""
        ghi = float(inputs.get("ghi", 0.0))
        temp = float(inputs.get("temp", self.t_ref))

        ratio = 0.0
        if self.ghi_ref > 0.0:
            ratio = max(0.0, ghi / self.ghi_ref)

        # 温度折减因子
        derate = 1.0 + self.temp_coeff * (temp - self.t_ref)
        derate = max(0.0, derate)

        p_mw = self.capacity_mw * ratio * derate
        # 限制不超过装机容量
        p_mw = min(p_mw, self.capacity_mw)

        return {"p_mw": p_mw, "ratio": ratio, "derate": derate}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        """从配置字典构造光伏电站。"""
        cap = float(cfg.get("capacity_mw", 0.0))
        ghi_ref = float(cfg.get("ghi_ref_w_per_m2", 1000.0))
        temp_coeff = float(cfg.get("temp_coeff_per_c", -0.004))
        t_ref = float(cfg.get("t_ref_c", 25.0))
        return cls(capacity_mw=cap, ghi_ref=ghi_ref, temp_coeff=temp_coeff, t_ref_c=t_ref)
