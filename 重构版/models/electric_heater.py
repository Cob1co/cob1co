"""电加热器模型（电-热）。"""

from typing import Any, Dict

from .base import Component


class ElectricHeater(Component):
    """电加热器模型：电功率转化为热功率。"""

    def __init__(self, cap_mw_th: float, eta: float = 0.98) -> None:
        # 额定热功率（MW_th）
        self.cap_th = float(cap_mw_th)
        # 电-热效率
        self.eta = float(eta)

    def reset(self, **kwargs: Any) -> None:
        # 无内部状态
        return None

    def step(self, inputs: Dict[str, Any], dt_hours: float = 1.0) -> Dict[str, Any]:
        """根据输入电功率计算输出热功率。"""
        p_elec_req = float(inputs.get("p_elec_in_mw", 0.0))

        # 电侧可用功率上限（考虑效率）
        p_elec_cap = self.cap_th / self.eta if self.eta > 0.0 else 0.0
        p_elec_used = max(0.0, min(p_elec_req, p_elec_cap))

        # 热侧输出
        p_th_out = min(self.cap_th, p_elec_used * self.eta)

        return {"p_elec_used_mw": p_elec_used, "p_th_out_mw": p_th_out}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        """从配置字典构造电加热器。"""
        cap = float(cfg.get("cap_mw_th", 0.0))
        eta = float(cfg.get("eta", 0.98))
        return cls(cap_mw_th=cap, eta=eta)
