"""汽轮发电机组模型（热-电）。"""

from typing import Any, Dict, Optional

from .base import Component


class SteamTurbine(Component):
    """简化汽轮机模型。

    只考虑热-电效率和爬坡约束，可选考虑最小技术出力。
    """

    def __init__(
        self,
        cap_mw_e: float,
        eta: float = 0.38,
        ramp_mw_per_h: Optional[float] = None,
        p_min_frac: float = 0.0,
    ) -> None:
        # 额定电功率（MW_e）
        self.cap_e = float(cap_mw_e)
        # 热-电效率
        self.eta = float(eta)
        # 爬坡率（MW/小时），None 表示不限制
        self.ramp = float(ramp_mw_per_h) if ramp_mw_per_h is not None else None
        # 最小技术出力占比（0~1）
        self.p_min_frac = max(0.0, min(1.0, float(p_min_frac)))
        # 上一时刻电功率（MW），用于爬坡约束
        self.p_last = 0.0

    def reset(self, p0_mw: float = 0.0, **kwargs: Any) -> None:
        """重置汽轮机初始出力。"""
        self.p_last = max(0.0, min(self.cap_e, float(p0_mw)))

    def step(self, inputs: Dict[str, Any], dt_hours: float = 1.0) -> Dict[str, Any]:
        """根据输入热功率计算电功率输出。"""
        p_th_in = float(inputs.get("p_th_in_mw", 0.0))

        # 理想电功率
        p_e_ideal = p_th_in * self.eta
        # 限制在[0, cap]
        p_e = max(0.0, min(self.cap_e, p_e_ideal))

        # 最小技术出力
        p_min = self.cap_e * self.p_min_frac
        if 0.0 < p_e < p_min:
            # 如果需要，可以选择关机(置0) 或 提升到最小出力，这里简单提升
            p_e = p_min

        # 爬坡约束
        if self.ramp is not None and dt_hours > 0.0:
            up = self.p_last + self.ramp * dt_hours
            down = max(0.0, self.p_last - self.ramp * dt_hours)
            if p_e > up:
                p_e = up
            if p_e < down:
                p_e = down

        # 对应的热功率使用量
        th_used = p_e / self.eta if self.eta > 0.0 else 0.0

        self.p_last = p_e

        return {"p_e_out_mw": p_e, "p_th_used_mw": th_used}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        """从配置字典构造汽轮机。"""
        cap = float(cfg.get("cap_mw_e", 0.0))
        eta = float(cfg.get("eta", 0.38))
        ramp = cfg.get("ramp_mw_per_h")
        p_min_frac = float(cfg.get("p_min_frac", 0.0))
        return cls(cap_mw_e=cap, eta=eta, ramp_mw_per_h=ramp, p_min_frac=p_min_frac)
