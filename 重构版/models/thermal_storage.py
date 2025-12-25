"""熔盐储热单元模型（总能量视角）。"""

from typing import Any, Dict

from .base import Component


class ThermalStorage(Component):
    """总能量视角的简化熔盐储热模型。

    只关心总热容量、SOC 上下限和热损。
    """

    def __init__(
        self,
        energy_cap_mwh: float,
        loss_rate_per_h: float = 0.0,
        soc_min_frac: float = 0.0,
        soc_max_frac: float = 1.0,
    ) -> None:
        self.e_cap = float(energy_cap_mwh)
        self.loss = float(loss_rate_per_h)
        self.soc_min = max(0.0, min(1.0, float(soc_min_frac)))
        self.soc_max = max(0.0, min(1.0, float(soc_max_frac)))
        # 当前能量（MWh_th）
        self.e = self.e_cap * self.soc_min

    def reset(self, soc_frac: float = None, **kwargs: Any) -> None:
        """重置 SOC。"""
        if soc_frac is None:
            self.e = self.e_cap * self.soc_min
        else:
            s = max(self.soc_min, min(self.soc_max, float(soc_frac)))
            self.e = self.e_cap * s

    def step(self, inputs: Dict[str, Any], dt_hours: float = 1.0) -> Dict[str, Any]:
        """根据充放热功率更新 SOC。"""
        charge_th = float(inputs.get("charge_th_mw", 0.0))
        discharge_th = float(inputs.get("discharge_th_mw", 0.0))

        charge_th = max(0.0, charge_th)
        discharge_th = max(0.0, discharge_th)

        # 计算容量约束
        e_free = self.e_cap * self.soc_max - self.e
        e_need = self.e - self.e_cap * self.soc_min
        max_charge = e_free / dt_hours if dt_hours > 0.0 else 0.0
        max_discharge = e_need / dt_hours if dt_hours > 0.0 else 0.0

        acc_charge = min(charge_th, max_charge)
        acc_discharge = min(discharge_th, max_discharge)

        # 更新能量
        self.e = self.e + (acc_charge - acc_discharge) * dt_hours

        # 热损耗
        if self.loss > 0.0 and dt_hours > 0.0:
            loss_e = self.e * self.loss * dt_hours
            self.e = max(0.0, self.e - loss_e)

        soc = 0.0 if self.e_cap <= 0.0 else self.e / self.e_cap

        return {
            "charge_accepted_mw": acc_charge,
            "discharge_accepted_mw": acc_discharge,
            "e_mwh": self.e,
            "soc": soc,
        }

    @classmethod
    def from_config(cls, assets_cfg: Dict[str, Any], physics_cfg: Dict[str, Any]):
        """从配置字典构造储热模型。

        assets_cfg: models.config.yaml 中 assets.thermal_storage。
        physics_cfg: models.config.yaml 中 physics.molten_salt。
        """
        e_cap = float(assets_cfg.get("total_energy_cap_mwh", 0.0))
        loss = float(assets_cfg.get("loss_rate_per_h", physics_cfg.get("loss_rate_per_h", 0.0)))
        soc_min = float(physics_cfg.get("soc_min", 0.0))
        soc_max = float(physics_cfg.get("soc_max", 1.0))
        return cls(energy_cap_mwh=e_cap, loss_rate_per_h=loss, soc_min_frac=soc_min, soc_max_frac=soc_max)
