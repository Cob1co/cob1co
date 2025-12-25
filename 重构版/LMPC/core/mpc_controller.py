"""Phase 3 MPC 控制器

基于 CVXPY 的简化 MPC：
- 单一储能功率变量 P_s (MW)：正为充电，负为放电；
- 线性储能动态：soc[k+1] = soc[k] + eta_storage * P_s[k] * dt / E_cap；
- 电网功率平衡：P_grid = Load - PV - Wind + P_s + P_curt；
- 通过 P_grid = P_export - P_import 分解导入/导出，
  只对 P_import 计入电价成本，保持凸性。

目标函数（对齐需求文档思想）：
J = Σ_k [
  w_soc  * ||soc - soc_ref||^2
+ w_grid * ||P_grid - grid_ref||^2
+ w_cost * price * P_import * dt
+ w_ramp * ||P_grid[k+1] - P_grid[k]||^2
+ w_curt * P_curt
]

其中 w_soc, w_grid, w_cost = alpha_* × base_weights.*。
"""

from __future__ import annotations

from typing import Any, Dict

import cvxpy as cp
import numpy as np
import warnings


class MPCController:
    """MPC 控制器

    只依赖配置和预测/参考计划，不直接关联外部环境类。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        mpc_cfg = config.get("mpc", {})
        dt_min = float(mpc_cfg.get("time_step_minutes", 15))
        self.dt_hours = dt_min / 60.0
        self.prediction_horizon = int(mpc_cfg.get("prediction_horizon", 16))
        self.control_horizon = int(mpc_cfg.get("control_horizon", 4))

        base_w = mpc_cfg.get("base_weights", {})
        self.w_soc_base = float(base_w.get("soc", 10.0))
        self.w_grid_base = float(base_w.get("grid", 5.0))

        # 三目标基础权重与缩放（成本 / 波动 / 弃电），与 Phase2 SAC / Oracle 保持一致
        obj_cfg = config.get("objective", {})
        self.w_cost_base = float(obj_cfg.get("w_cost", 0.8))
        self.w_ramp_base = float(obj_cfg.get("w_ramp", 1.0))
        self.w_curt_base = float(obj_cfg.get("w_curt", 0.6))
        self.cost_scale = float(obj_cfg.get("cost_scale", 10000.0))
        self.ramp_scale = float(obj_cfg.get("ramp_scale", 50.0))
        self.curtail_scale = float(obj_cfg.get("curtail_scale", 50.0))

        # 容量与 SOC 约束
        cap_cfg = config.get("capacity", {})
        self.E_cap = float(cap_cfg.get("ts_mwh", 200.0))
        self.P_max = float(cap_cfg.get("st_mw_e", 15.0))  # 储能功率上限（近似用汽轮机容量）

        wind_mw = float(cap_cfg.get("wind_mw", 20.0))
        pv_mw = float(cap_cfg.get("pv_mw", 35.0))
        # 电网功率上限：风+光+储能，近似一个合理范围
        self.grid_limit = wind_mw + pv_mw + self.P_max

        phy_cfg = config.get("physics", {})
        self.soc_min = float(phy_cfg.get("soc_min", 0.1))
        self.soc_max = float(phy_cfg.get("soc_max", 0.9))
        # 储能效率（如未配置，则视为 1.0）
        self.eta_storage = float(phy_cfg.get("eta_storage", 1.0))

        alpha_cfg = mpc_cfg.get("alpha_range", {})
        self.alpha_min = float(alpha_cfg.get("min", 0.5))
        self.alpha_max = float(alpha_cfg.get("max", 2.0))

        solver_cfg = mpc_cfg.get("solver", {})
        self.solver_name = solver_cfg.get("name", "OSQP")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _clip_alpha(self, alpha: float) -> float:
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    # ------------------------------------------------------------------
    # 主求解接口
    # ------------------------------------------------------------------
    def solve(
        self,
        current_state: Dict[str, float],
        forecast: Dict[str, np.ndarray],
        reference_plan: Dict[str, np.ndarray],
        dynamic_weights: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """求解 MPC。

        参数：
            current_state: {'soc': float, 'grid_power': float}
            forecast: {'load','pv','wind','price'}，单位均为 MW（电价元/kWh）
            reference_plan: {'soc','grid_power'}，长度 >= H
            dynamic_weights: {'alpha_soc','alpha_grid','alpha_cost'} 或 None
        返回：
            {
              'status': str,
              'soc_plan': np.array(H+1,),
              'grid_plan': np.array(H,),
              'P_storage': float,
              'P_grid': float,
              'cost': float,
            }
        """
        load = np.asarray(forecast["load"], dtype=float)
        pv = np.asarray(forecast["pv"], dtype=float)
        wind = np.asarray(forecast["wind"], dtype=float)
        price = np.asarray(forecast["price"], dtype=float)
        H = len(load)

        # 参考轨迹截断到 H
        soc_ref = np.asarray(reference_plan["soc"][:H], dtype=float)
        grid_ref = np.asarray(reference_plan["grid_power"][:H], dtype=float)

        soc0 = float(current_state.get("soc", 0.5))
        grid_prev = float(current_state.get("grid_power", 0.0))  # 仅用于调试/返回，不直接入模

        # 动态权重：约定
        # - alpha_cost 控制成本项权重
        # - alpha_grid 控制波动项权重
        # - alpha_soc  控制弃电项权重
        if dynamic_weights is None:
            alpha_cost = alpha_ramp = alpha_curt = 1.0
        else:
            alpha_curt = float(dynamic_weights.get("alpha_soc", 1.0))
            alpha_ramp = float(dynamic_weights.get("alpha_grid", 1.0))
            alpha_cost = float(dynamic_weights.get("alpha_cost", 1.0))

        alpha_cost = self._clip_alpha(alpha_cost)
        alpha_ramp = self._clip_alpha(alpha_ramp)
        alpha_curt = self._clip_alpha(alpha_curt)

        w_cost = self.w_cost_base * alpha_cost
        w_ramp = self.w_ramp_base * alpha_ramp
        w_curt = self.w_curt_base * alpha_curt

        dt = self.dt_hours

        # 变量
        soc = cp.Variable(H + 1)
        P_s = cp.Variable(H)          # 储能功率 (MW)
        P_grid = cp.Variable(H)       # 电网功率 (MW)
        P_curt = cp.Variable(H)       # 弃电功率 (MW)
        P_import = cp.Variable(H)     # 从电网购电 (MW)
        P_export = cp.Variable(H)     # 向电网售电 (MW)

        obj = 0
        cons = []

        # 初始与边界约束
        cons += [soc[0] == soc0]
        cons += [soc >= self.soc_min, soc <= self.soc_max]
        cons += [P_s >= -self.P_max, P_s <= self.P_max]
        cons += [P_curt >= 0]
        cons += [P_import >= 0, P_export >= 0]
        cons += [P_grid >= -self.grid_limit, P_grid <= self.grid_limit]

        for k in range(H):
            # 储能动态
            cons.append(
                soc[k + 1] == soc[k] + self.eta_storage * P_s[k] * dt / self.E_cap
            )

            # 功率平衡
            cons.append(
                P_grid[k] == load[k] - pv[k] - wind[k] + P_s[k] + P_curt[k]
            )

            # 电网导入/导出分解
            cons.append(P_grid[k] == P_export[k] - P_import[k])

            # 目标函数各项（三目标：购电成本 / 电网功率波动 / 弃电）
            # 1) 经济成本：仅计购电成本
            obj += w_cost * price[k] * P_import[k] * 1000.0 * dt

            # 2) 电网功率波动：相邻时间步电网功率变化的绝对值
            if k < H - 1:
                obj += w_ramp * self.ramp_scale * cp.abs(P_grid[k + 1] - P_grid[k])

            # 3) 弃风弃光功率
            obj += w_curt * self.curtail_scale * P_curt[k]

        prob = cp.Problem(cp.Minimize(obj), cons)

        try:
            solver = getattr(cp, self.solver_name, cp.OSQP)
            # 局部屏蔽 cvxpy 关于解精度的 UserWarning，避免日志刷屏
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Solution may be inaccurate.*",
                    category=UserWarning,
                )
                prob.solve(solver=solver, verbose=False)
        except Exception as e:  # noqa: F841
            return {
                "status": "error",
                "soc_plan": np.array([soc0], dtype=float),
                "grid_plan": np.array([grid_prev], dtype=float),
                "P_storage": 0.0,
                "P_grid": grid_prev,
                "cost": float("inf"),
            }

        status = prob.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {
                "status": str(status),
                "soc_plan": np.array([soc0], dtype=float),
                "grid_plan": np.array([grid_prev], dtype=float),
                "P_storage": 0.0,
                "P_grid": grid_prev,
                "cost": float("inf"),
            }

        soc_val = np.array(soc.value, dtype=float)
        grid_val = np.array(P_grid.value, dtype=float)
        P_s_val = np.array(P_s.value, dtype=float)
        P_curt_val = np.array(P_curt.value, dtype=float)
        P_import_val = np.array(P_import.value, dtype=float)
        P_export_val = np.array(P_export.value, dtype=float)

        first_P_s = float(P_s_val[0]) if H > 0 and P_s.value is not None else 0.0
        first_P_grid = float(grid_val[0]) if H > 0 else grid_prev

        total_cost = float(obj.value) if obj.value is not None else 0.0

        return {
            "status": "optimal",
            "soc_plan": soc_val,
            "grid_plan": grid_val,
            "P_storage": first_P_s,
            "P_grid": first_P_grid,
            "cost": total_cost,
            # 供 Oracle 标签评估使用的完整轨迹
            "P_storage_plan": P_s_val,
            "P_curtail_plan": P_curt_val,
            "P_import_plan": P_import_val,
            "P_export_plan": P_export_val,
        }
