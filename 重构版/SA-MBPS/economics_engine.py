"""经济性与边际效益分析模块。

核心功能：
- 基于场景聚合结果与运行仿真结果，计算年度经济与技术指标；
- 组装用于优化的单目标函数值 J(x)；
- 计算简单的 "边际效益" 向量，用于指导启发式算法搜索方向。

说明：
- 这里只实现一个相对简化但可运行的版本，后续可以根据论文需要细化 MUI 的定义。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys

import math

import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.params import load_model_config

from sim_kernel import simulate_day


def _annualize_capex(capex_sum: float, discount_rate: float, lifetime_years: int) -> float:
    """将一次性投资折算为等额年金。

    使用常见的资本回收系数 (Capital Recovery Factor)。
    """
    r = float(discount_rate)
    n = int(lifetime_years)
    if r <= 0.0 or n <= 0:
        return capex_sum / max(1, n)
    factor = r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)
    return capex_sum * factor


def _compute_capex_opex(capacity: Dict[str, float]) -> Tuple[float, float]:
    """根据容量向量和 models/config.yaml 中的经济参数计算 CAPEX 与 OPEX。"""
    cfg = load_model_config()
    econ = cfg.get("economics", {})
    capex_cfg = econ.get("capex", {})
    opex_rate = econ.get("opex_rate", {})

    # 各设备投资
    i_w = float(capacity.get("wind_mw", 0.0)) * float(capex_cfg.get("wind_cny_per_mw", 0.0))
    i_pv = float(capacity.get("pv_mw", 0.0)) * float(capex_cfg.get("pv_cny_per_mw", 0.0))
    i_ts = float(capacity.get("ts_mwh", 0.0)) * float(
        capex_cfg.get("thermal_storage_energy_cny_per_mwh", 0.0)
    )
    i_eh = float(capacity.get("eh_mw_th", 0.0)) * float(
        capex_cfg.get("electric_heater_cny_per_mw_th", 0.0)
    )
    i_st = float(capacity.get("st_mw_e", 0.0)) * float(
        capex_cfg.get("steam_turbine_cny_per_mw_e", 0.0)
    )

    capex_sum = i_w + i_pv + i_ts + i_eh + i_st

    # 年运维成本按 CAPEX 比例近似
    opex_w = i_w * float(opex_rate.get("wind_frac_per_year", 0.0))
    opex_pv = i_pv * float(opex_rate.get("pv_frac_per_year", 0.0))
    opex_ts = i_ts * float(opex_rate.get("thermal_storage_frac_per_year", 0.0))
    opex_eh = i_eh * float(opex_rate.get("electric_heater_frac_per_year", 0.0))
    opex_st = i_st * float(opex_rate.get("steam_turbine_frac_per_year", 0.0))

    opex_year = opex_w + opex_pv + opex_ts + opex_eh + opex_st

    return capex_sum, opex_year


def evaluate_capacity(
    capacity: Dict[str, float],
    scenarios: List[Tuple[pd.DataFrame, float]],
    phase1_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
    """评估一组容量配置的年度综合表现。

    参数：
    - capacity: 容量向量字典，键包括 wind_mw, pv_mw, ts_mwh, eh_mw_th, st_mw_e。
    - scenarios: 典型日场景列表 [(day_df, weight), ...]。
    - phase1_cfg: 第一阶段配置（来自 phase1_config.yaml）。

    返回：
    - obj_value: 单目标函数 J(x) 值（越小越好）。
    - metrics: 年度汇总指标字典（供分析与约束检查用）。
    - mui: 边际效益近似向量字典，用于指导启发式搜索。
    """
    op_cfg = phase1_cfg.get("operation", {})
    grid_import_limit_mw = float(op_cfg.get("grid_import_limit_mw", 0.0))
    initial_soc = float(op_cfg.get("initial_soc", 0.5))

    # 年度能量与经济量汇总
    e_load = 0.0
    e_served = 0.0
    e_unserved = 0.0
    e_ren = 0.0
    e_curtail = 0.0
    # 并网购电相关指标
    e_grid_import = 0.0
    cost_grid_import = 0.0
    revenue = 0.0

    eh_full_load_hours = 0.0
    st_full_load_hours = 0.0

    for day_df, weight in scenarios:
        day_res = simulate_day(
            day_df=day_df,
            capacity=capacity,
            initial_soc=initial_soc,
            grid_import_limit_mw=grid_import_limit_mw,
            dt_hours=phase1_cfg.get("project", {}).get("time_step_h", 1.0),
        )
        factor = 365.0 * float(weight)

        e_load += day_res["energy_load_mwh"] * factor
        e_served += day_res["energy_served_mwh"] * factor
        e_unserved += day_res["energy_unserved_mwh"] * factor
        e_ren += day_res["energy_ren_gen_mwh"] * factor
        e_curtail += day_res["energy_curtail_mwh"] * factor
        # 购电电量与成本（若为孤岛场景，则这两个值为0）
        e_grid_import += day_res.get("energy_grid_import_mwh", 0.0) * factor
        cost_grid_import += day_res.get("cost_grid_import_cny", 0.0) * factor

        revenue += day_res["revenue_cny"] * factor

        eh_full_load_hours += day_res["eh_full_load_hours"] * factor
        st_full_load_hours += day_res["st_full_load_hours"] * factor

    # CAPEX 与 OPEX
    capex_sum, opex_year = _compute_capex_opex(capacity)

    model_cfg = load_model_config()
    econ = model_cfg.get("economics", {})
    discount_rate = float(econ.get("discount_rate", 0.0))
    lifetime_years = int(econ.get("lifetime_years", 20))

    capex_ann = _annualize_capex(capex_sum, discount_rate, lifetime_years)

    penalties_cfg = phase1_cfg.get("penalties", {})
    pen_unserved = float(penalties_cfg.get("unserved_cny_per_mwh", 0.0))
    pen_curtail = float(penalties_cfg.get("curtail_cny_per_mwh", 0.0))

    cost_penalty = pen_unserved * e_unserved + pen_curtail * e_curtail

    # 年度总成本 = 年化投资 + 年运维 + 购电成本 + 罚则
    total_cost_year = capex_ann + opex_year + cost_grid_import + cost_penalty
    net_profit_year = revenue - total_cost_year

    # 指标归一化
    eps = 1e-6
    load_ratio_unserved = e_unserved / (e_load + eps)
    ren_ratio_curtail = e_curtail / (e_ren + eps)

    econ_term = -net_profit_year / 1e6  # 以百万元为量级归一化
    rel_term = load_ratio_unserved
    curt_term = ren_ratio_curtail

    w_cfg = phase1_cfg.get("objective_weights", {})
    w_econ = float(w_cfg.get("econ", 1.0))
    w_rel = float(w_cfg.get("reliability", 1.0))
    w_curt = float(w_cfg.get("curtailment", 1.0))

    obj_value = w_econ * econ_term + w_rel * rel_term + w_curt * curt_term

    # --- 边际效益向量（简化版） ---
    # 思路：
    # - 缺电比例高 -> wind/pv/st/ts 容量边际效益为正；
    # - 弃风比例高 -> wind/pv/eh/ts 容量边际效益为负或正（用于多吃弃风）；
    # - 设备满负荷小时数多，说明该设备容量偏紧，边际效益偏正。

    # 为了数值稳定，这里用简单的线性组合并通过 tanh 限幅到 [-1, 1]。
    def _limit(v: float) -> float:
        return math.tanh(v)

    mu_common_pos = load_ratio_unserved  # 缺电越多，增加容量越有利
    mu_common_neg = ren_ratio_curtail  # 弃风越多，发电侧容量边际效益越低

    # 风电与光伏：缺电推动扩容，弃风抑制扩容
    mu_wind = _limit(mu_common_pos - mu_common_neg)
    mu_pv = _limit(mu_common_pos - mu_common_neg)

    # 储热容量：同时缓解弃风与缺电
    mu_ts = _limit(0.5 * mu_common_pos + 0.5 * mu_common_neg)

    # 电加热器：主要用于多吃弃风
    eh_full_ratio = eh_full_load_hours / (8760.0 + eps)
    mu_eh = _limit(mu_common_neg + eh_full_ratio)

    # 汽轮机：主要用于缓解缺电
    st_full_ratio = st_full_load_hours / (8760.0 + eps)
    mu_st = _limit(mu_common_pos + st_full_ratio)

    mui: Dict[str, float] = {
        "wind_mw": mu_wind,
        "pv_mw": mu_pv,
        "ts_mwh": mu_ts,
        "eh_mw_th": mu_eh,
        "st_mw_e": mu_st,
    }

    metrics: Dict[str, Any] = {
        "e_load_mwh": e_load,
        "e_served_mwh": e_served,
        "e_unserved_mwh": e_unserved,
        "e_ren_mwh": e_ren,
        "e_curtail_mwh": e_curtail,
        "revenue_cny": revenue,
        "capex_sum_cny": capex_sum,
        "capex_ann_cny": capex_ann,
        "opex_year_cny": opex_year,
        "e_grid_import_mwh": e_grid_import,
        "cost_grid_import_cny": cost_grid_import,
        "penalty_cny": cost_penalty,
        "total_cost_year_cny": total_cost_year,
        "net_profit_year_cny": net_profit_year,
        "load_ratio_unserved": load_ratio_unserved,
        "ren_ratio_curtail": ren_ratio_curtail,
        "eh_full_load_hours": eh_full_load_hours,
        "st_full_load_hours": st_full_load_hours,
        "econ_term": econ_term,
        "rel_term": rel_term,
        "curt_term": curt_term,
    }

    return obj_value, metrics, mui
