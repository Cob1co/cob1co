"""阶段一用的简易运行仿真内核。

功能：
- 给定某一日的逐时数据和一组容量配置，按固定规则进行能量平衡仿真；
- 返回该日的发电、弃风弃光、缺电、储能状态等统计指标；
- 供经济性与边际效益分析模块调用。

注意：
- 这里的调度逻辑是“规则型”的，追求简单清晰，而不是最优调度。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from models import WindFarm, PVPlant, ThermalStorage, ElectricHeater, SteamTurbine
from models.params import load_model_config


def _build_components(capacity: Dict[str, float]) -> Dict[str, Any]:
    """根据容量向量构建各个设备实例。

    容量字典字段：
    - wind_mw: 风电装机容量（MW）
    - pv_mw: 光伏装机容量（MW）
    - ts_mwh: 熔盐储热容量（MWh_th）
    - eh_mw_th: 电加热器热功率（MW_th）
    - st_mw_e: 汽轮机电功率（MW_e）
    """
    cfg = load_model_config()
    physics = cfg.get("physics", {})
    molten = physics.get("molten_salt", {})

    # 风电功率曲线（使用简单的三段式：切入/额定/切出风速）
    assets = cfg.get("assets", {})
    wind_cfg = assets.get("wind", {})
    v_in = float(wind_cfg.get("v_in_mps", 3.0))
    v_rated = float(wind_cfg.get("v_rated_mps", 12.0))
    v_out = float(wind_cfg.get("v_out_mps", 25.0))

    # 简化功率曲线：
    # 0~v_in: 0；v_in~v_rated: 线性上升到1；v_rated~v_out: 1 降到 0
    power_curve = [
        (0.0, 0.0),
        (v_in, 0.0),
        (v_rated, 1.0),
        (v_out, 0.0),
    ]

    # 构造各设备
    wind = WindFarm(capacity_mw=float(capacity.get("wind_mw", 0.0)), power_curve=power_curve)

    pv_assets = assets.get("pv", {})
    pv = PVPlant(
        capacity_mw=float(capacity.get("pv_mw", 0.0)),
        ghi_ref=float(pv_assets.get("ghi_ref_w_per_m2", 1000.0)),
        temp_coeff=float(pv_assets.get("temp_coeff_per_c", -0.004)),
        t_ref_c=float(pv_assets.get("t_ref_c", 25.0)),
    )

    ts = ThermalStorage(
        energy_cap_mwh=float(capacity.get("ts_mwh", 0.0)),
        loss_rate_per_h=float(molten.get("loss_rate_per_h", 0.0)),
        soc_min_frac=float(molten.get("soc_min", 0.0)),
        soc_max_frac=float(molten.get("soc_max", 1.0)),
    )

    eh_cfg = assets.get("electric_heater", {})
    eh = ElectricHeater(
        cap_mw_th=float(capacity.get("eh_mw_th", 0.0)),
        eta=float(eh_cfg.get("eta", 0.98)),
    )

    st_cfg = assets.get("steam_turbine", {})
    st = SteamTurbine(
        cap_mw_e=float(capacity.get("st_mw_e", 0.0)),
        eta=float(st_cfg.get("eta", 0.38)),
        ramp_mw_per_h=float(st_cfg.get("ramp_mw_per_h", 0.0)) or None,
        p_min_frac=float(st_cfg.get("p_min_frac", 0.0)),
    )

    return {"wind": wind, "pv": pv, "ts": ts, "eh": eh, "st": st}


def simulate_day(
    day_df: pd.DataFrame,
    capacity: Dict[str, float],
    initial_soc: float,
    grid_import_limit_mw: float,
    dt_hours: float = 1.0,
) -> Dict[str, Any]:
    """在给定容量和简单运行规则下，模拟单个典型日。

    返回的关键指标包括：
    - energy_load_mwh: 当日总负荷电量
    - energy_served_mwh: 当日实际供电量
    - energy_unserved_mwh: 当日缺电电量
    - energy_ren_gen_mwh: 当日风光总发电量
    - energy_curtail_mwh: 当日弃风弃光电量
    - revenue_cny: 当日供能收益（负荷侧电价 * 实际供电量）
    - eh_full_load_hours: 电加热器满负荷小时数
    - st_full_load_hours: 汽轮机满负荷小时数
    """
    comps = _build_components(capacity)
    wind: WindFarm = comps["wind"]
    pv: PVPlant = comps["pv"]
    ts: ThermalStorage = comps["ts"]
    eh: ElectricHeater = comps["eh"]
    st: SteamTurbine = comps["st"]

    # 重置状态
    ts.reset(soc_frac=float(initial_soc))
    eh.reset()
    st.reset(p0_mw=0.0)
    wind.reset()
    pv.reset()

    energy_load_mwh = 0.0
    energy_served_mwh = 0.0
    energy_unserved_mwh = 0.0
    energy_ren_gen_mwh = 0.0
    energy_curtail_mwh = 0.0
    # 并网购电统计
    energy_grid_import_mwh = 0.0
    cost_grid_import_cny = 0.0
    revenue_cny = 0.0

    eh_full_load_hours = 0.0
    st_full_load_hours = 0.0

    # 逐时仿真
    for _, row in day_df.iterrows():
        temp_c = float(row["Temperature_C"])
        ghi = float(row["Solar_W_m2"])
        wind_speed = float(row["Wind_Speed_m_s"])
        load_mw = float(row["Load_kW"]) / 1000.0
        price = float(row["Price_CNY_kWh"])

        energy_load_mwh += load_mw * dt_hours

        # 风光出力
        w_out = wind.step({"wind_speed": wind_speed}, dt_hours=dt_hours)
        p_w_mw = float(w_out["p_mw"])

        pv_out = pv.step({"ghi": ghi, "temp": temp_c}, dt_hours=dt_hours)
        p_pv_mw = float(pv_out["p_mw"])

        p_ren_mw = p_w_mw + p_pv_mw
        energy_ren_gen_mwh += p_ren_mw * dt_hours

        # 当前时刻供需平衡
        p_st_e_mw = 0.0
        p_eh_e_in_mw = 0.0
        p_eh_th_out_mw = 0.0
        p_curtail_mw = 0.0
        p_unserved_mw = 0.0

        if p_ren_mw >= load_mw:
            # 可再生出力超过负荷：先满足负荷，多余电充储热，剩余弃风弃光
            p_surplus = p_ren_mw - load_mw
            # 电加热器充热
            eh_out = eh.step({"p_elec_in_mw": p_surplus}, dt_hours=dt_hours)
            p_eh_e_in_mw = float(eh_out["p_elec_used_mw"])
            p_eh_th_out_mw = float(eh_out["p_th_out_mw"])

            # 储热仅充电
            ts_out = ts.step(
                {"charge_th_mw": p_eh_th_out_mw, "discharge_th_mw": 0.0},
                dt_hours=dt_hours,
            )
            p_charge_acc = float(ts_out["charge_accepted_mw"])

            # 电加热器未用满的电量 + 储热未接收的热量都视为弃风弃光
            # 热量差折算为等效电功率（近似按电加热效率还原）
            unused_elec_mw = max(0.0, p_surplus - p_eh_e_in_mw)
            unused_th_mw = max(0.0, p_eh_th_out_mw - p_charge_acc)
            # 这里简单按 1:1 折算到电侧
            p_curtail_mw = unused_elec_mw + unused_th_mw
        else:
            # 可再生出力不足：先用储热+汽轮机补，仍不足则视为缺电
            p_deficit = load_mw - p_ren_mw

            # 尝试由汽轮机补足缺口
            # 先按电功率目标估算需要的热功率
            eta_st = st.eta if st.eta > 0 else 0.0
            if eta_st > 0.0:
                p_th_need_mw = p_deficit / eta_st
            else:
                p_th_need_mw = 0.0

            # 储热放电
            ts_out = ts.step(
                {"charge_th_mw": 0.0, "discharge_th_mw": p_th_need_mw},
                dt_hours=dt_hours,
            )
            p_th_dis_mw = float(ts_out["discharge_accepted_mw"])

            # 汽轮机发电
            st_out = st.step({"p_th_in_mw": p_th_dis_mw}, dt_hours=dt_hours)
            p_st_e_mw = float(st_out["p_e_out_mw"])

            # 可再生+汽轮机可用电功率
            p_available_mw = p_ren_mw + p_st_e_mw

            # 如允许从大电网购电，则在此处补足一部分缺口
            p_import_mw = 0.0
            if grid_import_limit_mw > 0.0:
                # 仅在本地资源不足时才考虑购电
                p_def_after_local = max(0.0, load_mw - p_available_mw)
                if p_def_after_local > 0.0:
                    p_import_mw = min(p_def_after_local, grid_import_limit_mw)
                    if p_import_mw > 0.0:
                        # 购电影量与成本
                        energy_grid_import_mwh += p_import_mw * dt_hours
                        cost_grid_import_cny += p_import_mw * dt_hours * 1000.0 * price
                        p_available_mw += p_import_mw

            if p_available_mw >= load_mw:
                # 通过可再生+汽轮机+购电即可满足负荷
                p_unserved_mw = 0.0
                # 理论上不会出现过剩，但若因汽轮机最小出力导致过剩，则计入弃风
                p_curtail_mw = max(0.0, p_available_mw - load_mw)
            else:
                # 仍有缺口，视为缺电
                p_unserved_mw = load_mw - p_available_mw

        # 统计结果
        energy_served_mwh += (load_mw - p_unserved_mw) * dt_hours
        energy_unserved_mwh += p_unserved_mw * dt_hours
        energy_curtail_mwh += p_curtail_mw * dt_hours

        # 收益：按负荷电价 * 实际供电量（内部售电）
        served_mwh = (load_mw - p_unserved_mw) * dt_hours
        revenue_cny += served_mwh * 1000.0 * price

        # 满负荷运行统计（简单按接近额定功率判断）
        if eh.cap_th > 0.0 and p_eh_th_out_mw >= 0.95 * eh.cap_th:
            eh_full_load_hours += dt_hours
        if st.cap_e > 0.0 and p_st_e_mw >= 0.95 * st.cap_e:
            st_full_load_hours += dt_hours

    return {
        "energy_load_mwh": energy_load_mwh,
        "energy_served_mwh": energy_served_mwh,
        "energy_unserved_mwh": energy_unserved_mwh,
        "energy_ren_gen_mwh": energy_ren_gen_mwh,
        "energy_curtail_mwh": energy_curtail_mwh,
        "energy_grid_import_mwh": energy_grid_import_mwh,
        "cost_grid_import_cny": cost_grid_import_cny,
        "revenue_cny": revenue_cny,
        "eh_full_load_hours": eh_full_load_hours,
        "st_full_load_hours": st_full_load_hours,
    }
