"""第一阶段容量配置主入口脚本。

使用方法：
- 在项目根目录下运行：python -m SA-MBPS.run_phase1_main （或在该目录下直接 python run_phase1_main.py）。
- 依赖：pandas, pyyaml, scikit-learn。

功能：
- 读取 phase1_config.yaml 和 data2023.csv；
- 使用 scenario_manager 构建典型日场景；
- 调用 MUGHOptimizer 搜索最优容量配置；
- 输出配置结果和关键年度指标。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import math
import pandas as pd
import yaml

from scenario_manager import build_scenarios
from phase1_optimizer import MUGHOptimizer
from economics_engine import evaluate_capacity


def load_phase1_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    return cfg


def ceil_to_multiple(value: float, step: float = 5.0) -> float:
    """将数值向上取整到指定步长的倍数。"""
    if value <= 0:
        return 0.0
    return math.ceil(value / step) * step


def main() -> None:
    # 定位配置与数据
    base_dir = Path(__file__).resolve().parent.parent  # 项目根目录
    phase1_dir = Path(__file__).resolve().parent

    phase1_cfg_path = phase1_dir / "phase1_config.yaml"
    cfg = load_phase1_config(phase1_cfg_path)

    data_rel = cfg.get("data", {}).get("weather_load_price_csv")
    if not data_rel:
        raise RuntimeError("phase1_config.yaml 中 data.weather_load_price_csv 未设置")
    data_path = base_dir / str(data_rel)

    # 读取全年数据
    df = pd.read_csv(data_path)

    # 构建典型日场景
    scenarios_cfg = cfg.get("scenarios", {})
    scenarios = build_scenarios(df, scenarios_cfg)
    if not scenarios:
        raise RuntimeError("未能构建任何典型日场景，请检查数据和配置。")

    # 优化
    optimizer = MUGHOptimizer(cfg, scenarios)
    best_cap, best_metrics = optimizer.optimize()

    # 输出结果
    print("=== Phase 1 容量配置结果 ===")
    print("最优容量 (连续值，可根据需要向上取整)：")
    for k, v in best_cap.items():
        print(f"  {k}: {v:.3f}")

    # --- 向上取整并重新评估 --- 
    print("\n=== 工程取整后的配置 (向上取整至 5 的倍数) ===")
    final_cap = {
        k: ceil_to_multiple(v, 5.0) for k, v in best_cap.items()
    }
    # 特殊处理：如果某个容量原本是0，且取整后仍为0，保持0；如果原本很小但被取整成5，也可以接受。
    # 这里简单打印一下
    for k, v in final_cap.items():
        print(f"  {k}: {v:.1f}")

    # 使用取整后的容量重新评估
    _, metrics_final, _ = evaluate_capacity(final_cap, scenarios, cfg)

    print("\n=== 最终配置的年度关键指标 (基于取整后容量) ===")
    
    # 定义中文标签映射，方便阅读
    labels = {
        "net_profit_year_cny": "年度净利润 (元)",
        "total_cost_year_cny": "年度总成本 (元)",
        "capex_sum_cny": "初始总投资 (元)",
        "capex_ann_cny": "年化投资成本 (元)",
        "opex_year_cny": "年运维成本 (元)",
        "penalty_cny": "年罚款金额 (元)",
        "revenue_cny": "年售电收入 (元)",
        "e_load_mwh": "年总负荷 (MWh)",
        "e_served_mwh": "年供电量 (MWh)",
        "e_unserved_mwh": "年缺电量 (MWh)",
        "e_ren_mwh": "年新能源发电量 (MWh)",
        "e_curtail_mwh": "年弃风弃光量 (MWh)",
        "load_ratio_unserved": "缺电率 (Unserved Ratio)",
        "ren_ratio_curtail": "弃电率 (Curtailment Ratio)",
        "eh_full_load_hours": "电加热器满载小时数 (h)",
        "st_full_load_hours": "汽轮机满载小时数 (h)",
    }

    # 打印主要指标
    keys_to_print = [
        "net_profit_year_cny", "total_cost_year_cny", "revenue_cny",
        "capex_sum_cny", "penalty_cny",
        "e_load_mwh", "e_served_mwh", "e_unserved_mwh",
        "e_ren_mwh", "e_curtail_mwh",
        "load_ratio_unserved", "ren_ratio_curtail",
        "eh_full_load_hours", "st_full_load_hours"
    ]

    for k in keys_to_print:
        val = metrics_final.get(k, 0.0)
        label = labels.get(k, k)
        
        # 针对不同量级做格式化
        if "ratio" in k:
            print(f"  {label}: {val * 100:.2f}%")
        elif "cny" in k:
            print(f"  {label}: {val:,.2f}")
        else:
            print(f"  {label}: {val:,.3f}")


if __name__ == "__main__":
    main()
