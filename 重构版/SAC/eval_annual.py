"""全年评估脚本 eval_annual.py

用于在聚类后的全年数据上，对第二阶段训练得到的所有专家策略做汇总评估。

设计目标：
- 对每个专家单独调用 eval_expert.evaluate_expert 获取指标均值；
- 利用 clustered_training_data.csv 中每个 Day_Label 对应的天数，
  按天数作为权重，对各专家指标做年度加权平均；
- 将结果保存到 eval_results/eval_annual.json，并在终端打印简要汇总。

使用方法：
    python eval_annual.py
    python eval_annual.py --episodes 50
    python eval_annual.py --config phase2_config.yaml --stochastic
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

from eval_expert import evaluate_expert


def _resolve_path(path_str: str) -> Path:
    """根据配置中的路径字符串返回可用路径。

    与 eval_expert.py 中的逻辑保持一致，尽量兼容相对路径写法。
    """
    p = Path(path_str)
    if p.exists():
        return p
    # 尝试相对于当前脚本目录
    alt = Path(__file__).parent / path_str
    if alt.exists():
        return alt
    # 尝试嵌套 SAC 目录（兼容旧路径）
    alt2 = Path(__file__).parent / "SAC" / path_str
    if alt2.exists():
        return alt2
    return p


def _load_clustered_data(config: Dict[str, Any]) -> pd.DataFrame:
    """加载聚类后的全年数据 clustered_training_data.csv。"""
    data_cfg = config.get("data", {})
    clustered_path_str = data_cfg.get("clustered", "clustered_training_data.csv")
    data_path = _resolve_path(clustered_path_str)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到聚类数据文件: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"Date", "Day_Index", "Day_Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"clustered_training_data.csv 缺少必要列: {missing}")
    return df


def _compute_days_per_label(df: pd.DataFrame) -> Dict[int, int]:
    """统计每个 Day_Label 对应的天数（按 Day_Index 去重）。"""
    day_info = df[["Date", "Day_Index", "Day_Label"]].drop_duplicates(subset=["Day_Index"])
    group = day_info.groupby("Day_Label")["Day_Index"].nunique()
    return {int(k): int(v) for k, v in group.to_dict().items()}


def main():
    parser = argparse.ArgumentParser(description="全年评估所有专家策略")
    parser.add_argument("--config", type=str, default="phase2_config.yaml", help="配置文件路径")
    parser.add_argument("--episodes", type=int, default=20, help="每个专家评估的 episode 数量")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="评估时使用随机策略（默认使用确定性策略）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # 兼容相对路径
        alt = Path(__file__).parent / args.config
        if alt.exists():
            config_path = alt
        else:
            raise FileNotFoundError(f"找不到配置文件: {args.config}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_experts = int(config.get("training", {}).get("num_experts", 5))

    print("=" * 70)
    print("🚀 全年评估所有专家策略")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"专家数量: {num_experts}")
    print(f"每个专家评估 episodes: {args.episodes}")
    print(f"评估策略: {'随机' if args.stochastic else '确定性'}")

    # 加载聚类后的全年数据，统计每个专家对应的天数
    df_clustered = _load_clustered_data(config)
    days_per_label = _compute_days_per_label(df_clustered)
    total_days = int(df_clustered["Day_Index"].nunique())

    print("\n📊 每个专家对应的天数 (按 Day_Label 统计):")
    for expert_id in range(num_experts):
        days = days_per_label.get(expert_id, 0)
        print(f"  专家 {expert_id}: {days} 天")

    # 对每个专家单独评估
    expert_summaries: Dict[int, Dict[str, Any]] = {}

    for expert_id in range(num_experts):
        print(f"\n{'=' * 70}")
        print(f"评估专家 {expert_id}/{num_experts - 1}")
        print(f"{'=' * 70}")
        summary = evaluate_expert(
            expert_id,
            config,
            episodes=args.episodes,
            deterministic=not args.stochastic,
        )
        expert_summaries[expert_id] = summary

    # 统一的指标键
    metric_keys = ["return", "cost", "curtail", "import", "export", "ramp"]

    # 年度加权平均（按天数加权）
    annual_weighted: Dict[str, float] = {}
    for key in metric_keys:
        num = 0.0
        for expert_id, summary in expert_summaries.items():
            days = days_per_label.get(expert_id, 0)
            value = float(summary.get(key, 0.0))
            num += value * days
        if total_days > 0:
            annual_weighted[key] = num / total_days
        else:
            annual_weighted[key] = 0.0

    # 汇总结果
    result: Dict[str, Any] = {
        "config_path": str(config_path),
        "episodes_per_expert": int(args.episodes),
        "stochastic_policy": bool(args.stochastic),
        "num_experts": num_experts,
        "total_days": total_days,
        "days_per_label": days_per_label,
        "experts": expert_summaries,
        "annual_weighted": annual_weighted,
    }

    # 保存到 eval_results/eval_annual.json
    result_dir = Path("eval_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "eval_annual.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📊 全年加权指标 (按天数加权)")
    print("=" * 70)
    for k, v in annual_weighted.items():
        print(f"  {k}: {v:.2f}")
    print(f"\n✓ 全年评估结果已保存: {output_path}")


if __name__ == "__main__":
    main()
