"""运行 Phase 3 DataCollector，基于 2023 年 1-12 月数据收集训练样本（Oracle 方案）。

说明：
- 对每个月调用一次 DataCollector.collect_month(year=2023, month=m, save=False)；
- 将所有月份的样本合并为一个大列表；
- 最终统一保存到 phase3_config.yaml 中 data.transformer_training_data 指定的路径；
- 在终端打印总样本数和全年的最优权重分布统计。

使用方法（在项目根目录）：

    python tests/run_data_collection_2023_full.py

注意：
- 该脚本运行时间较长（约为单月 3 月的 10-12 倍），建议空闲时运行；
- 会覆盖原来的 transformer_training_data.pkl（如果存在），请事先确认不再需要旧的 3 月小数据集。
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

import yaml
import numpy as np
import pickle

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from LMPC.training.data_collector import DataCollector  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("🚀 Phase 3 DataCollector - 2023全年 (Oracle)")
    print("=" * 70)

    # 加载配置
    config_path = PROJECT_ROOT / "LMPC" / "phase3_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # 初始化 DataCollector
    collector = DataCollector(config)

    # 需要收集的月份：1-12 月
    months = list(range(1, 13))

    all_samples: List[Dict[str, Any]] = []
    counter: Counter = Counter()

    for month in months:
        print("\n" + "-" * 70)
        print(f"📋 开始收集 2023-{month:02d} 的训练数据...")
        dataset = collector.collect_month(year=2023, month=month, save=False)

        num_samples = len(dataset)
        print(f"✅ 2023-{month:02d} 收集完成，样本数: {num_samples}")

        all_samples.extend(dataset)

        # 更新全年权重统计
        for s in dataset:
            w = tuple(np.round(s["optimal_weights"], 3))
            counter[w] += 1

    total_samples = len(all_samples)
    print("\n" + "=" * 70)
    print(f"✅ 2023 全年收集完成，总样本数: {total_samples}")

    if total_samples == 0:
        print("⚠️ 警告：全年未生成任何样本，请检查数据对齐和配置。")
        return

    # 保存到配置中指定的路径（会覆盖原文件）
    data_cfg = config.get("data", {})
    rel_path = data_cfg.get("transformer_training_data", "data/transformer_training_data.pkl")
    out_path = PROJECT_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"\n✅ 全年训练数据已保存到: {out_path}，样本数: {total_samples}")

    # 打印全年最优权重分布
    print("\n📊 全年最优权重分布 (alpha_soc, alpha_grid, alpha_cost) → 计数：")
    for w, c in counter.most_common():
        print(f"  {w}: {c}")

    print("\n✅ DataCollector 全年运行结束！")


if __name__ == "__main__":
    main()
