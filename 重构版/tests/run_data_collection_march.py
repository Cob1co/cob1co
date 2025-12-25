"""运行 Phase 3 DataCollector，基于 2023 年 3 月数据收集训练样本。

输出：
- 使用 DataCollector.collect_month(year=2023, month=3, save=True)
- 将训练数据保存到 phase3_config.yaml 中 data.transformer_training_data 指定路径
- 在终端打印样本数量和最优权重分布统计
"""

import sys
from pathlib import Path
from collections import Counter

import argparse
import yaml
import numpy as np

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from LMPC.training.data_collector import DataCollector  # noqa: E402
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print(f"🚀 Phase 3 DataCollector - {args.year}年{args.month}月")
    print("=" * 70)

    # 加载配置
    config_path = PROJECT_ROOT / "LMPC" / "phase3_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化 DataCollector
    collector = DataCollector(config)

    # 收集 2023 年 3 月的数据
    print(f"\n📋 开始收集 {args.year}-{args.month:02d} 的训练数据...")
    dataset = collector.collect_month(year=args.year, month=args.month, save=True)

    num_samples = len(dataset)
    print(f"\n✅ 收集完成，样本数: {num_samples}")

    if num_samples == 0:
        print("⚠️ 警告：当前没有生成任何样本，请检查数据对齐和配置。")
        return

    # 统计最优权重分布
    weights_list = [tuple(np.round(s["optimal_weights"], 3)) for s in dataset]
    counter = Counter(weights_list)

    print("\n📊 最优权重分布 (alpha_soc, alpha_grid, alpha_cost) → 计数：")
    for w, c in counter.most_common():
        print(f"  {w}: {c}")

    print("\n✅ DataCollector 运行结束！")


if __name__ == "__main__":
    main()
