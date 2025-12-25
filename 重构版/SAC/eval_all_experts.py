"""批量评估所有专家策略并生成对比报告"""

import json
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

from eval_expert import evaluate_expert


def evaluate_all_experts(config: dict, episodes: int = 20):
    """评估所有5个专家并生成对比报告"""
    
    num_experts = config["training"]["num_experts"]
    results = []
    
    print("=" * 70)
    print("批量评估所有专家策略")
    print("=" * 70)
    print(f"\n配置信息:")
    print(f"  专家数量: {num_experts}")
    print(f"  评估Episodes: {episodes}")
    print()
    
    # 逐个评估专家
    for expert_id in range(num_experts):
        print(f"\n{'=' * 70}")
        print(f"评估专家 {expert_id}")
        print(f"{'=' * 70}\n")
        
        try:
            evaluate_expert(expert_id, config, episodes=episodes, deterministic=True)
            
            # 读取评估结果
            result_path = Path("eval_results") / f"expert_{expert_id}_eval.json"
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                result["expert_id"] = expert_id
                results.append(result)
            
            print(f"✓ 专家 {expert_id} 评估完成")
            
        except Exception as e:
            print(f"✗ 专家 {expert_id} 评估失败: {e}")
            continue
    
    # 生成对比报告
    if results:
        generate_comparison_report(results)
        plot_comparison_charts(results)
    
    print("\n" + "=" * 70)
    print("所有专家评估完成!")
    print("=" * 70)


def generate_comparison_report(results: list):
    """生成对比报告"""
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存详细数据
    output_path = Path("eval_results") / "all_experts_comparison.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 对比数据已保存: {output_path}")
    
    # 打印对比表格
    print("\n" + "=" * 90)
    print("所有专家性能对比")
    print("=" * 90)
    
    print(f"\n{'专家ID':<8} {'平均回报':<12} {'平均成本(元)':<15} {'弃电(MWh)':<12} {'购电(MW)':<12} {'售电(MW)':<12}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{int(row['expert_id']):<8} "
              f"{row['return']:<12.2f} "
              f"{row['cost']:<15.2f} "
              f"{row['curtail']:<12.2f} "
              f"{row['import']:<12.2f} "
              f"{row['export']:<12.2f}")
    
    print("-" * 90)
    
    # 统计汇总
    print(f"\n{'汇总统计':<8} "
          f"{'平均':<12} "
          f"{'平均':<15} "
          f"{'平均':<12} "
          f"{'平均':<12} "
          f"{'平均':<12}")
    
    print(f"{'所有专家':<8} "
          f"{df['return'].mean():<12.2f} "
          f"{df['cost'].mean():<15.2f} "
          f"{df['curtail'].mean():<12.2f} "
          f"{df['import'].mean():<12.2f} "
          f"{df['export'].mean():<12.2f}")
    
    print(f"{'标准差':<8} "
          f"{df['return'].std():<12.2f} "
          f"{df['cost'].std():<15.2f} "
          f"{df['curtail'].std():<12.2f} "
          f"{df['import'].std():<12.2f} "
          f"{df['export'].std():<12.2f}")
    
    # 找出最佳专家
    print("\n" + "=" * 90)
    print("最佳专家")
    print("=" * 90)
    
    best_return = df.loc[df['return'].idxmax()]
    best_cost = df.loc[df['cost'].idxmin()]  # 成本越负越好（收益越高）
    best_curtail = df.loc[df['curtail'].idxmin()]
    
    print(f"  最高回报: 专家 {int(best_return['expert_id'])} (回报={best_return['return']:.2f})")
    print(f"  最高收益: 专家 {int(best_cost['expert_id'])} (成本={best_cost['cost']:.2f}元)")
    print(f"  最低弃电: 专家 {int(best_curtail['expert_id'])} (弃电={best_curtail['curtail']:.2f} MWh)")


def plot_comparison_charts(results: list):
    """绘制对比图表"""
    
    df = pd.DataFrame(results)
    expert_ids = df['expert_id'].values
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('所有专家性能对比', fontsize=16, fontweight='bold')
    
    metrics = [
        ('return', '平均回报', 'tab:blue'),
        ('cost', '平均成本 (元)', 'tab:red'),
        ('curtail', '弃电量 (MWh)', 'tab:orange'),
        ('import', '购电量 (MW)', 'tab:green'),
        ('export', '售电量 (MW)', 'tab:purple'),
        ('ramp', '电网波动 (MW)', 'tab:brown'),
    ]
    
    for idx, (metric, label, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = df[metric].values
        
        bars = ax.bar(expert_ids, values, color=color, alpha=0.7, edgecolor='black')
        
        # 标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('专家ID', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(expert_ids)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path("eval_results") / "all_experts_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存: {output_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="批量评估所有专家策略")
    parser.add_argument("--config", type=str, default="phase2_config.yaml", help="配置文件路径")
    parser.add_argument("--episodes", type=int, default=20, help="每个专家的评估episodes数量")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    evaluate_all_experts(config, episodes=args.episodes)


if __name__ == "__main__":
    main()
