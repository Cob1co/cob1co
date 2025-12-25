"""分析训练数据质量

检查标签分布、异常值、样本质量
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_training_data(data_path):
    """分析训练数据"""
    
    print("="*70)
    print("📊 训练数据质量分析")
    print("="*70)
    
    # 加载数据
    print("\n📋 加载数据...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = np.array(data['features'])
    labels = np.array(data['labels'])
    
    print(f"✅ 样本数: {len(labels)}")
    print(f"✅ 特征形状: {features.shape}")
    print(f"✅ 标签形状: {labels.shape}")
    
    # 分析标签分布
    print("\n📊 标签统计分析")
    print("-"*70)
    
    alpha_soc = labels[:, 0]
    alpha_grid = labels[:, 1]
    alpha_cost = labels[:, 2]
    
    print(f"\nα_soc (SOC跟踪权重):")
    print(f"  均值: {np.mean(alpha_soc):.3f}")
    print(f"  标准差: {np.std(alpha_soc):.3f}")
    print(f"  最小值: {np.min(alpha_soc):.3f}")
    print(f"  最大值: {np.max(alpha_soc):.3f}")
    print(f"  中位数: {np.median(alpha_soc):.3f}")
    
    print(f"\nα_grid (电网跟踪权重):")
    print(f"  均值: {np.mean(alpha_grid):.3f}")
    print(f"  标准差: {np.std(alpha_grid):.3f}")
    print(f"  最小值: {np.min(alpha_grid):.3f}")
    print(f"  最大值: {np.max(alpha_grid):.3f}")
    print(f"  中位数: {np.median(alpha_grid):.3f}")
    
    print(f"\nα_cost (成本权重):")
    print(f"  均值: {np.mean(alpha_cost):.3f}")
    print(f"  标准差: {np.std(alpha_cost):.3f}")
    print(f"  最小值: {np.min(alpha_cost):.3f}")
    print(f"  最大值: {np.max(alpha_cost):.3f}")
    print(f"  中位数: {np.median(alpha_cost):.3f}")
    
    # 分析标签组合
    print("\n📊 标签组合分析")
    print("-"*70)
    
    # 统计每种组合出现的次数
    from collections import Counter
    label_tuples = [tuple(label) for label in labels]
    label_counts = Counter(label_tuples)
    
    print(f"不同标签组合数: {len(label_counts)}")
    print(f"\n前10个最常见的标签组合:")
    for i, (label, count) in enumerate(label_counts.most_common(10), 1):
        pct = count / len(labels) * 100
        print(f"  {i}. {label} - {count}次 ({pct:.1f}%)")
    
    # 分析偏离程度
    print("\n📊 权重偏离基准(1.0)的分析")
    print("-"*70)
    
    deviations = np.mean(np.abs(labels - 1.0), axis=1)
    
    normal = np.sum(deviations < 0.1)
    important = np.sum((deviations >= 0.1) & (deviations < 0.3))
    extreme = np.sum(deviations >= 0.3)
    
    print(f"普通样本 (<0.1): {normal} ({normal/len(labels)*100:.1f}%)")
    print(f"重要样本 (0.1-0.3): {important} ({important/len(labels)*100:.1f}%)")
    print(f"极端样本 (>0.3): {extreme} ({extreme/len(labels)*100:.1f}%)")
    
    print(f"\n平均偏离度: {np.mean(deviations):.3f}")
    print(f"最大偏离度: {np.max(deviations):.3f}")
    print(f"最小偏离度: {np.min(deviations):.3f}")
    
    # 可视化
    print("\n📊 生成可视化...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. α_soc直方图
    axes[0, 0].hist(alpha_soc, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(1.0, color='r', linestyle='--', label='基准(1.0)')
    axes[0, 0].set_xlabel('α_soc')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('α_soc分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. α_grid直方图
    axes[0, 1].hist(alpha_grid, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(1.0, color='r', linestyle='--', label='基准(1.0)')
    axes[0, 1].set_xlabel('α_grid')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('α_grid分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. α_cost直方图
    axes[0, 2].hist(alpha_cost, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(1.0, color='r', linestyle='--', label='基准(1.0)')
    axes[0, 2].set_xlabel('α_cost')
    axes[0, 2].set_ylabel('频数')
    axes[0, 2].set_title('α_cost分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. α_soc vs α_grid散点图
    axes[1, 0].scatter(alpha_soc, alpha_grid, alpha=0.3, s=10)
    axes[1, 0].axvline(1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('α_soc')
    axes[1, 0].set_ylabel('α_grid')
    axes[1, 0].set_title('α_soc vs α_grid')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. α_soc vs α_cost散点图
    axes[1, 1].scatter(alpha_soc, alpha_cost, alpha=0.3, s=10)
    axes[1, 1].axvline(1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('α_soc')
    axes[1, 1].set_ylabel('α_cost')
    axes[1, 1].set_title('α_soc vs α_cost')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 偏离度直方图
    axes[1, 2].hist(deviations, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(0.1, color='orange', linestyle='--', label='普通/重要阈值')
    axes[1, 2].axvline(0.3, color='red', linestyle='--', label='重要/极端阈值')
    axes[1, 2].set_xlabel('平均偏离度')
    axes[1, 2].set_ylabel('频数')
    axes[1, 2].set_title('权重偏离度分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / 'LMPC' / 'data' / 'training_data_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ 可视化已保存: {output_path}")
    
    # 检查潜在问题
    print("\n⚠️  潜在问题检查")
    print("-"*70)
    
    issues = []
    
    # 1. 检查是否有异常值
    if np.max(alpha_soc) > 1.5 or np.min(alpha_soc) < 0.5:
        issues.append("α_soc存在超出合理范围的异常值")
    
    if np.max(alpha_grid) > 1.5 or np.min(alpha_grid) < 0.5:
        issues.append("α_grid存在超出合理范围的异常值")
    
    if np.max(alpha_cost) > 1.5 or np.min(alpha_cost) < 0.5:
        issues.append("α_cost存在超出合理范围的异常值")
    
    # 2. 检查分布是否过于集中
    if len(label_counts) < 10:
        issues.append(f"标签组合过少({len(label_counts)}种)，可能数据多样性不足")
    
    # 3. 检查是否有候选值占比过高
    unique_soc, counts_soc = np.unique(alpha_soc, return_counts=True)
    max_count_soc = np.max(counts_soc)
    if max_count_soc / len(labels) > 0.5:
        issues.append(f"α_soc有单个值占比过高({max_count_soc/len(labels)*100:.1f}%)")
    
    # 4. 检查普通样本比例
    if normal == 0:
        issues.append("⚠️ 没有普通样本(偏离<0.1)，所有样本都需要大幅调整权重")
    
    if len(issues) == 0:
        print("✅ 未发现明显问题")
    else:
        print(f"❌ 发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'LMPC' / 'data' / 'training_data_30days.pkl'
    analyze_training_data(data_path)
