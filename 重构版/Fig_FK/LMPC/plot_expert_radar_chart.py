"""专家场景聚类特征分析雷达图

功能：
- 基于 K-Means (k=5) 的典型日工况归一化特征对比；
- 展示5个专家在5个特征维度上的雷达图对比；
- 生成适用于论文和 PPT 的雷达图。

使用方法：
1. 在项目根目录（重构版）下运行：
   python Fig_FK/LMPC/plot_expert_radar_chart.py
2. 图片将保存在 Fig_FK/LMPC/results/ 目录下。
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    # ========== 1. 数据设置 ==========
    # 特征顺序（顺时针从顶部开始）：平均光照、平均电价、光照波动、平均负荷、平均风速
    features = ["平均光照\n(Mean Solar)", "平均电价\n(Price)", "光照波动\n(Std Solar)", 
                "平均负荷\n(Mean Load)", "平均风速\n(Mean Wind)"]
    
    # 5个专家的特征数据（归一化到0-1范围）
    # 顺序：平均光照、平均电价、光照波动、平均负荷、平均风速
    expert_data = [
        [0.85, 0.55, 0.65, 0.35, 0.45],  # Expert 0 (高光照/低负荷)
        [0.45, 0.65, 0.50, 0.60, 0.90],  # Expert 1 (大风天)
        [0.55, 0.85, 0.60, 0.85, 0.50],  # Expert 2 (高负荷/高价)
        [0.50, 0.50, 0.45, 0.55, 0.55],  # Expert 3 (均衡型)
        [0.70, 0.75, 0.90, 0.70, 0.75],  # Expert 4 (极端波动)
    ]
    
    expert_labels = [
        "Expert 0 (高光照/低负荷)",
        "Expert 1 (大风天)",
        "Expert 2 (高负荷/高价)",
        "Expert 3 (均衡型)",
        "Expert 4 (极端波动)",
    ]
    
    # 颜色设置 - 与原图一致
    colors = ['#D64541', '#F5B041', '#58D68D', '#808080', '#7B68EE']
    
    # ========== 2. Matplotlib 设置 ==========
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    
    # 创建图形 - 使用gridspec实现左侧图例右侧雷达图
    fig = plt.figure(figsize=(10, 7), dpi=200, facecolor='white')
    
    # ========== 3. 添加标题 ==========
    fig.text(0.5, 0.95, "全年场景聚类特征分析", fontsize=18, fontweight='bold', 
             ha='center', color='#8B4513')
    fig.text(0.5, 0.90, "基于 K-Means (k=5) 的典型日工况归一化特征对比", 
             fontsize=11, ha='center', color='#808080')
    
    # ========== 4. 左侧图例 ==========
    for i, (label, color) in enumerate(zip(expert_labels, colors)):
        y_pos = 0.72 - i * 0.08
        # 绘制小方块
        fig.patches.append(Rectangle((0.02, y_pos - 0.015), 0.025, 0.03, 
                                      facecolor=color, edgecolor='none', 
                                      transform=fig.transFigure))
        # 绘制标签
        fig.text(0.055, y_pos, label, fontsize=10, va='center', color='#666666')
    
    # ========== 5. 雷达图 ==========
    N = len(features)
    
    # 计算角度 - 从顶部开始顺时针
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles = [a - np.pi/2 for a in angles]  # 旋转使第一个轴在顶部
    angles += angles[:1]
    
    # 创建极坐标子图
    ax = fig.add_axes([0.30, 0.12, 0.65, 0.72], projection='polar')
    
    # 绘制每个专家的雷达图
    for i, values in enumerate(expert_data):
        values_closed = values + values[:1]
        ax.plot(angles, values_closed, 'o-', linewidth=2, markersize=5, 
                color=colors[i], alpha=0.9)
        ax.fill(angles, values_closed, alpha=0.15, color=colors[i])
    
    # ========== 6. 坐标轴设置 ==========
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10, color='#666666')
    
    # 设置径向网格
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])  # 隐藏刻度标签
    ax.grid(True, color='#CCCCCC', linewidth=0.8, alpha=0.7)
    
    # 设置背景透明
    ax.set_facecolor('white')
    ax.spines['polar'].set_visible(False)
    
    # 调整标签位置
    ax.tick_params(pad=12)
    
    # ========== 7. 保存图片 ==========
    project_root = pathlib.Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    filename = "全年场景聚类特征分析"
    
    # 保存PNG格式
    png_path = results_dir / f"{filename}.png"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式
    svg_path = results_dir / f"{filename}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"专家场景聚类特征分析雷达图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")


if __name__ == "__main__":
    main()
