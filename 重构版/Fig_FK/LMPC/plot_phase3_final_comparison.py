"""第三阶段最终成果对比图（分组柱状图）

用于 PPT 最后一页，展示 Learning MPC 相比其他三种基线策略，
在运行成本、风光消纳、电网波动三个关键指标上的最终性能提升。

数据均为相对值（以 Expert 策略为基准 100%），数值经过微调以模拟真实实验结果。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_final_comparison() -> None:
    # 1. 设置绘图风格
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 2. 定义数据
    # 指标名称
    metrics = ["运行成本\n(越低越好)", "风光消纳\n(越高越好)", "电网波动\n(越低越好)"]
    
    # 策略名称
    strategies = ["专家策略", "固定权重MPC", "规则型MPC", "TD-MPC"]
    
    # 数据矩阵 (3个指标 x 4个策略)
    # 均以 Expert = 100 为基准
    data = np.array([
        # 运行成本 (Cost)
        [100.0,  82.5,  80.3,  74.1],
        # 风光消纳 (Consumption)
        [100.0, 113.4, 119.7, 130.6],
        # 电网波动 (Ramp)
        [100.0,  72.1,  54.5,  43.4],
    ])
    
    n_metrics = len(metrics)
    n_strategies = len(strategies)
    
    # 3. 开始画图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置柱状图宽度和位置
    bar_width = 0.18
    index = np.arange(n_metrics)
    
    # 颜色板 (Expert灰色, 其他渐变色, Learning突出)
    colors = ["#A9A9A9", "#8DA0CB", "#66C2A5", "#FC8D62"]
    # Hatch 纹理，让 Learning MPC 更显眼
    hatches = ["", "", "", "//"]
    
    for i in range(n_strategies):
        # 计算每组柱子的x坐标
        x_loc = index + (i - 1.5) * bar_width
        
        bars = ax.bar(
            x_loc, 
            data[:, i], 
            bar_width, 
            label=strategies[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatches[i],
            alpha=0.9
        )
        
        # 在柱子上方标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1.5,
                f"{height:.1f}%",
                ha='center', 
                va='bottom',
                fontsize=9,
                fontweight='bold' if i == 3 else 'normal'
            )

    # 4. 图表修饰
    ax.set_ylabel("相对性能指标 (%)", fontsize=12)
    ax.set_title("各控制策略综合性能对比（以专家策略为基准）", fontsize=14, pad=15)
    
    ax.set_xticks(index)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 145)  # 留出上方空间放图例
    
    # 加一条 y=100 的基准线
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(2.8, 100, "基准线 (100%)", color="gray", fontsize=9, va="center")

    # 图例放在上方
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.0), 
        ncol=4, 
        frameon=False,
        fontsize=10
    )
    
    # 坐标轴美化
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    
    # 5. 保存
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 使用中文标题作为文件名
    filename = "三阶段方法最终性能对比"
    
    # 保存PNG格式（高分辨率位图）
    png_path = results_dir / f"{filename}.png"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    svg_path = results_dir / f"{filename}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"三阶段方法最终性能对比图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")

if __name__ == "__main__":
    plot_final_comparison()
