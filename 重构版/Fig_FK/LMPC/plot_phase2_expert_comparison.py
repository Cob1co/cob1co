"""Phase 2 SAC 专家评估结果可视化脚本

功能：
- 读取 SAC/eval_results/all_experts_comparison.csv 中的评估结果；
- 对 5 个专家在关键指标上的表现进行柱状对比；
- 生成适合 PPT 使用的中文图：phase2_experts_comparison.png。

使用方法：
1. 确保已经运行过第二阶段评估脚本：
   python SAC/eval_all_experts.py
   会在 SAC/eval_results/ 下生成 all_experts_comparison.csv；
2. 在项目根目录（重构版）下运行本脚本：
   python Fig_FK/plot_phase2_expert_comparison.py
3. 生成的图片保存在 Fig_FK/phase2_experts_comparison.png，可直接插入 PPT。
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    # ========== 1. 读取评估结果 ==========
    # 脚本在 Fig_FK/LMPC/ 目录下，需要回到项目根目录（重构版）
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    eval_csv = project_root / "SAC" / "eval_results" / "all_experts_comparison.csv"

    if not eval_csv.exists():
        print("未找到评估结果文件: all_experts_comparison.csv")
        print("请先在项目根目录下运行: python SAC/eval_all_experts.py")
        return

    df = pd.read_csv(eval_csv)

    # 专家 ID
    expert_ids = df["expert_id"].to_numpy()

    # 选取三项核心指标：平均回报、平均成本、平均弃电量
    metric_names = ["return", "cost", "curtail"]
    metric_labels = ["平均回报", "平均成本 (元)", "弃电量 (KWh)"]
    colors = ["tab:blue", "tab:red", "tab:orange"]

    # ========== 2. Matplotlib 全局中文 & 样式设置 ==========
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=200)

    for ax, m_name, m_label, color in zip(axes, metric_names, metric_labels, colors):
        values = df[m_name].to_numpy(dtype=float)

        bars = ax.bar(expert_ids, values, color=color, alpha=0.8, edgecolor="black")

        # 在柱子上方标出数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            # 确保文本始终显示在柱子上方，避免遮挡
            if height >= 0:
                y_pos = height + abs(height) * 0.02  # 正值：在柱子上方留小间距
                va_align = "bottom"
            else:
                y_pos = height - abs(height) * 0.02  # 负值：在柱子下方留小间距
                va_align = "top"
            
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{val:.1f}",
                ha="center",
                va=va_align,
                fontsize=8,
            )

        ax.set_xlabel("专家 ID", fontsize=10)
        ax.set_ylabel(m_label, fontsize=10)
        ax.set_title(m_label, fontsize=11)
        ax.set_xticks(expert_ids)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SAC 多专家评估结果对比", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # ========== 3. 保存图片 ==========
    project_root = pathlib.Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 使用中文标题作为文件名
    filename = "SAC多专家评估结果对比"
    
    # 保存PNG格式（高分辨率位图）
    png_path = results_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    svg_path = results_dir / f"{filename}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"专家评估对比图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")


if __name__ == "__main__":
    main()
