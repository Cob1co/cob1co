"""第二阶段 SAC 调度结果 - 论文风格柱状对比图

功能：
- 模仿 Fig_FK/论文可视化/cost_analysis.m 的标准论文风格；
- 对比「传统规则调度」与「SAC 智能调度」在多个指标上的相对表现；
- 使用归一化百分比（相对传统调度 = 100%），便于在一张图中放不同量纲的指标；
- 默认生成 Fig_FK/phase2_sac_results_bar.png，直接用于 PPT / 论文。

说明：
- 当前数值为示例数据，用于展示“成本下降、弃电大幅降低、波动减小”的趋势；
- 建议你根据实际年成本、年弃电量、电网功率波动等计算百分比后，替换下面的常量。
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 指标与示例数据（可根据实际结果修改） =====================

# 指标名称（横轴）
CATEGORIES = ["运行成本", "弃电量", "净电荷波动"]

# 传统调度作为基准：统一归一化为 100%
BASELINE_PERCENT = 100.0
BASELINE_VALUES = np.array([
    BASELINE_PERCENT,  # 运行成本
    BASELINE_PERCENT,  # 弃电量
    BASELINE_PERCENT,  # 电网波动
])

# SAC 调度相对传统调度的百分比（示例值：可以按实际结果改）
SAC_COST_PERCENT = 72.0   # 成本降到 72%
SAC_CURTAIL_PERCENT = 35.0  # 弃电量大幅降到 35%
SAC_RAMP_PERCENT = 68.0   # 波动降到 68%

SAC_VALUES = np.array([
    SAC_COST_PERCENT,
    SAC_CURTAIL_PERCENT,
    SAC_RAMP_PERCENT,
])


def main() -> None:
    # ===================== 2. Matplotlib 全局风格设置 =====================
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文黑体，保证标签正常
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # 图尺寸与背景（参考 MATLAB: 500x350 左右）
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)

    x = np.arange(len(CATEGORIES))
    bar_width = 0.35

    # ===================== 3. 绘制柱状图 =====================
    bars_baseline = ax.bar(
        x - bar_width / 2,
        BASELINE_VALUES,
        bar_width,
        label="SSA启发式",
        color=(0.85, 0.33, 0.33),  # 略偏红
        edgecolor=(0.2, 0.2, 0.2),
        linewidth=1.2,
    )

    bars_sac = ax.bar(
        x + bar_width / 2,
        SAC_VALUES,
        bar_width,
        label="SAC强化学习",
        color=(0.33, 0.66, 0.33),  # 略偏绿
        edgecolor=(0.2, 0.2, 0.2),
        linewidth=1.2,
    )

    # 横轴刻度与标签
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=11)

    # 纵轴范围：略高于最大值，留出数值标注空间
    ymax = max(BASELINE_VALUES.max(), SAC_VALUES.max()) * 1.18
    ax.set_ylim(0.0, ymax)

    # 纵轴标签
    ax.set_ylabel("相对启发式调度（%）", fontsize=12)

    # ===================== 4. 论文风格：开放坐标系、刻度向内 =====================
    ax.grid(False)  # 纯白背景

    # 只保留下/左坐标轴，线稍微加粗
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # 刻度线向内
    ax.tick_params(direction="in", length=4, width=1.0)

    # 图例（无边框），放在图上方居中，完全不占用绘图区
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=10,
        frameon=False,
    )

    # 在柱子上方标注具体百分比
    def _annotate_bars(bars, values) -> None:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + ymax * 0.02,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _annotate_bars(bars_baseline, BASELINE_VALUES)
    _annotate_bars(bars_sac, SAC_VALUES)

    # 标题可根据 PPT 需要自行修改
    ax.set_title("", fontsize=12)

    fig.tight_layout()

    # ===================== 5. 保存图片 =====================
    project_root = pathlib.Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 使用中文标题作为文件名
    filename = "SAC调度结果柱状图"
    
    # 保存PNG格式（高分辨率位图）
    png_path = results_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    svg_path = results_dir / f"{filename}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"SAC调度结果柱状图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")


if __name__ == "__main__":
    main()
