"""Phase 1 收敛曲线绘图脚本

功能：
- 参考 SSA_compare.m 的风格，对比两种麻雀搜索算法（SSA）配置
  （基线 SSA vs 引入边际效益修正的改进 SSA）的收敛过程；
- X 轴为迭代次数，Y 轴为单目标函数值（Fitness / J(x)，越小越好）；
- 生成适用于论文和 PPT 的收敛曲线图。

使用方法：
1. 根据实际结果，填写 `obj_baseline` 和 `obj_improved` 两个数组；
2. 在项目根目录（重构版）下运行：
   python Fig_FK/plot_phase1_convergence.py
3. 图片将保存在 Fig_FK/phase1_convergence.png。
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # ========== 1. 手动填写不同算法下的收敛数据 ==========
    # 说明：
    # - 建议使用“每次迭代的全局最优 J(x)”作为纵轴数据；
    # - 如果暂时只有一种算法，也可以只画一条曲线；
    # - 下面是占位示例，请替换为你真实的计算结果。

    # 这里构造 500 次迭代的示例数据：
    # - 两条曲线统一起点 common_start，便于直接比较收敛速度和最终适应度；
    # - 基线 SSA：收敛较慢，最终停在更高的适应度值；
    # - 改进 SSA：收敛更快，最终适应度更优。

    max_iter = 500
    iters = np.arange(1, max_iter + 1)

    common_start = 4200.0

    # 终值与收敛速率（仅用于 PPT 演示，可按需要微调）
    ssa_val = 940.0
    ssa_rate = 0.09
    issa_val = 870.0
    issa_rate = 0.12

    obj_ssa = ssa_val + (common_start - ssa_val) * np.exp(-ssa_rate * (iters - 1))
    obj_issa = issa_val + (common_start - issa_val) * np.exp(-issa_rate * (iters - 1))

    # ========== 2. 绘图参数设置（使用中文字体保证坐标轴与标题正常显示） ==========
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用中文黑体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    # 基线 SSA 曲线
    ax.plot(
        iters,
        obj_ssa,
        color="tab:blue",
        linestyle="-",
        linewidth=1.5,
        label="SSA",
    )

    # 改进 SSA 曲线
    ax.plot(
        iters,
        obj_issa,
        color="tab:red",
        linestyle="-",
        linewidth=2.0,
        label="ISSA",
    )

    # 坐标轴与标题（英文标签，更接近论文风格）
    ax.set_xlabel("迭代", fontsize=12)
    ax.set_ylabel("适应度", fontsize=12)
    ax.set_title("算法性能对比", fontsize=13)

    # Y 轴范围适当放大一些，保证起点和收敛段都清晰
    ymin = min(obj_issa.min(), obj_ssa.min()) - 50.0
    ymax = common_start + 200.0
    ax.set_ylim(max(0.0, ymin), ymax)
    ax.set_xlim(0, max_iter)

    # 开放坐标系：移除上/右边框，只保留下/左边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # 关闭网格，刻度线向内，略微加长，符合论文风格
    ax.grid(False)
    ax.tick_params(direction="in", length=4, width=1.0)

    ax.legend(loc="upper right", fontsize=10, frameon=False)

    fig.tight_layout()

    # ========== 3. 保存图片 ==========
    project_root = pathlib.Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 使用中文标题作为文件名
    filename = "算法性能对比"
    
    # 保存PNG格式（高分辨率位图）
    png_path = results_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    svg_path = results_dir / f"{filename}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"收敛曲线图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")


if __name__ == "__main__":
    main()
