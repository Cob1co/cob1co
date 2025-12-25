# -*- coding: utf-8 -*-
"""绘制微电网三阶段智能控制系统整体架构图

运行方式：
    python draw_system_architecture.py

运行后会在当前文件夹生成：
    system_architecture.png
    system_architecture.svg

说明：
    - 所有文字使用中文描述，字体为微软雅黑（Microsoft YaHei）
    - 仅展示系统主要模块与数据/信息流，方便汇报和论文讲解
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ===================== 全局字体与样式设置 =====================
# 统一使用微软雅黑（Windows下一般为 Microsoft YaHei）
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 14
plt.rcParams["mathtext.fontset"] = "stix"


# ===================== 基础绘图工具函数 =====================
def add_box(
    ax,
    xy,
    text,
    width=3.2,
    height=1.0,
    facecolor="#E8F1FF",
    edgecolor="#333333",
    fontsize=14,
):
    """添加带圆角的矩形框（中文说明用）"""
    x, y = xy
    x0 = x - width / 2.0
    y0 = y - height / 2.0

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.1,rounding_size=0.08",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#000000",
        wrap=True,
        zorder=3,
    )
    return (x, y, width, height)


def add_arrow(
    ax,
    start,
    end,
    text=None,
    style="->",
    connectionstyle="arc3,rad=0.0",
    color="#444444",
):
    """添加带箭头的连线，可选文字说明"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.8,
        color=color,
        connectionstyle=connectionstyle,
        alpha=0.9,
        zorder=1,
    )
    ax.add_patch(arrow)

    if text:
        mx = (start[0] + end[0]) / 2.0
        my = (start[1] + end[1]) / 2.0
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="#222222",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.2,
            ),
            zorder=4,
        )


def add_stage_box(ax, center, width, height, title, facecolor="#FFFFFF"):
    """绘制一个阶段大框（仅边框+标题，不填充），返回 (x, y, w, h)。"""
    x, y = center
    x0 = x - width / 2.0
    y0 = y - height / 2.0

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.15,rounding_size=0.12",
        edgecolor="#555555",
        facecolor=facecolor,
        linewidth=2.0,
        linestyle="-",
        zorder=0,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y + height / 2.0 + 0.2,
        title,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="#222222",
    )
    return (x, y, width, height)


# ===================== 主函数：绘制系统架构图 =====================

def main() -> None:
    # 画布加大，拉开留白
    fig, ax = plt.subplots(figsize=(13, 7.5))

    # 顶部：统一数据/配置
    p_data_cfg = add_box(
        ax,
        (0.0, 4.0),
        "数据与配置层\n(历史/实时数据、预测数据、物理与约束配置)",
        width=9.0,
        height=1.4,
        facecolor="#FFF2CC",
    )

    # 底部：物理系统
    p_physical = add_box(
        ax,
        (0.0, -4.3),
        "微电网物理系统\n(风电、光伏、熔盐储能、电网接口等)",
        width=9.0,
        height=1.4,
        facecolor="#F2F2F2",
    )

    # 中间：三个阶段大框
    stage_width = 5.4
    stage_height = 5.0

    s1 = add_stage_box(ax, (-6.0, -0.2), stage_width, stage_height, "第一阶段：容量规划与场景分析")
    s2 = add_stage_box(ax, (0.0, -0.2), stage_width, stage_height, "第二阶段：多专家强化学习训练")
    s3 = add_stage_box(ax, (6.0, -0.2), stage_width, stage_height, "第三阶段：Transformer + MPC 协同控制")

    # ---------- 阶段 1 内部小框 ----------
    add_box(
        ax,
        (s1[0], s1[1] + 1.4),
        "典型日场景聚类",
        width=4.2,
        height=0.9,
        facecolor="#E1D5E7",
    )
    add_box(
        ax,
        (s1[0], s1[1] + 0.1),
        "容量组合搜索与评估",
        width=4.2,
        height=0.9,
        facecolor="#DDEBF7",
    )
    add_box(
        ax,
        (s1[0], s1[1] - 1.2),
        "规则调度与运行成本分析",
        width=4.2,
        height=0.9,
        facecolor="#FCE4D6",
    )

    # ---------- 阶段 2 内部小框 ----------
    add_box(
        ax,
        (s2[0], s2[1] + 1.4),
        "微电网环境仿真",
        width=4.2,
        height=0.9,
        facecolor="#F8CECC",
    )
    add_box(
        ax,
        (s2[0], s2[1] + 0.1),
        "多专家 SAC 策略训练",
        width=4.2,
        height=0.9,
        facecolor="#F4CCCC",
    )
    add_box(
        ax,
        (s2[0], s2[1] - 1.2),
        "专家模型与归一化边界保存",
        width=4.2,
        height=0.9,
        facecolor="#FFE6CC",
    )

    # ---------- 阶段 3 内部小框 ----------
    add_box(
        ax,
        (s3[0], s3[1] + 1.8),
        "离线数据采集与 Oracle 权重",
        width=4.4,
        height=0.9,
        facecolor="#FCE4D6",
    )
    add_box(
        ax,
        (s3[0], s3[1] + 0.6),
        "特征提取与状态序列构造",
        width=4.4,
        height=0.9,
        facecolor="#E1D5E7",
    )
    add_box(
        ax,
        (s3[0], s3[1] - 0.6),
        "Transformer 权重控制器训练",
        width=4.4,
        height=0.9,
        facecolor="#DAE8FC",
    )
    add_box(
        ax,
        (s3[0], s3[1] - 1.8),
        "在线 Transformer 权重预测 + MPC 实时求解",
        width=4.4,
        height=0.9,
        facecolor="#D5E8D4",
    )

    # ---------- 跨阶段箭头（折线，避免横穿方框） ----------
    # 顶部数据 → 三个阶段（仍然是从上方直接进入阶段大框上边缘，不穿过内部）
    add_arrow(
        ax,
        (p_data_cfg[0] - 3.5, p_data_cfg[1] - 0.7),
        (s1[0] - stage_width / 4, s1[1] + stage_height / 2),
        "数据/配置",
    )
    add_arrow(
        ax,
        (p_data_cfg[0], p_data_cfg[1] - 0.7),
        (s2[0], s2[1] + stage_height / 2),
        "数据/配置",
    )
    add_arrow(
        ax,
        (p_data_cfg[0] + 3.5, p_data_cfg[1] - 0.7),
        (s3[0] + stage_width / 4, s3[1] + stage_height / 2),
        "数据/配置",
    )

    # 阶段 1 → 阶段 2：折线（先向下，再水平，再向上），不穿过阶段大框内部
    y_bus12 = s1[1] - stage_height / 2 - 0.4
    add_arrow(
        ax,
        (s1[0], s1[1] - stage_height / 2),
        (s1[0], y_bus12),
        style="-",
    )
    add_arrow(
        ax,
        (s1[0], y_bus12),
        (s2[0], y_bus12),
        "容量方案与运行场景",
    )
    add_arrow(
        ax,
        (s2[0], y_bus12),
        (s2[0], s2[1] - stage_height / 2),
        style="-",
    )

    # 阶段 2 → 阶段 3：同样使用折线，略微再向下，避免与上方箭头重叠
    y_bus23 = s2[1] - stage_height / 2 - 0.7
    add_arrow(
        ax,
        (s2[0], s2[1] - stage_height / 2),
        (s2[0], y_bus23),
        style="-",
    )
    add_arrow(
        ax,
        (s2[0], y_bus23),
        (s3[0], y_bus23),
        "多专家策略与专家接口",
    )
    add_arrow(
        ax,
        (s3[0], y_bus23),
        (s3[0], s3[1] - stage_height / 2),
        style="-",
    )

    # 第三阶段 → 物理系统（保持从阶段 3 底部斜向进入物理系统，不穿过其他阶段）
    add_arrow(
        ax,
        (s3[0], s3[1] - stage_height / 2),
        (p_physical[0] + 2.0, p_physical[1] + 0.7),
        "实时控制指令",
    )

    # 物理系统 → 数据配置：沿左侧绕行的折线，完全不穿过三个阶段大框
    x_side = -8.9
    # 先从物理系统左侧拉出一小段水平线
    add_arrow(
        ax,
        (p_physical[0] - 4.0, p_physical[1] + 0.7),
        (x_side, p_physical[1] + 0.7),
        style="-",
    )
    # 再沿左侧竖直向上，并在这条线上加箭头和文字
    add_arrow(
        ax,
        (x_side, p_physical[1] + 0.7),
        (x_side, p_data_cfg[1] - 0.7),
        "运行数据回流",
    )
    # 最后一小段水平线接入到数据与配置层的左下侧
    add_arrow(
        ax,
        (x_side, p_data_cfg[1] - 0.7),
        (p_data_cfg[0] - 4.0, p_data_cfg[1] - 0.7),
        style="-",
    )

    # 画布美化
    ax.set_xlim(-9.0, 9.0)
    ax.set_ylim(-5.5, 5.2)
    ax.axis("off")
    plt.tight_layout()

    # 保存图片
    plt.savefig(
        "system_architecture.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    try:
        plt.savefig(
            "system_architecture.svg",
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print("系统架构图已保存为:")
        print("  - system_architecture.png (高分辨率PNG)")
        print("  - system_architecture.svg (矢量图SVG)")
    except Exception as e:  # noqa: BLE001
        print(f"SVG 格式保存失败: {e}")
        print("系统架构图仅保存为 system_architecture.png")

    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
