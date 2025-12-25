# -*- coding: utf-8 -*-
"""绘制 SAC 多专家训练结构图

运行方式：
    python draw_sac_training_structure.py

运行后会在当前文件夹生成：
    sac_training_structure.png
    sac_training_structure.svg

说明：
    - 风格与 draw_flowchart_summary.py、draw_transformer_paper_style.py 保持一致
    - 中文使用仿宋_GB2312，英文使用 Times New Roman，中英分行显示
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties


# ===================== 字体与文本工具 =====================
# 中文：仿宋_GB2312，英文：Times New Roman
FONT_CN_BASE = FontProperties(family="FangSong_GB2312")
FONT_EN_BASE = FontProperties(family="Times New Roman")


def _font_with_size(base_font: FontProperties, size: float) -> FontProperties:
    font = base_font.copy()
    font.set_size(size)
    return font


def _normalize_lines(content):
    if not content:
        return []
    if isinstance(content, (list, tuple)):
        lines = []
        for item in content:
            lines.extend(_normalize_lines(item))
        return lines
    return [line.strip() for line in str(content).split("\n") if line.strip()]


def draw_multilingual_text(
    ax,
    x: float,
    y: float,
    *,
    text_cn=None,
    text_en=None,
    fontsize: float = 18,
    ha: str = "center",
    va: str = "center",
    line_gap: float = 0.35,
    **kwargs,
) -> None:
    """按行绘制中英文文本：中文在上，英文在下，字体分别设置。"""
    cn_lines = [(line, FONT_CN_BASE) for line in _normalize_lines(text_cn)]
    en_lines = [(line, FONT_EN_BASE) for line in _normalize_lines(text_en)]
    lines = cn_lines + en_lines
    if not lines:
        return

    total = len(lines)
    offset = line_gap * (total - 1) / 2
    for idx, (content, font_base) in enumerate(lines):
        font = _font_with_size(font_base, fontsize)
        ax.text(
            x,
            y + offset - idx * line_gap,
            content,
            ha=ha,
            va=va,
            fontproperties=font,
            **kwargs,
        )


# 设置论文字体格式
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["FangSong_GB2312"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"


def add_box(
    ax,
    xy,
    text=None,
    *,
    text_cn=None,
    text_en=None,
    width=3.0,
    height=0.9,
    facecolor="#E8F1FF",
    fontsize=16,
    edgecolor="#333333",
    line_gap=0.32,
):
    """添加带圆角的矩形框，支持中英文分别设置。"""
    x, y = xy
    x0 = x - width / 2.0
    y0 = y - height / 2.0

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=2,
    )
    ax.add_patch(box)

    if text_cn is None and text_en is None and text is not None:
        text_cn = text

    draw_multilingual_text(
        ax,
        x,
        y,
        text_cn=text_cn,
        text_en=text_en,
        fontsize=fontsize,
        ha="center",
        va="center",
        line_gap=line_gap,
        color="#000000",
    )
    return (x, y, width, height)


def add_arrow(
    ax,
    start,
    end,
    *,
    text=None,
    text_cn=None,
    text_en=None,
    style="->",
    connectionstyle="arc3,rad=0.2",
    color="#444444",
    fontsize=18,
):
    """添加箭头，并可选添加中英文标签。"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=15,
        linewidth=3,
        color=color,
        connectionstyle=connectionstyle,
        alpha=0.8,
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(arrow)

    if text_cn is None and text_en is None and text is not None:
        text_cn = text

    if text_cn or text_en:
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        draw_multilingual_text(
            ax,
            mid_x,
            mid_y,
            text_cn=text_cn,
            text_en=text_en,
            fontsize=fontsize,
            ha="center",
            va="center",
            line_gap=0.3,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.1,
            ),
        )


# ===================== 主函数：绘制 SAC 训练结构图 =====================


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    # 坐标分层（自上而下）：数据与配置 → 交互与经验 → 专家输出
    y_data = 3.4       # 数据与配置层
    y_core = 1.4       # 交互与经验层
    y_output = -2.6    # 专家输出

    # === 1. 数据与配置层（合并为一个大框，减少连线） ===
    p_data_cfg = add_box(
        ax,
        (0.0, y_data),
        text_cn=[
            "训练数据与配置",
            "典型日/天气场景，历史/预测运行数据，物理参数与约束",
        ],
        text_en="Scenarios / Data / Params",
        facecolor="#E1D5E7",
        width=6.0,
        height=1.0,
    )

    # === 2. 核心交互与经验采集层 ===
    p_env = add_box(
        ax,
        (-4.0, y_core),
        text_cn=["微电网环境仿真", "状态/奖励生成"],
        text_en="Microgrid Env",
        facecolor="#F8CECC",
        width=3.4,
        height=1.0,
    )

    p_buffer = add_box(
        ax,
        (0.0, y_core),
        text_cn=["经验回放缓冲区", "(s, a, r, s')"],
        text_en="Replay Buffer",
        facecolor="#FFF2CC",
        width=3.4,
        height=1.0,
    )

    p_agent = add_box(
        ax,
        (4.0, y_core),
        text_cn=["SAC 智能体", "策略/价值网络"],
        text_en="SAC Agent",
        facecolor="#E2F0D9",
        width=3.4,
        height=1.0,
    )

    # === 3. 专家输出与接口层 ===
    p_expert_multi = add_box(
        ax,
        (-2.3, y_output),
        text_cn=["多专家 SAC 策略", "按典型日/天气划分"],
        text_en="Multi-expert policies",
        facecolor="#FCE4D6",
        width=3.8,
        height=1.0,
    )

    p_expert_if = add_box(
        ax,
        (2.3, y_output),
        text_cn=["专家池与接口模块", "供后续协同控制调用"],
        text_en="Expert pool & interface",
        facecolor="#FFCCBC",
        width=4.2,
        height=1.0,
    )

    p_phase3 = add_box(
        ax,
        (5.0, y_output - 1.2),
        text_cn="第三阶段：协同控制",
        text_en="TD-MPC / LMPC",
        facecolor="#DAE8FC",
        width=3.6,
        height=0.9,
    )

    # === 4. 箭头与数据流 ===
    # 4.1 数据/参数 → 环境（合并为一条主线）
    add_arrow(
        ax,
        (p_data_cfg[0], p_data_cfg[1] - 0.6),
        (p_env[0], p_env[1] + 0.6),
        text_cn="驱动环境与奖励构造",
        text_en="Env inputs",
        connectionstyle="arc3,rad=0.0",
    )

    # 4.2 SAC 交互环：Agent → Env → Buffer → Agent
    add_arrow(
        ax,
        (p_agent[0] - 1.7, p_agent[1]),
        (p_env[0] + 1.7, p_env[1]),
        text_cn="动作 a",
        text_en="Action",
        connectionstyle="arc3,rad=0.0",
    )

    # 经验：沿盒子底部水平传递，避免斜线
    add_arrow(
        ax,
        (p_env[0] + 1.7, p_env[1] - 0.5),
        (p_buffer[0] - 1.7, p_buffer[1] - 0.5),
        text_cn="经验 (s, a, r, s')",
        text_en="Transitions",
        connectionstyle="arc3,rad=0.0",
    )

    # Mini-batch：从经验池到智能体，同样沿底边水平传递
    add_arrow(
        ax,
        (p_buffer[0] + 1.7, p_buffer[1] - 0.5),
        (p_agent[0] - 1.7, p_agent[1] - 0.5),
        text_cn="采样 batch / 参数更新",
        text_en="Mini-batch & update",
        connectionstyle="arc3,rad=0.0",
    )

    # 训练完成的专家输出
    add_arrow(
        ax,
        (p_agent[0], p_agent[1] - 0.55),
        (p_expert_multi[0] + 1.4, p_expert_multi[1] + 0.55),
        text_cn="不同场景下训练的专家策略",
        text_en="Trained experts",
        connectionstyle="arc3,rad=0.0",
    )

    add_arrow(
        ax,
        (p_expert_multi[0] + 1.9, p_expert_multi[1]),
        (p_expert_if[0] - 1.9, p_expert_if[1]),
        text_cn="保存/注册到专家池",
        text_en="Save & register",
        connectionstyle="arc3,rad=0.0",
    )

    add_arrow(
        ax,
        (p_expert_if[0] + 2.1, p_expert_if[1] - 0.2),
        (p_phase3[0] - 1.8, p_phase3[1] + 0.2),
        text_cn="为协同控制提供专家参考",
        text_en="Expert reference",
        connectionstyle="arc3,rad=0.0",
    )

    # 布局与保存
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-4.5, 4.5)
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(
        "sac_training_structure.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    try:
        plt.savefig(
            "sac_training_structure.svg",
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print("SAC 训练结构图已保存为:")
        print("  - sac_training_structure.png (高分辨率PNG)")
        print("  - sac_training_structure.svg (矢量图SVG)")
    except Exception as e:  # noqa: BLE001
        print(f"SVG 格式保存失败: {e}")
        print("SAC 训练结构图仅保存为 sac_training_structure.png")

    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
