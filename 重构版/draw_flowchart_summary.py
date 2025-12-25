# -*- coding: utf-8 -*-
"""
绘制风光电熔盐储能微电网三阶段智能控制总体流程图（项目总结版）

运行方式：
    python draw_flowchart_summary.py

运行后会在当前文件夹生成：
    microgrid_flowchart_summary.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties


# ===================== 字体与文本工具 =====================
FONT_CN_BASE = FontProperties(family='FangSong_GB2312')
FONT_EN_BASE = FontProperties(family='Times New Roman')


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


def draw_multilingual_text(ax,
                           x: float,
                           y: float,
                           text_cn=None,
                           text_en=None,
                           fontsize: float = 18,
                           ha: str = "center",
                           va: str = "center",
                           line_gap: float = 0.35,
                           **kwargs) -> None:
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
# 中文：仿宋_GB2312，英文：Times New Roman
# 使用sans-serif确保中文字体正确显示，英文单独设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['FangSong_GB2312']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.size'] = 18  # 调整为14pt，更清晰
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用STIX字体

# Times New Roman字体属性（用于纯英文文本）
from matplotlib.font_manager import FontProperties
TNR_FONT = FontProperties(family='Times New Roman', size=18)

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
    fontsize=15,
    edgecolor="#333333",
    line_gap=0.32,
):
    """添加带圆角的矩形框，支持中英文分别设置"""
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

    if text_cn is None and text_en is None and text:
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
    """
    添加箭头，并可选添加标签
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=15,
        linewidth=3,
        color=color,
        connectionstyle=connectionstyle,
        alpha=0.8,
        shrinkA=8,  # 增加缩进量，使箭头两端远离方框
        shrinkB=8   # 增加缩进量，使箭头两端远离方框
    )
    ax.add_patch(arrow)
    
    if text_cn is None and text_en is None and text:
        text_cn = text
    
    if text_cn or text_en:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
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
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.1)
        )

def main():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 坐标体系
    # 左侧：数据与场景层 (x=-4)
    # 中间：核心处理层 (x=0)
    # 右侧：模型闭环层 (x=4)
    
    # y轴分层
    y_data = 4       # 物理与数据层
    y_phase1 = 2     # 第一阶段
    y_phase2 = 0     # 第二阶段
    y_phase3_train = -2 # 第三阶段（训练）
    y_phase3_run = -4   # 第三阶段（运行）

    # === 1. 物理与数据层 ===
    p_data = add_box(
        ax,
        (-4, y_data),
        text_cn="历史/预测数据",
        text_en="(Load/Solar/Wind/Price)",
        facecolor="#E1D5E7",
        width=2.8,
        height=1.0,
    )
    
    p_phy = add_box(
        ax,
        (0, y_data),
        text_cn=["物理参数库", "容量/效率/损耗"],
        facecolor="#E1D5E7",
        width=2.6,
        height=1.0,
    )
                   
    p_cluster = add_box(
        ax,
        (-4, y_phase1),
        text_cn="典型日场景聚类",
        text_en="(K-means)",
        facecolor="#FFF2CC",
        width=2.6,
        height=0.9,
    )

    # === 2. 第一阶段 (SA-MBPS) ===
    p_phase1 = add_box(
        ax,
        (0, y_phase1),
        text_cn=["第一阶段：容量规划", "启发式优化+规则调度"],
        facecolor="#DDEBF7",
        width=3.0,
        height=1.0,
    )

    # === 3. 第二阶段 (SAC) ===
    p_phase2 = add_box(
        ax,
        (0, y_phase2),
        text_cn=["第二阶段：多专家", "典型日分别训练"],
        text_en=["SAC"],
        facecolor="#E2F0D9",
        width=3.0,
        height=1.0,
    )
    
    p_expert_if = add_box(
        ax,
        (0, y_phase3_train),
        text_cn=["专家接口", "生成参考轨迹"],
        facecolor="#E2F0D9",
        width=2.4,
        height=0.8,
    )

    # === 4. 第三阶段 (Transformer+MPC) ===
    p_classifier = add_box(
        ax,
        (-4, y_phase3_train),
        text_cn="场景识别\n(匹配)",
        text_en=[ "K-means"],
        facecolor="#FFF2CC",
        width=2.8,
        height=0.9,
    )

    p_collector = add_box(
        ax,
        (4, y_phase2),
        text_cn=["离线数据采集", "网格搜索最优权重"],
        facecolor="#FCE4D6",
        width=2.8,
        height=1.0,
    )

    p_trans_train = add_box(
        ax,
        (4, y_phase3_train),
        text_cn="权重模型训练",
        text_en="State → Weights",
        facecolor="#FCE4D6",
        width=2.8,
        height=0.9,
    )

    p_mpc = add_box(
        ax,
        (0, y_phase3_run),
        text_cn=["第三阶段运行：协同控制", "动态平衡：专家经验与实时优化"],
        text_en="TD-MPC",
        facecolor="#FFCCBC",
        width=5.0,
        height=1.0,
    )

    # ========== 连线与数据流 ==========

    # 1. 物理参数下发
    add_arrow(
        ax,
        (p_phy[0], p_phy[1]-0.5),
        (p_phase1[0], p_phase1[1]+0.5),
        text_cn="参数",
        connectionstyle="arc3,rad=0",
    )
    add_arrow(
        ax,
        (p_phy[0]+1.3, p_phy[1]),
        (p_collector[0], p_collector[1]+0.5),
        style="-",
        connectionstyle="arc3,rad=-0.2",
    )
             
    # 修改：物理参数 -> SAC (从右侧绕过 Phase 1)
    add_arrow(
        ax,
        (p_phy[0]+1.3, p_phy[1]-0.2),
        (p_phase2[0]+1.5, p_phase2[1]+0.2),
        style="->",
        connectionstyle="arc3,rad=-0.5",
    )

    # 2. 数据流向
    add_arrow(
        ax,
        (p_data[0], p_data[1]-0.5),
        (p_cluster[0], p_cluster[1]+0.45),
        connectionstyle="arc3,rad=0",
    )
    add_arrow(
        ax,
        (p_data[0]+1.4, p_data[1]),
        (p_phase1[0]-1.5, p_phase1[1]),
        text_cn="驱动",
        connectionstyle="arc3,rad=0",
    )

    # 第一阶段 -> 第二阶段
    add_arrow(
        ax,
        (p_phase1[0], p_phase1[1]-0.55),
        (p_phase2[0], p_phase2[1]+0.45),
        text_cn="容量输出",
        connectionstyle="arc3,rad=0",
    )

    # 3. 聚类结果应用
    add_arrow(
        ax,
        (p_cluster[0]+1.3, p_cluster[1]),
        (p_phase1[0]-1.5, p_phase1[1]-0.2),
        text_cn="典型日",
        connectionstyle="arc3,rad=0",
    )
    # 修改：将起点下移至 -0.55，避免压到方框底部
    add_arrow(
        ax,
        (p_cluster[0], p_cluster[1]-0.55),
        (p_phase2[0]-1.5, p_phase2[1]),
        text_cn="划分",
        connectionstyle="arc3,rad=-0.1",
    )
    add_arrow(
        ax,
        (p_cluster[0], p_cluster[1]-0.55),
        (p_classifier[0], p_classifier[1]+0.45),
        text_cn="复用",
        connectionstyle="arc3,rad=0",
    )

    # 4. 专家策略流
    add_arrow(
        ax,
        (p_phase2[0], p_phase2[1]-0.5),
        (p_expert_if[0], p_expert_if[1]+0.4),
        text_cn="模型",
        connectionstyle="arc3,rad=0",
    )
    add_arrow(
        ax,
        (p_expert_if[0], p_expert_if[1]-0.4),
        (p_mpc[0], p_mpc[1]+0.5),
        text_cn="轨迹",
        connectionstyle="arc3,rad=0",
    )

    # 5. 场景识别流
    add_arrow(
        ax,
        (p_classifier[0]+1.4, p_classifier[1]),
        (p_expert_if[0]-1.2, p_expert_if[1]),
        text_en="ID",
        connectionstyle="arc3,rad=0",
    )

    # 6. 离线训练闭环
    # 专家接口 -> 数据采集
    add_arrow(
        ax,
        (p_expert_if[0]+1.2, p_expert_if[1]),
        (p_collector[0]-1.4, p_collector[1]-0.3),
        text_cn="参考",
        connectionstyle="arc3,rad=0",
    )
    # 采集 -> 训练
    add_arrow(
        ax,
        (p_collector[0], p_collector[1]-0.5),
        (p_trans_train[0], p_trans_train[1]+0.45),
        text_en="Data",
        connectionstyle="arc3,rad=0",
    )
    # 训练 -> MPC
    add_arrow(
        ax,
        (p_trans_train[0], p_trans_train[1]-0.45),
        (p_mpc[0]+2.5, p_mpc[1]+0.5),
        text_cn="权重",
        connectionstyle="arc3,rad=-0.2",
    )

    # === 布局美化 ===
    # 缩小四周空白，紧凑显示内容
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-4.8, 4.8)
    ax.axis("off")
    
    plt.tight_layout()
    
    # 保存PNG格式（高分辨率位图）
    plt.savefig("microgrid_flowchart_summary.png", dpi=600, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    try:
        plt.savefig("microgrid_flowchart_summary.svg", format='svg', bbox_inches="tight",
                    facecolor='white', edgecolor='none')
        print("项目总结版流程图已保存为:")
        print("  - microgrid_flowchart_summary.png (高分辨率PNG)")
        print("  - microgrid_flowchart_summary.svg (矢量图SVG)")
        print("\n注意：SVG格式可在Word中插入并转换为EMF格式")
    except Exception as e:
        print(f"SVG格式保存失败: {e}")
        print("项目总结版流程图已保存为 microgrid_flowchart_summary.png")
    
    plt.close(fig)

if __name__ == "__main__":
    main()
