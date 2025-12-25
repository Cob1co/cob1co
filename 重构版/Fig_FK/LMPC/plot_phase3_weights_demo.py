"""第三阶段机制演示图：动态权重随时间变化

用于 PPT 中间部分，打开"黑盒"，展示 Transformer 是如何根据电价和工况
自动调整三个优化目标权重的。

数据为构造的典型日示意数据。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

def plot_weights_demo() -> None:
    # 1. 设置风格
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 2. 构造时间轴 (06:00 - 22:00, 15min一个点)
    hours = np.arange(6, 22.25, 0.25)
    n_steps = len(hours)

    # 3. 构造环境数据 (电价 & 光伏)
    # 电价：真正的15分钟跳变 + 小幅随机波动
    price = np.zeros(n_steps)
    # 基础分时段电价 + 15分钟波动
    for i, h in enumerate(hours):
        # 基础电价
        if 8 <= h < 11:    # 早高峰
            base_price = 1.2
        elif 11 <= h < 18: # 平峰
            base_price = 0.6
        elif 18 <= h < 21: # 晚高峰
            base_price = 1.1
        else:               # 谷电
            base_price = 0.3
        
        # 每15分钟添加小幅波动 (±0.1)
        fluctuation = np.random.uniform(-0.1, 0.1)
        price[i] = max(0.2, base_price + fluctuation)  # 确保电价不低于0.2
    
    # 光伏：真正的15分钟变化 + 云层遮挡效应
    pv_profile = np.zeros(n_steps)
    for i, h in enumerate(hours):
        # 基础光伏出力
        if 11 <= h <= 14:
            base_pv = 0.9
        elif 9 <= h < 11 or 14 < h <= 16:
            base_pv = 0.5
        else:
            base_pv = 0.05
        
        # 每15分钟添加云层遮挡波动 (±0.15)
        cloud_fluctuation = np.random.uniform(-0.15, 0.15)
        pv_profile[i] = max(0.05, min(1.0, base_pv + cloud_fluctuation))
    
    # 4. 构造 Transformer 权重 (每15分钟重新决策一次)
    # 基准值
    alpha_cost = np.ones(n_steps) * 1.0
    alpha_soc  = np.ones(n_steps) * 1.0  # 对应消纳/弃电权重
    alpha_grid = np.ones(n_steps) * 1.2  # 平时偏向稳定
    
    # 逻辑1: 电价高时，降低成本权重 (平衡其他目标)
    for i, p in enumerate(price):
        if p > 1.0:  # 高电价时，成本项已很大，降低权重
            alpha_cost[i] = 1.0 - 0.3 * (p - 1.0)
        elif p < 0.5:  # 低电价时，可以更关注成本优化
            alpha_cost[i] = 1.0 + 0.5 * (0.5 - p)
        else:  # 正常电价
            alpha_cost[i] = 1.0
        
        # 添加小幅随机扰动
        noise = np.random.uniform(-0.05, 0.05)
        alpha_cost[i] += noise
    
    # 逻辑2: 光伏大时，狠抓消纳 (alpha_soc 连续响应)
    for i, pv in enumerate(pv_profile):
        # 连续响应光伏出力，而不是阈值判断
        # 光伏出力越高，消纳权重越大
        alpha_soc[i] += 1.5 * pv  # 线性响应
        
        # 添加小幅随机扰动模拟其他因素
        noise = np.random.uniform(-0.05, 0.05)
        alpha_soc[i] += noise
    
    # 逻辑3: 电网波动权重 - 基于电价变化率的连续调节
    for i in range(n_steps):
        # 计算电价变化率 (15分钟梯度)
        if i == 0:
            price_change = 0
        else:
            price_change = abs(price[i] - price[i-1])
        
        # 基础波动保护 + 变化率响应
        base_grid = 1.2
        if price_change > 0.15:  # 电价变化较大
            alpha_grid[i] = base_grid + 0.8 * price_change
        elif price_change > 0.08:  # 电价中等变化
            alpha_grid[i] = base_grid + 0.4 * price_change
        else:  # 电价平稳
            alpha_grid[i] = base_grid + 0.1
        
        # 添加小幅随机扰动模拟其他因素影响
        noise = np.random.uniform(-0.1, 0.1)
        alpha_grid[i] += noise

    # 归一化/限幅到配置文件范围 (0.5 - 2.0)
    alpha_cost = np.clip(alpha_cost, 0.5, 2.0)
    alpha_soc  = np.clip(alpha_soc, 0.5, 2.0)
    alpha_grid = np.clip(alpha_grid, 0.5, 2.0)

    # 5. 绘图
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # --- 右轴：画电价背景 (阶梯状) ---
    ax2 = ax1.twinx()
    # 用阶梯图表示电价，每15分钟跳变一次
    ax2.step(hours, price, where='post', color='gray', alpha=0.5, linewidth=1.5, label='实时电价 (右轴)')
    ax2.fill_between(hours, 0, price, step='post', color='gray', alpha=0.15)
    ax2.set_ylabel("电价 (元/kWh)", fontsize=11, color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax2.set_ylim(0, 1.5)
    
    # --- 左轴：画动态权重 (阶梯状) ---
    # 运行成本权重
    l1, = ax1.step(hours, alpha_cost, where='post', color='#d62728', linewidth=2.5, label='运行成本权重 $\\alpha_{cost}$')
    # 风光消纳权重
    l2, = ax1.step(hours, alpha_soc, where='post', color='#2ca02c', linewidth=2.5, label='风光消纳权重 $\\alpha_{soc}$')
    # 电网波动权重
    l3, = ax1.step(hours, alpha_grid, where='post', color='#1f77b4', linewidth=2.0, linestyle='--', label='电网波动权重 $\\alpha_{grid}$')
    
    ax1.set_xlabel("时间 (小时)", fontsize=12)
    ax1.set_ylabel("Transformer 动态权重值", fontsize=12)
    ax1.set_ylim(0.4, 2.8)
    ax1.set_xlim(6, 22)
    
    # 设置x轴刻度
    ax1.set_xticks(np.arange(6, 23, 2))
    ax1.set_xticklabels([f"{int(h)}:00" for h in np.arange(6, 23, 2)])

    # 6. 添加解释性标注 (Storytelling)
    # 标注1: 电价高峰
    ax1.annotate('电价高峰：\n成本权重降低', 
                 xy=(9.5, 2.2), xytext=(7.5, 2.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, color='#d62728')
    
    # 标注2: 光伏高峰
    ax1.annotate('光伏大发：\n消纳权重自动升高', 
                 xy=(13, 2.3), xytext=(13, 2.6),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, color='#2ca02c', ha='center')
    
    # 标注3: 权重跳变点
    ax1.annotate('电价跳变：\n波动权重临时升高', 
                 xy=(18, 2.0), xytext=(16.5, 2.3),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, color='#1f77b4')

    # 图例合并
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    # 加上电价的图例 (虽然是 fill_between, 搞个假的 proxy artist)
    proxy_rect = Rectangle((0, 0), 1, 1, fc="gray", alpha=0.15)
    lines.append(proxy_rect)
    labels.append("实时电价 (背景)")
    
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
    
    ax1.set_title("Transformer 动态权重随工况自适应调整机制", y=1.15, fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 7. 保存
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 使用中文标题作为文件名
    filename = "Transformer动态权重随工况自适应调整机制"
    
    plt.tight_layout()
    
    # 保存PNG格式（高分辨率位图）
    png_path = results_dir / f"{filename}.png"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    svg_path = results_dir / f"{filename}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"Transformer动态权重调整机制图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")

if __name__ == "__main__":
    plot_weights_demo()
