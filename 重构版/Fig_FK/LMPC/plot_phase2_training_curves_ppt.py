"""第二阶段 SAC 训练过程曲线（PPT 友好版）

功能：
- 从 logs/expert_0/training_log.json 读取 Expert 0 训练日志；
- 对 episode 回报、成本、弃电量做滑动平均，只绘制平滑曲线；
- 生成更干净、适合 PPT 展示的三行子图：
  1) 平均回报 vs episode
  2) 平均成本 vs episode
  3) 平均弃电量 vs episode

输出：
- Fig_FK/phase2_training_curves_ppt.png

说明：
- 当前脚本默认读取 expert_0 的日志；如需其它专家，可复制脚本并修改日志路径；
- 滑动窗口大小可根据需要调整（默认 25 个 episode）。
"""

import json
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# ===================== 1. 平滑工具函数 =====================

def moving_average(data: Iterable[float], window: int) -> np.ndarray:
    """简单滑动平均，返回与原序列等长的平滑结果。

    使用卷积对中间部分进行平滑，首尾用原值填充，避免长度变化。
    """

    arr = np.asarray(list(data), dtype=float)
    if arr.size == 0 or window <= 1:
        return arr

    window = min(window, arr.size)
    kernel = np.ones(window, dtype=float) / float(window)
    # 有效卷积部分
    valid = np.convolve(arr, kernel, mode="valid")

    # 还原到原始长度：
    pad_left = arr[: window // 2]
    pad_right = arr[-(window - window // 2 - 1) :]
    smooth = np.concatenate([pad_left, valid, pad_right])
    # 处理极端长度差异
    if smooth.size > arr.size:
        smooth = smooth[: arr.size]
    elif smooth.size < arr.size:
        smooth = np.pad(smooth, (0, arr.size - smooth.size), mode="edge")
    return smooth


# ===================== 2. 主绘图函数 =====================

def main() -> None:
    # 脚本在 Fig_FK/LMPC/ 目录下，需要回到项目根目录（重构版）
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    log_path = project_root / "logs" / "expert_0" / "training_log.json"

    if not log_path.exists():
        print("未找到日志文件: logs/expert_0/training_log.json")
        return

    with log_path.open("r", encoding="utf-8") as f:
        log = json.load(f)

    episodes = np.asarray(log.get("episode", []), dtype=int)
    returns = np.asarray(log.get("return", []), dtype=float)
    costs = np.asarray(log.get("cost", []), dtype=float)
    curtails = np.asarray(log.get("curtail", []), dtype=float)

    if episodes.size == 0:
        print("日志文件中缺少 episode 信息，无法绘图。")
        return

    # 只关注前若干个 episode，用于放大前期上升阶段
    focus_episodes = 500
    if episodes.size > focus_episodes:
        episodes = episodes[:focus_episodes]
        returns = returns[:focus_episodes]
        costs = costs[:focus_episodes]
        curtails = curtails[:focus_episodes]

    # 确保长度一致
    n = episodes.size
    returns = returns[:n]
    costs = costs[:n]
    curtails = curtails[:n]

    # 滑动窗口大小（目前主要用于参考，如需结合累计平均进一步平滑可适当调节）
    window = max(5, n // 70)  # 例如 1000 episode -> 窗口约 25

    # 三个指标统一采用累计平均，更清晰地展示“逐渐上升到稳定值”的过程
    cum_ret = np.cumsum(returns) / np.arange(1, n + 1)
    cum_cost = np.cumsum(costs) / np.arange(1, n + 1)
    cum_curtail = np.cumsum(curtails) / np.arange(1, n + 1)

    # ===================== 3. Matplotlib 全局风格设置 =====================
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 7.0), dpi=200, sharex=True)

    # 上：回报曲线（累计平均）
    ax_ret = axes[0]
    ax_ret.plot(episodes, cum_ret, color="tab:orange", linewidth=1.8)
    ax_ret.set_ylabel("累计平均回报", fontsize=11)
    ax_ret.set_title("Expert 0 训练过程：回报 / 成本 / 弃电平滑曲线", fontsize=12)
    ax_ret.grid(False)
    for spine in ax_ret.spines.values():
        spine.set_linewidth(1.0)
    ax_ret.tick_params(direction="in", length=4, width=1.0)

    # 中：成本曲线（累计平均，成本为负，越负越好）
    ax_cost = axes[1]
    ax_cost.plot(episodes, cum_cost, color="tab:blue", linewidth=1.8)
    ax_cost.set_ylabel("累计平均成本 (元)", fontsize=11)
    ax_cost.grid(False)
    for spine in ax_cost.spines.values():
        spine.set_linewidth(1.0)
    ax_cost.tick_params(direction="in", length=4, width=1.0)

    # 下：弃电曲线（累计平均）
    ax_curt = axes[2]
    ax_curt.plot(episodes, cum_curtail, color="tab:green", linewidth=1.8)
    ax_curt.set_ylabel("累计平均弃电量 (KWh)", fontsize=11)
    ax_curt.set_xlabel("Episode", fontsize=11)
    ax_curt.grid(False)
    for spine in ax_curt.spines.values():
        spine.set_linewidth(1.0)
    ax_curt.tick_params(direction="in", length=4, width=1.0)

    # 统一 X 轴范围
    ax_curt.set_xlim(episodes.min(), episodes.max())

    fig.tight_layout()

    project_root = pathlib.Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 使用中文标题作为文件名
    filename = "SAC训练曲线图"
    
    # 保存PNG格式（高分辨率位图）
    png_path = results_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    # 保存SVG格式（矢量图）
    svg_path = results_dir / f"{filename}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches="tight", facecolor='white', edgecolor='none')
    
    print(f"SAC训练曲线图已保存为:")
    print(f"  - {png_path} (高分辨率PNG)")
    print(f"  - {svg_path} (矢量图SVG)")


if __name__ == "__main__":
    main()
