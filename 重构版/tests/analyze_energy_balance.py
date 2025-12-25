"""简单的年度能量平衡分析脚本

根据 realtime2024.csv 和当前 Phase3 配置中的装机容量，
用与 LMPC 近似一致的线性模型估算风光出力，并与负荷对比，
得到年度总负荷电量、风电电量、光伏电量以及覆盖率。

使用方法：
    python tests/analyze_energy_balance.py

输出：
    - 终端打印年度汇总结果
    - 在 tests 目录下生成 energy_balance_2024.csv
"""

from pathlib import Path

import argparse
import pandas as pd
import yaml


def load_config(project_root: Path) -> dict:
    """加载 Phase3 配置文件。"""
    cfg_path = project_root / "LMPC" / "phase3_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_realtime(project_root: Path) -> pd.DataFrame:
    """加载 2024 实时数据。"""
    csv_path = project_root / "data" / "realtime2024.csv"
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    return df


def compute_energy_balance(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """按年份统计能量平衡（目前 realtime2024 只有 2024 年）。

    约定：
    - 负荷列为 Load_kW，单位 kW；
    - 光伏列为 Solar_W_m2（辐照度），用与 LMPC 一致的线性近似：PV_MW = Solar_W_m2 / 1000 * pv_mw；
    - 风速列为 Wind_Speed_m_s，线性近似：Wind_MW = Wind_Speed_m_s / 10 * wind_mw；
    - 时间步长为 15 分钟，即 dt_hours = 0.25；
    - 年度电量单位统一为 MWh。
    """
    cap_cfg = config.get("capacity", {})
    pv_mw = float(cap_cfg.get("pv_mw", 35.0))
    wind_mw = float(cap_cfg.get("wind_mw", 20.0))

    # 时间步长（小时）
    dt_hours = 0.25

    # 负荷功率：kW -> MW
    load_kw = df["Load_kW"].astype(float)
    load_mw = load_kw / 1000.0

    # 光伏出力估算：与 LMPC 评估脚本相同的线性关系
    solar_w = df["Solar_W_m2"].astype(float)
    pv_mw_series = solar_w / 1000.0 * pv_mw

    # 风电出力估算：与 LMPC 评估脚本相同的线性关系
    wind_ms = df["Wind_Speed_m_s"].astype(float)
    wind_mw_series = wind_ms / 10.0 * wind_mw

    # 每个时间步对应的电量（MWh）
    load_mwh = load_mw * dt_hours
    pv_mwh = pv_mw_series * dt_hours
    wind_mwh = wind_mw_series * dt_hours

    # 按年份聚合（目前只有 2024 一年，但代码写成通用形式）
    years = df["Time"].dt.year
    summary = (
        pd.DataFrame(
            {
                "year": years,
                "load_mwh": load_mwh,
                "pv_mwh": pv_mwh,
                "wind_mwh": wind_mwh,
            }
        )
        .groupby("year", as_index=False)
        .sum()
    )

    # 计算风光合计以及可再生覆盖率
    summary["renew_mwh"] = summary["pv_mwh"] + summary["wind_mwh"]
    summary["renew_ratio"] = summary["renew_mwh"] / summary["load_mwh"]

    return summary


def main() -> None:
    """命令行入口：支持指定任意 CSV 文件进行年度能量平衡分析。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="data/realtime2024.csv",
        help="要分析的 CSV 相对路径，例如 data/realtime2024.csv 或 data/data2023.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果 CSV 输出相对路径，默认 tests/energy_balance_<year>.csv",
    )
    args = parser.parse_args()

    # 项目根目录：tests 的上级目录
    project_root = Path(__file__).parent.parent

    # 加载配置与数据
    config = load_config(project_root)
    csv_path = project_root / args.file
    df = pd.read_csv(csv_path, parse_dates=["Time"])

    summary = compute_energy_balance(df, config)

    # 打印结果（保留两位小数）
    print("\n================ 年度能量平衡统计 ================")
    for _, row in summary.iterrows():
        year = int(row["year"])
        load_mwh = float(row["load_mwh"])
        pv_mwh = float(row["pv_mwh"])
        wind_mwh = float(row["wind_mwh"])
        renew_mwh = float(row["renew_mwh"])
        renew_ratio = float(row["renew_ratio"])

        print(f"年份: {year}")
        print(f"  总负荷电量: {load_mwh:.2f} MWh")
        print(f"  光伏发电量: {pv_mwh:.2f} MWh")
        print(f"  风电发电量: {wind_mwh:.2f} MWh")
        print(f"  风光总发电量: {renew_mwh:.2f} MWh")
        print(f"  可再生覆盖率: {renew_ratio*100:.2f}%")
        print("--------------------------------------------------")

    # 保存到 CSV 便于后续画图或做更多分析
    if args.output:
        out_path = project_root / args.output
    else:
        if len(summary) == 1:
            year = int(summary.loc[0, "year"])
            out_name = f"energy_balance_{year}.csv"
        else:
            out_name = "energy_balance_multi_year.csv"
        out_path = project_root / "tests" / out_name

    summary.to_csv(out_path, index=False)
    print(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
