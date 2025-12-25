"""Phase 3 训练数据收集器（DataCollector）

目标：
- 基于 2023 年 3 月的真实数据 + 8 小时预测数据，
- 在当前的 "天气分类 → 专家接口 → MPC" 流程上，
- 通过 H 步前瞻 + 网格搜索生成 Transformer 训练标签：
  - 输入：24×12 特征序列 state_seq
  - 标签：最优 (alpha_soc, alpha_grid, alpha_cost)

注意：
- 为了简单起步，本版本只收集 2023-03-01 ~ 2023-03-31 的样本；
- 所有超参数（候选权重、H 步长度等）从 phase3_config.yaml 中读取；
- 使用当前实现的 WeatherClassifier / ExpertInterface / MPCController / FeatureExtractor。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from LMPC.core.weather_classifier import WeatherClassifier
from LMPC.core.expert_interface import ExpertInterface
from LMPC.core.feature_extractor import FeatureExtractor
from LMPC.core.mpc_controller import MPCController


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIOLATION_WEIGHT = 1000.0  # SOC 越界惩罚权重


@dataclass
class DataCollectorConfig:
    """从 phase3_config.yaml 中提取的数据收集相关配置。"""

    candidates: List[float]
    horizon_steps: int
    discount_factor: float


class DataCollector:
    """Phase 3 训练数据收集器

    使用方法（示例）：

    >>> from pathlib import Path
    >>> import yaml
    >>> cfg_path = PROJECT_ROOT / 'LMPC' / 'phase3_config.yaml'
    >>> config = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    >>> collector = DataCollector(config)
    >>> dataset = collector.collect_month(year=2023, month=3, save=True)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 读取数据路径
        data_cfg = config.get("data", {})
        self.path_historical_2023 = PROJECT_ROOT / data_cfg.get("historical_2023", "data/data2023.csv")
        # 训练用 2023 预测数据
        self.path_forecast_2023 = PROJECT_ROOT / data_cfg.get(
            "forecast_2023_training", "LMPC/data/forecast_2023_8h_training.pkl"
        )
        # 输出训练数据路径
        self.path_output = PROJECT_ROOT / data_cfg.get(
            "transformer_training_data", "data/transformer_training_data.pkl"
        )

        # 读取数据收集配置
        t_cfg = config.get("transformer", {}).get("training", {})
        dc_cfg = t_cfg.get("data_collection", {})
        self.dc_cfg = DataCollectorConfig(
            candidates=list(dc_cfg.get("candidates", [0.7, 1.0, 1.3])),
            horizon_steps=int(dc_cfg.get("horizon_steps", 16)),
            discount_factor=float(dc_cfg.get("discount_factor", 0.99)),
        )

        # 容量信息（用于 MW 转换和储能物理约束）
        cap_cfg = config.get("capacity", {})
        self.cap_pv = float(cap_cfg.get("pv_mw", 35.0))
        self.cap_wind = float(cap_cfg.get("wind_mw", 20.0))
        self.E_cap = float(cap_cfg.get("ts_mwh", 200.0))
        self.P_max = float(cap_cfg.get("st_mw_e", 15.0))

        # 物理参数（SOC 约束、效率等）
        phy_cfg = config.get("physics", {})
        self.soc_min = float(phy_cfg.get("soc_min", 0.1))
        self.soc_max = float(phy_cfg.get("soc_max", 0.9))
        self.eta_storage = float(phy_cfg.get("eta_storage", 1.0))

        obj_cfg = config.get("objective", {})
        self.w_cost = float(obj_cfg.get("w_cost", 0.8))
        self.w_ramp = float(obj_cfg.get("w_ramp", 1.0))
        self.w_curt = float(obj_cfg.get("w_curt", 0.6))
        self.cost_scale = float(obj_cfg.get("cost_scale", 10000.0))
        self.ramp_scale = float(obj_cfg.get("ramp_scale", 50.0))
        self.curtail_scale = float(obj_cfg.get("curtail_scale", 50.0))

        # 控制周期（小时），用于累计 expert_switch_time 和储能动态
        mpc_cfg = config.get("mpc", {})
        self.dt_hours = float(mpc_cfg.get("time_step_minutes", 15)) / 60.0

        # 模块实例
        self.weather_classifier = WeatherClassifier(config)
        self.expert_interface = ExpertInterface(config)
        self.feature_extractor = FeatureExtractor(config)
        self.mpc = MPCController(config)

        # Oracle 理论改善统计用缓冲
        self.oracle_improvements: List[float] = []

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_historical_data(self) -> pd.DataFrame:
        """加载 2023 年历史数据，并确保 Time 为 datetime。"""
        if not self.path_historical_2023.exists():
            raise FileNotFoundError(f"历史数据文件不存在: {self.path_historical_2023}")

        df = pd.read_csv(self.path_historical_2023, parse_dates=["Time"])
        if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
            df["Time"] = pd.to_datetime(df["Time"])
        return df

    def _load_forecast_data(self) -> List[Dict[str, Any]]:
        """加载 2023 年 8 小时预测数据列表。"""
        path = self.path_forecast_2023
        if not path.exists():
            # 尝试从旧路径回退（保持兼容性）
            alt = PROJECT_ROOT / "LMPC" / "data" / "forecast_2023_8h_training.pkl"
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(
                    f"预测数据文件不存在: {self.path_forecast_2023} 或 {alt}"
                )

        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError("forecast_2023_8h_training.pkl 格式异常，预期为 list")
        return data

    # ------------------------------------------------------------------
    # 预测误差计算（滚动窗口，避免数据泄露）
    # ------------------------------------------------------------------
    def _compute_forecast_error(
        self,
        df: pd.DataFrame,
        forecast_data: List[Dict[str, Any]],
        idx: int,
        key: str,
        window: int = 4,
    ) -> float:
        """计算某一物理量在过去 window 步的平均预测误差。

        参数：
            df: 全年真实数据 DataFrame
            forecast_data: 预测数据列表
            idx: 当前全局时间步索引
            key: 'load' / 'solar' / 'wind'
            window: 回看步数
        误差定义（示例，以 load 为例）：
            real: df.loc[t, 'Load_kW']
            pred: forecast_data[t]['forecast']['load'][0]  # 对当时刻自身的预测
            error = |real - pred| / max(real, 1e-6)
        然后对 t-1..t-window 做平均。
        """
        errors: List[float] = []
        for i in range(1, window + 1):
            t = idx - i
            if t < 0 or t >= len(df) or t >= len(forecast_data):
                continue

            f_item = forecast_data[t]
            f_dict = f_item.get("forecast", {})

            if key == "load":
                real = float(df.iloc[t]["Load_kW"])
                pred_arr = np.asarray(f_dict.get("load", []), dtype=float)
                if pred_arr.size == 0:
                    continue
                pred = float(pred_arr[0])  # 对自身时刻的预测
            elif key == "solar":
                real = float(df.iloc[t]["Solar_W_m2"])
                pred_arr = np.asarray(f_dict.get("solar", []), dtype=float)
                if pred_arr.size == 0:
                    continue
                pred = float(pred_arr[0])
            elif key == "wind":
                real = float(df.iloc[t]["Wind_Speed_m_s"])
                pred_arr = np.asarray(f_dict.get("wind", []), dtype=float)
                if pred_arr.size == 0:
                    continue
                pred = float(pred_arr[0])
            else:
                continue

            if real <= 0:
                continue
            err = abs(real - pred) / real
            errors.append(err)

        return float(np.mean(errors)) if errors else 0.0

    # ------------------------------------------------------------------
    # 主收集函数（按月份）
    # ------------------------------------------------------------------
    def collect_month(self, year: int = 2023, month: int = 3, save: bool = True) -> List[Dict[str, Any]]:
        """收集指定年份/月份的训练数据（默认 2023 年 3 月）。

        返回的数据集列表中，每个元素为：
        {
            'state_sequence': np.array(24,12),
            'optimal_weights': np.array(3,),
        }
        若 save=True，则同时保存到配置中的 transformer_training_data 路径。
        """
        df = self._load_historical_data()
        forecast_data = self._load_forecast_data()

        # 选取指定月份的数据索引（全局索引）
        mask = (df["Time"].dt.year == year) & (df["Time"].dt.month == month)
        idx_all = df.index[mask].to_list()
        if not idx_all:
            raise ValueError(f"在 {year}-{month:02d} 未找到任何数据，请检查 data2023.csv")

        idx_all = sorted(idx_all)

        # 历史状态与特征缓冲
        history_states: List[Dict[str, Any]] = []
        feature_buffer: List[np.ndarray] = []
        history_len = int(self.feature_extractor.history_len)

        # 当前物理状态（用来驱动专家接口与 MPC）
        phy_cfg = self.config.get("physics", {})
        initial_soc = float(phy_cfg.get("initial_soc", 0.5))
        current_soc = initial_soc
        prev_grid_power = 0.0

        prev_expert_id: int | None = None
        time_since_switch_h: float = 0.0

        dataset: List[Dict[str, Any]] = []

        # 使用 tqdm 显示进度
        for idx in tqdm(idx_all, desc=f"收集 {year}-{month:02d} 训练数据"):
            # 对齐预测数据
            if idx >= len(forecast_data):
                break

            row = df.iloc[idx]
            current_time = row["Time"]

            # ------ 构造用于天气分类的历史真实数据（过去最多24步） ------
            hist_start = max(0, idx - 23)
            history_df = df.iloc[hist_start : idx + 1]

            expert_id = self.weather_classifier.classify_from_history(history_df)

            # ------ 构造 8 小时预测数据（与 evaluate_system 保持一致） ------
            f_item = forecast_data[idx]
            f_dict = f_item.get("forecast", {})

            load_kw = np.asarray(f_dict.get("load", []), dtype=float)
            solar_w = np.asarray(f_dict.get("solar", []), dtype=float)
            wind_ms = np.asarray(f_dict.get("wind", []), dtype=float)
            price = np.asarray(f_dict.get("price", []), dtype=float)

            if not (load_kw.size and solar_w.size and wind_ms.size and price.size):
                # 预测数据不完整，跳过本步
                continue

            # 转为 MW
            pv_mw = solar_w / 1000.0 * self.cap_pv
            wind_mw = wind_ms / 10.0 * self.cap_wind

            forecast_8h = {
                "load": load_kw,
                "pv": pv_mw,
                "wind": wind_mw,
                "price": price,
            }

            # ------ 专家参考计划（8 小时 32 步） ------
            current_state = {"soc": current_soc, "grid_power": prev_grid_power}
            expert_plan = self.expert_interface.get_plan(expert_id, current_state, forecast_8h)

            # ------ 预测误差（过去4步） ------
            load_err = self._compute_forecast_error(df, forecast_data, idx, key="load")
            pv_err = self._compute_forecast_error(df, forecast_data, idx, key="solar")
            wind_err = self._compute_forecast_error(df, forecast_data, idx, key="wind")
            forecast_errors = {"load": load_err, "pv": pv_err, "wind": wind_err}

            # ------ 专家切换时间（以小时累计） ------
            if prev_expert_id is None or expert_id == prev_expert_id:
                time_since_switch_h += self.dt_hours
            else:
                time_since_switch_h = 0.0
            prev_expert_id = expert_id

            # ------ 维护历史状态序列（用于特征提取） ------
            history_states.append(
                {
                    "soc": current_soc,
                    "grid_power": prev_grid_power,
                    "time": current_time,
                }
            )
            if len(history_states) > history_len:
                history_states.pop(0)

            # ------ 提取当前特征 ------
            features_t = self.feature_extractor.extract_features(
                history_states=history_states,
                expert_plan=expert_plan,
                forecast_errors=forecast_errors,
                expert_id=expert_id,
                expert_switch_time=time_since_switch_h,
            )
            feature_buffer.append(features_t)
            if len(feature_buffer) > history_len:
                feature_buffer.pop(0)

            # 若历史特征长度不足 24，则暂时不生成训练样本
            if len(feature_buffer) < history_len:
                # 仍需要推进物理状态
                self._update_physics_baseline(current_state, forecast_8h, expert_plan)
                current_soc = current_state["soc"]
                prev_grid_power = current_state["grid_power"]
                continue

            # ------ 构造 state_sequence (24,12) ------
            state_seq = np.stack(feature_buffer).astype(np.float32)

            # ------ 生成标签：H 步前瞻 + Oracle 网格搜索 ------
            best_cost = np.inf
            best_weights: Tuple[float, float, float] | None = None
            base_cost = None  # 固定 α=1.0 的基线代价

            # 预测窗口长度（受预测和真实数据共同限制）
            H = min(self.dc_cfg.horizon_steps, load_kw.size)
            # 确保真实数据也有足够长度
            max_H_real = len(df) - idx
            H = min(H, max_H_real)
            if H <= 0:
                self._update_physics_baseline(current_state, forecast_8h, expert_plan)
                current_soc = current_state["soc"]
                prev_grid_power = current_state["grid_power"]
                continue

            # 预测世界下的 MPC 输入
            mpc_forecast = {
                "load": load_kw[:H] / 1000.0,  # kW -> MW
                "pv": pv_mw[:H],
                "wind": wind_mw[:H],
                "price": price[:H],
            }
            reference_plan_H = {
                "soc": expert_plan["soc"][:H],
                "grid_power": expert_plan["grid_power"][:H],
            }

            # 真实世界数据切片（用于 Oracle 模拟）
            real_slice = df.iloc[idx : idx + H]
            load_real_mw = real_slice["Load_kW"].to_numpy(dtype=float) / 1000.0
            solar_real = real_slice["Solar_W_m2"].to_numpy(dtype=float)
            wind_real = real_slice["Wind_Speed_m_s"].to_numpy(dtype=float)
            price_real = real_slice["Price_CNY_kWh"].to_numpy(dtype=float)

            pv_real_mw = solar_real / 1000.0 * self.cap_pv
            wind_real_mw = wind_real / 10.0 * self.cap_wind

            for a_soc in self.dc_cfg.candidates:
                for a_grid in self.dc_cfg.candidates:
                    for a_cost in self.dc_cfg.candidates:
                        dyn_w = {
                            "alpha_soc": float(a_soc),
                            "alpha_grid": float(a_grid),
                            "alpha_cost": float(a_cost),
                        }
                        sol = self.mpc.solve(
                            current_state=current_state,
                            forecast=mpc_forecast,
                            reference_plan=reference_plan_H,
                            dynamic_weights=dyn_w,
                        )
                        if sol.get("status") != "optimal":
                            continue

                        P_s_plan = np.asarray(sol.get("P_storage_plan", []), dtype=float)[:H]
                        P_curt_plan = np.asarray(sol.get("P_curtail_plan", []), dtype=float)[:H]

                        soc_traj, grid_traj, import_traj = self._simulate_real_trajectory(
                            soc0=current_soc,
                            load_mw=load_real_mw,
                            pv_mw=pv_real_mw,
                            wind_mw=wind_real_mw,
                            P_s_plan=P_s_plan,
                            P_curt_plan=P_curt_plan,
                        )

                        j_true = self._calculate_j_true(
                            soc_traj=soc_traj,
                            grid_traj=grid_traj,
                            import_traj=import_traj,
                            price_real=price_real,
                            grid_power_prev=prev_grid_power,
                            curtail_traj=P_curt_plan,
                        )

                        # 记录固定 α=1.0 的基线代价
                        if (
                            base_cost is None
                            and float(a_soc) == 1.0
                            and float(a_grid) == 1.0
                            and float(a_cost) == 1.0
                        ):
                            base_cost = j_true

                        if j_true < best_cost:
                            best_cost = j_true
                            best_weights = (float(a_soc), float(a_grid), float(a_cost))

            if best_weights is not None:
                # 统计 Oracle 相对基线的理论改善比例
                if base_cost is not None and np.isfinite(base_cost) and base_cost > 0.0:
                    improvement = (base_cost - best_cost) / base_cost
                    self.oracle_improvements.append(float(improvement))

                sample = {
                    "state_sequence": state_seq,
                    "optimal_weights": np.array(best_weights, dtype=np.float32),
                }
                dataset.append(sample)

            # ------ 推进物理状态（仍然使用固定权重 α=1.0 的基线） ------
            self._update_physics_baseline(current_state, forecast_8h, expert_plan)
            current_soc = current_state["soc"]
            prev_grid_power = current_state["grid_power"]

        # Oracle 理论改善统计
        if self.oracle_improvements:
            improvements = np.asarray(self.oracle_improvements, dtype=float)
            mean_improve = float(improvements.mean()) * 100.0
            median_improve = float(np.median(improvements)) * 100.0
            p90_improve = float(np.percentile(improvements, 90)) * 100.0
            max_improve = float(improvements.max()) * 100.0
            print("\n📊 Oracle 理论改善统计（基于 candidates 网格搜索）:")
            print(f"  样本数: {len(improvements)}")
            print(f"  平均改善: {mean_improve:.2f}%")
            print(f"  中位数: {median_improve:.2f}%")
            print(f"  90分位改善: {p90_improve:.2f}%")
            print(f"  最大改善: {max_improve:.2f}%\n")

        # 保存
        if save:
            self.path_output.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path_output, "wb") as f:
                pickle.dump(dataset, f)
            print(f"✅ 训练数据已保存: {self.path_output}，样本数: {len(dataset)}")

        return dataset

    # ------------------------------------------------------------------
    # 物理状态推进（用于构造历史状态）
    # ------------------------------------------------------------------
    def _update_physics_baseline(
        self,
        state: Dict[str, float],
        forecast_8h: Dict[str, np.ndarray],
        expert_plan: Dict[str, np.ndarray],
    ) -> None:
        """使用固定权重 α=1.0 的 MPC 结果推进一个时间步的 SOC 与电网功率。

        更新传入的 state 字典：
            state['soc'] -> 下一步 SOC
            state['grid_power'] -> 下一步电网功率
        """
        load_kw = np.asarray(forecast_8h["load"], dtype=float)
        pv_mw = np.asarray(forecast_8h["pv"], dtype=float)
        wind_mw = np.asarray(forecast_8h["wind"], dtype=float)
        price = np.asarray(forecast_8h["price"], dtype=float)

        H = min(self.dc_cfg.horizon_steps, load_kw.size)
        if H <= 0:
            return

        mpc_forecast = {
            "load": load_kw[:H] / 1000.0,
            "pv": pv_mw[:H],
            "wind": wind_mw[:H],
            "price": price[:H],
        }
        reference_plan_H = {
            "soc": expert_plan["soc"][:H],
            "grid_power": expert_plan["grid_power"][:H],
        }

        sol = self.mpc.solve(
            current_state={"soc": state["soc"], "grid_power": state["grid_power"]},
            forecast=mpc_forecast,
            reference_plan=reference_plan_H,
            dynamic_weights=None,  # 固定 α=1.0
        )
        if sol.get("status") != "optimal":
            return

        soc_plan = np.asarray(sol.get("soc_plan", []), dtype=float)
        grid_plan = np.asarray(sol.get("grid_plan", []), dtype=float)

        if soc_plan.size >= 2:
            state["soc"] = float(soc_plan[1])  # 使用第1步的 SOC 作为下一时刻状态
        if grid_plan.size >= 1:
            state["grid_power"] = float(grid_plan[0])

    # ------------------------------------------------------------------
    # Oracle 真实环境模拟：根据规划动作和真实数据滚动 H 步
    # ------------------------------------------------------------------
    def _simulate_real_trajectory(
        self,
        soc0: float,
        load_mw: np.ndarray,
        pv_mw: np.ndarray,
        wind_mw: np.ndarray,
        P_s_plan: np.ndarray,
        P_curt_plan: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在真实数据下，使用 MPC 规划的动作滚动 H 步。

        返回：
            soc_traj:  长度 H 的 SOC 轨迹（不含终点）
            grid_traj: 长度 H 的电网功率轨迹 P_grid_real
            import_traj: 长度 H 的购电功率轨迹 P_import_real
        """
        H = min(
            len(load_mw),
            len(pv_mw),
            len(wind_mw),
            len(P_s_plan),
            len(P_curt_plan),
        )
        if H <= 0:
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        soc_real = np.zeros(H + 1, dtype=float)
        grid_real = np.zeros(H, dtype=float)
        import_real = np.zeros(H, dtype=float)
        soc_real[0] = float(soc0)

        dt = self.dt_hours

        for k in range(H):
            soc_k = soc_real[k]

            # 物理限制：依据当前 SOC 计算允许的最大充/放电功率
            # 充电上限（受 soc_max 约束）
            headroom_charge = max(0.0, self.soc_max - soc_k)
            max_charge_energy = headroom_charge * self.E_cap  # MWh
            max_charge_power = 0.0
            if dt > 0 and max_charge_energy > 0:
                max_charge_power = max_charge_energy / dt
            max_charge_power = min(max_charge_power, self.P_max)

            # 放电上限（受 soc_min 约束）
            available_discharge = max(0.0, soc_k - self.soc_min)
            max_discharge_energy = available_discharge * self.E_cap
            max_discharge_power = 0.0
            if dt > 0 and max_discharge_energy > 0:
                max_discharge_power = max_discharge_energy / dt
            max_discharge_power = min(max_discharge_power, self.P_max)

            p_min_phys = -max_discharge_power
            p_max_phys = max_charge_power

            # 截断动作
            p_cmd = float(P_s_plan[k])
            p_real = float(np.clip(p_cmd, p_min_phys, p_max_phys))

            # 更新 SOC（允许略微越界，由 J_true 做软约束惩罚）
            soc_next = soc_k + self.eta_storage * p_real * dt / self.E_cap
            # 保证在 [0, 1] 范围内，物理上不可能超过 0~1
            soc_real[k + 1] = float(np.clip(soc_next, 0.0, 1.0))

            # 真实功率平衡
            load_k = float(load_mw[k])
            pv_k = float(pv_mw[k])
            wind_k = float(wind_mw[k])
            p_curt_k = float(P_curt_plan[k])

            P_grid_k = load_k - pv_k - wind_k + p_real + p_curt_k
            grid_real[k] = P_grid_k
            import_real[k] = max(0.0, -P_grid_k)

        # 返回前 H 步的轨迹（不含末端 SOC）
        return soc_real[:-1], grid_real, import_real

    # ------------------------------------------------------------------
    # Oracle 评价函数：固定权重的真实代价 J_true
    # ------------------------------------------------------------------
    def _calculate_j_true(
        self,
        soc_traj: np.ndarray,
        grid_traj: np.ndarray,
        import_traj: np.ndarray,
        price_real: np.ndarray,
        grid_power_prev: float,
        curtail_traj: np.ndarray,
    ) -> float:
        """计算固定权重下的真实代价 J_true。

        使用的权重：
        - self.mpc.w_soc_base
        - self.mpc.w_grid_base
        - self.mpc.w_cost_base
        - self.mpc.w_ramp
        - VIOLATION_WEIGHT (用于 SOC 越界惩罚)
        """
        H = min(
            len(soc_traj),
            len(grid_traj),
            len(import_traj),
            len(price_real),
            len(curtail_traj),
        )
        if H <= 0:
            return float("inf")

        cost = 0.0
        dt = self.dt_hours
        p_prev = float(grid_power_prev)

        for k in range(H):
            soc_k = float(soc_traj[k])
            grid_k = float(grid_traj[k])
            imp_k = float(import_traj[k])
            curtail_k = float(curtail_traj[k])

            price_k = float(price_real[k])

            cost_buy = imp_k * price_k * 1000.0 * dt
            cost_net = cost_buy

            ramp = abs(grid_k - p_prev)
            ramp_penalty = ramp * self.ramp_scale
            curtail_penalty = curtail_k * self.curtail_scale

            step_cost = (
                self.w_cost * cost_net
                + self.w_ramp * ramp_penalty
                + self.w_curt * curtail_penalty
            )

            violation = 0.0
            if soc_k < self.soc_min:
                violation += self.soc_min - soc_k
            if soc_k > self.soc_max:
                violation += soc_k - self.soc_max
            if violation > 0.0:
                step_cost += VIOLATION_WEIGHT * (violation ** 2)

            cost += step_cost
            p_prev = grid_k

        return float(cost / self.cost_scale)
