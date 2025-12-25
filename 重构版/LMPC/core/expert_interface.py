"""Phase 3 专家接口

功能：
- 加载 Phase 2 训练好的 SAC 专家（5 个）
- 基于未来 8 小时预测数据，滚动调用专家策略生成 8 小时参考计划
- 内部使用与 Phase 2 一致的物理模型和归一化方式
- 在专家切换时，对参考轨迹前若干步做线性平滑，避免突变

输入数据约定（与 evaluate_system.py / 预测脚本保持一致）：
- forecast_8h: {
    'load':  长度 >= 32 的数组，单位 kW
    'pv':    长度 >= 32 的数组，单位 MW
    'wind':  长度 >= 32 的数组，单位 MW
    'price': 长度 >= 32 的数组，单位 元/kWh
  }

输出参考计划：
- {'soc': np.array(32,), 'grid_power': np.array(32,)}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from SAC.sac_agent import SACAgent

# 项目根目录，例如 c:/.../重构版
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class NormBounds:
    """状态归一化边界，和 Phase 2 保持一致。"""

    max_load: float   # MW
    max_pv: float     # MW
    max_wind: float   # MW
    max_price: float  # 元/kWh
    max_grid: float   # MW


class ExpertInterface:
    """SAC 专家接口

    用法：
    >>> interface = ExpertInterface(config)
    >>> plan = interface.get_plan(expert_id, state, forecast_8h)
    >>> plan['soc'].shape == (32,)
    >>> plan['grid_power'].shape == (32,)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # ---------- 读取容量与物理参数 ----------
        cap_cfg = config.get("capacity", {})
        self.cap_wind = float(cap_cfg.get("wind_mw", 20.0))
        self.cap_pv = float(cap_cfg.get("pv_mw", 35.0))
        self.cap_ts = float(cap_cfg.get("ts_mwh", 200.0))
        self.cap_eh = float(cap_cfg.get("eh_mw_th", 25.0))
        self.cap_st = float(cap_cfg.get("st_mw_e", 15.0))

        phy_cfg = config.get("physics", {})
        self.eta_eh = float(phy_cfg.get("eta_heater", 0.98))
        self.eta_st = float(phy_cfg.get("eta_turbine", 0.40))
        self.loss_rate = float(phy_cfg.get("loss_rate_per_h", 0.005))
        self.soc_min = float(phy_cfg.get("soc_min", 0.1))
        self.soc_max = float(phy_cfg.get("soc_max", 0.9))

        # MPC 时间步长（小时）
        mpc_cfg = config.get("mpc", {})
        dt_min = float(mpc_cfg.get("time_step_minutes", 15))
        self.dt_hours = dt_min / 60.0

        # ---------- 归一化边界（和 Phase 2 一致） ----------
        self.norm = self._load_norm_bounds()

        # ---------- 专家切换平滑配置 ----------
        ei_cfg = config.get("expert_interface", {})
        sw_cfg = ei_cfg.get("switching", {})
        self.smooth_transition = bool(sw_cfg.get("smooth_transition", True))
        self.transition_window = int(sw_cfg.get("transition_window", 4))

        # ---------- 加载 SAC 配置与专家模型 ----------
        self.sac_config = self._load_sac_config()
        self.num_experts = int(self.sac_config.get("training", {}).get("num_experts", 5))
        self.expert_models_dir = PROJECT_ROOT / self.config.get("models", {}).get("expert_models_dir", "SAC/models")

        self.experts: Dict[int, SACAgent] = {}
        self._load_experts()

        # ---------- 切换状态 ----------
        self.prev_expert_id: int | None = None
        self.prev_plan: Dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # 配置与归一化
    # ------------------------------------------------------------------
    def _load_sac_config(self) -> Dict[str, Any]:
        """加载 Phase 2 的配置，用于构建 SACAgent。"""
        cfg_path = PROJECT_ROOT / "SAC" / "phase2_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"未找到 Phase 2 配置文件: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_norm_bounds(self) -> NormBounds:
        """从 SAC 的 clustered_training_data.csv 统计归一化边界。

        这样可以保证 Phase 3 喂给专家的状态分布和 Phase 2 训练环境一致。
        """
        csv_path = PROJECT_ROOT / "SAC" / "clustered_training_data.csv"
        if not csv_path.exists():
            # 兜底：用容量做一个合理的上界
            max_load = 100.0
            max_pv = self.cap_pv
            max_wind = self.cap_wind
            max_price = 1.0
            max_grid = max(max_load, max_pv + max_wind)
            return NormBounds(max_load, max_pv, max_wind, max_price, max_grid)

        df = pd.read_csv(csv_path)

        if "Load_MW" in df.columns:
            max_load = float(df["Load_MW"].max())
        elif "Load_kW" in df.columns:
            max_load = float(df["Load_kW"].max()) / 1000.0
        else:
            max_load = 100.0

        max_pv = float(df["PV_Gen_MW"].max()) if "PV_Gen_MW" in df.columns else self.cap_pv
        max_wind = float(df["Wind_Gen_MW"].max()) if "Wind_Gen_MW" in df.columns else self.cap_wind
        max_price = float(df["Price_CNY_kWh"].max()) if "Price_CNY_kWh" in df.columns else 1.0
        max_grid = max(max_load, max_pv + max_wind)

        return NormBounds(max_load, max_pv, max_wind, max_price, max_grid)

    # ------------------------------------------------------------------
    # 专家加载
    # ------------------------------------------------------------------
    def _load_experts(self) -> None:
        """加载所有 SAC 专家模型的 Actor + Critic。

        这里直接复用 SACAgent，确保网络结构与训练时一致。
        """
        print("📦 加载 SAC 专家模型...")

        state_dim = 6
        action_dim = 1
        for expert_id in range(self.num_experts):
            agent = SACAgent(state_dim, action_dim, self.sac_config, device="cpu")
            agent.load(self.expert_models_dir, expert_id)
            self.experts[expert_id] = agent

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def get_plan(
        self,
        expert_id: int,
        state: Dict[str, float],
        forecast_8h: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """生成 8 小时参考计划（32 步）。

        参数：
            expert_id: 0~num_experts-1
            state: {'soc': float, 'grid_power': float(MW)}
            forecast_8h: 见模块文档说明
        返回：
            {'soc': (32,), 'grid_power': (32,)}
        """
        new_plan = self._generate_plan(expert_id, state, forecast_8h)

        # 专家切换平滑
        if (
            self.smooth_transition
            and self.prev_plan is not None
            and self.prev_expert_id is not None
            and expert_id != self.prev_expert_id
        ):
            plan = self._smooth_transition(self.prev_plan, new_plan)
        else:
            plan = new_plan

        self.prev_expert_id = expert_id
        self.prev_plan = plan
        return plan

    # ------------------------------------------------------------------
    # 参考计划生成与平滑
    # ------------------------------------------------------------------
    def _generate_plan(
        self,
        expert_id: int,
        state: Dict[str, float],
        forecast_8h: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """用指定专家生成 8 小时、32 步的 SOC 与电网功率参考轨迹。"""
        soc = float(state.get("soc", 0.5))
        grid_prev = float(state.get("grid_power", 0.0))

        # 预测长度
        horizon = min(
            32,
            len(forecast_8h.get("load", [])),
            len(forecast_8h.get("pv", [])),
            len(forecast_8h.get("wind", [])),
            len(forecast_8h.get("price", [])),
        )
        if horizon <= 0:
            # 兜底：返回常数轨迹
            return {
                "soc": np.full(32, soc, dtype=np.float32),
                "grid_power": np.full(32, grid_prev, dtype=np.float32),
            }

        soc_traj = np.zeros(32, dtype=np.float32)
        grid_traj = np.zeros(32, dtype=np.float32)

        agent = self.experts[int(expert_id)]

        for k in range(32):
            idx = min(k, horizon - 1)

            # 构造 SAC 状态（6 维）
            state_vec = self._build_sac_state(soc, grid_prev, forecast_8h, idx)
            action = float(agent.take_action(state_vec, deterministic=True)[0])  # ∈ [-1, 1]

            # 物理步进
            soc, grid_mw = self._physics_step(soc, grid_prev, action, forecast_8h, idx)

            soc_traj[k] = soc
            grid_traj[k] = grid_mw
            grid_prev = grid_mw

        return {"soc": soc_traj, "grid_power": grid_traj}

    def _smooth_transition(
        self,
        old: Dict[str, np.ndarray],
        new: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """专家切换时，对前 transition_window 步做线性插值平滑。"""
        window = max(1, self.transition_window)
        out: Dict[str, np.ndarray] = {}

        for key in ["soc", "grid_power"]:
            old_arr = np.asarray(old.get(key, new.get(key)))
            new_arr = np.asarray(new.get(key, old.get(key)))
            length = min(len(old_arr), len(new_arr))

            result = new_arr.copy()
            w = min(window, length)
            if w > 1:
                alphas = np.linspace(0.0, 1.0, w)
                for i in range(w):
                    result[i] = (1 - alphas[i]) * old_arr[i] + alphas[i] * new_arr[i]
            elif w == 1:
                result[0] = 0.5 * old_arr[0] + 0.5 * new_arr[0]

            out[key] = result

        return out

    # ------------------------------------------------------------------
    # SAC 状态构造与物理步进
    # ------------------------------------------------------------------
    def _build_sac_state(
        self,
        soc: float,
        grid_prev: float,
        forecast_8h: Dict[str, Any],
        idx: int,
    ) -> np.ndarray:
        """构造给 SAC 专家的 6 维状态向量。"""
        load_kw = float(forecast_8h["load"][idx])
        load_mw = load_kw / 1000.0
        pv_mw = float(forecast_8h["pv"][idx])
        wind_mw = float(forecast_8h["wind"][idx])
        price = float(forecast_8h["price"][idx])

        s = np.array(
            [
                load_mw / (self.norm.max_load + 1e-6),
                pv_mw / (self.norm.max_pv + 1e-6),
                wind_mw / (self.norm.max_wind + 1e-6),
                soc,
                price / (self.norm.max_price + 1e-6),
                grid_prev / (self.norm.max_grid + 1e-6),
            ],
            dtype=np.float32,
        )
        return s

    def _physics_step(
        self,
        soc: float,
        grid_prev: float,
        action: float,
        forecast_8h: Dict[str, Any],
        idx: int,
    ) -> Tuple[float, float]:
        """在 15 分钟时间步上，复用 Phase 2 的物理模型逻辑。

        返回：
            new_soc, grid_mw
        """
        # 解析当前预测数据
        load_kw = float(forecast_8h["load"][idx])
        load_mw = load_kw / 1000.0
        pv_mw = float(forecast_8h["pv"][idx])
        wind_mw = float(forecast_8h["wind"][idx])
        price = float(forecast_8h["price"][idx])  # 当前实现中未直接用到

        ren_mw = pv_mw + wind_mw

        # 储能动作执行（与 MicrogridEnv.step 相同，但使用 dt_hours）
        a = float(np.clip(action, -1.0, 1.0))
        p_eh_in_mw = 0.0
        p_st_out_mw = 0.0

        if a > 0:  # 充电模式
            soc_headroom = self.soc_max - soc
            max_charge_energy = soc_headroom * self.cap_ts  # MWh_th
            max_charge_power_th = max_charge_energy / self.dt_hours  # MW_th
            max_charge_power_e = max_charge_power_th / self.eta_eh  # MW_e

            max_eh_power = self.cap_eh / self.eta_eh
            p_eh_target = a * min(max_charge_power_e, max_eh_power)

            # 电加热器实际消耗的电功率，不能超过可用电力
            p_eh_in_mw = min(p_eh_target, ren_mw + load_mw)

            # 转换为热功率并更新SOC
            p_eh_th = p_eh_in_mw * self.eta_eh
            delta_energy = p_eh_th * self.dt_hours
            soc = min(self.soc_max, soc + delta_energy / self.cap_ts)

        elif a < 0:  # 放电模式
            soc_available = soc - self.soc_min
            max_discharge_energy = soc_available * self.cap_ts  # MWh_th
            max_discharge_power_th = max_discharge_energy / self.dt_hours  # MW_th
            max_discharge_power_e = max_discharge_power_th * self.eta_st  # MW_e

            p_st_target = abs(a) * min(max_discharge_power_e, self.cap_st)

            # 汽轮机实际输出电功率
            p_st_out_mw = p_st_target

            # 消耗的热功率并更新SOC
            p_st_th = p_st_out_mw / self.eta_st
            delta_energy = p_st_th * self.dt_hours
            soc = max(self.soc_min, soc - delta_energy / self.cap_ts)

        # 热损失
        if self.loss_rate > 0:
            loss_energy = soc * self.cap_ts * self.loss_rate * self.dt_hours
            soc = max(0.0, soc - loss_energy / self.cap_ts)

        # 功率平衡
        p_supply = ren_mw + p_st_out_mw
        p_demand = load_mw + p_eh_in_mw
        p_grid = p_supply - p_demand

        return soc, p_grid
