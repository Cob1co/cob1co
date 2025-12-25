"""微电网强化学习环境 - 用于SAC训练

设计要点：
1. 状态空间：6维（负荷/光伏/风电/SOC/电价/上时刻电网功率）
2. 动作空间：1维连续（储能充放电指令 ∈ [-1, 1]）
3. 奖励函数：运行成本 + 电网波动
4. Episode：4天连续（96步）
5. 物理模型：简化计算，速度优先
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from pathlib import Path
import yaml


class MicrogridEnv:
    """微电网日前调度环境（1维动作版本）"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        expert_id: int = 0,
    ):
        """
        参数：
            df: 某个专家对应的多日数据（已包含Wind_Gen_MW和PV_Gen_MW）
            config: phase2_config.yaml配置
            expert_id: 专家ID (0-4)
        """
        self.df = df.reset_index(drop=True)
        self.config = config
        self.expert_id = expert_id
        
        # 环境参数
        env_cfg = config["environment"]
        self.episode_days = env_cfg["episode_days"]
        self.max_steps = self.episode_days * 24
        self.dt_hours = env_cfg["dt_hours"]
        self.soc_min = env_cfg["soc_min"]
        self.soc_max = env_cfg["soc_max"]
        self.initial_soc = env_cfg["initial_soc"]
        
        # 容量参数
        cap_cfg = config["capacity"]
        self.cap_wind = cap_cfg["wind_mw"]
        self.cap_pv = cap_cfg["pv_mw"]
        self.cap_ts = cap_cfg["ts_mwh"]
        self.cap_eh = cap_cfg["eh_mw_th"]
        self.cap_st = cap_cfg["st_mw_e"]
        
        # 从models/config.yaml读取效率参数
        self.eta_eh = self._load_efficiency("electric_heater", 0.98)
        self.eta_st = self._load_efficiency("steam_turbine", 0.40)
        self.loss_rate = self._load_loss_rate()
        
        # 多目标权重（成本 / 波动 / 弃电）
        obj_cfg = config["objective"]
        self.w_cost = obj_cfg["w_cost"]
        self.w_ramp = obj_cfg["w_ramp"]
        self.w_curt = obj_cfg["w_curt"]
        self.cost_scale = obj_cfg["cost_scale"]
        self.ramp_scale = obj_cfg["ramp_scale"]
        self.curtail_scale = obj_cfg["curtail_scale"]
        
        # 归一化边界
        self._compute_normalization_bounds()
        
        # 状态追踪
        self.global_idx = 0
        self.episode_step = 0
        self.soc = self.initial_soc
        self.p_grid_prev = 0.0
        
        # 统计信息
        self.episode_cost = 0.0
        self.episode_curtail = 0.0
    
    def _load_efficiency(self, device_name: str, default: float) -> float:
        """从models/config.yaml加载设备效率"""
        try:
            model_config_path = Path(__file__).parent.parent / "models" / "config.yaml"
            with open(model_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            assets = cfg.get("assets", {})
            device_cfg = assets.get(device_name, {})
            return float(device_cfg.get("eta", default))
        except:
            return default
    
    def _load_loss_rate(self) -> float:
        """加载储热损失率"""
        try:
            model_config_path = Path(__file__).parent.parent / "models" / "config.yaml"
            with open(model_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            physics = cfg.get("physics", {})
            molten = physics.get("molten_salt", {})
            return float(molten.get("loss_rate_per_h", 0.005))
        except:
            return 0.005
    
    def _compute_normalization_bounds(self):
        """计算归一化边界"""
        self.max_load = self.df["Load_MW"].max() if "Load_MW" in self.df.columns else 100.0
        self.max_pv = self.df["PV_Gen_MW"].max() if "PV_Gen_MW" in self.df.columns else self.cap_pv
        self.max_wind = self.df["Wind_Gen_MW"].max() if "Wind_Gen_MW" in self.df.columns else self.cap_wind
        self.max_price = self.df["Price_CNY_kWh"].max() if "Price_CNY_kWh" in self.df.columns else 1.0
        self.max_grid = max(self.max_load, self.max_pv + self.max_wind)
    
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        # 随机选择起始点（确保有足够的数据）
        max_start = len(self.df) - self.max_steps
        if max_start > 0:
            self.global_idx = np.random.randint(0, max_start)
        else:
            self.global_idx = 0
        
        # 重置状态
        self.episode_step = 0
        self.soc = self.initial_soc
        self.p_grid_prev = 0.0
        
        # 重置统计
        self.episode_cost = 0.0
        self.episode_curtail = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """构造状态向量（6维）
        
        [0] load_mw/max_load：负荷（归一化）
        [1] pv_mw/max_pv：光伏出力（归一化）
        [2] wind_mw/max_wind：风电出力（归一化）
        [3] soc：储能SOC
        [4] price/max_price：电价（归一化）
        [5] p_grid_prev/max_grid：上时刻电网功率（归一化）
        """
        if self.global_idx >= len(self.df):
            return np.zeros(6, dtype=np.float32)
        
        row = self.df.iloc[self.global_idx]
        
        load = row["Load_MW"] if "Load_MW" in row else row["Load_kW"] / 1000.0
        pv = row["PV_Gen_MW"] if "PV_Gen_MW" in row else 0.0
        wind = row["Wind_Gen_MW"] if "Wind_Gen_MW" in row else 0.0
        price = row["Price_CNY_kWh"] if "Price_CNY_kWh" in row else 0.5
        
        state = np.array([
            load / (self.max_load + 1e-6),
            pv / (self.max_pv + 1e-6),
            wind / (self.max_wind + 1e-6),
            self.soc,
            price / (self.max_price + 1e-6),
            self.p_grid_prev / (self.max_grid + 1e-6),
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步
        
        参数：
            action: ∈ [-1, 1]
                   > 0: 充电（电加热器）
                   < 0: 放电（汽轮机）
        
        返回：
            next_state, reward, done, info
        """
        if self.global_idx >= len(self.df):
            return self._get_state(), 0.0, True, {}
        
        row = self.df.iloc[self.global_idx]
        
        # ========== 1. 解析当前数据 ==========
        load_mw = row["Load_MW"] if "Load_MW" in row else row["Load_kW"] / 1000.0
        pv_mw = row["PV_Gen_MW"] if "PV_Gen_MW" in row else 0.0
        wind_mw = row["Wind_Gen_MW"] if "Wind_Gen_MW" in row else 0.0
        price = row["Price_CNY_kWh"] if "Price_CNY_kWh" in row else 0.5
        
        ren_mw = pv_mw + wind_mw
        
        # ========== 2. 储能动作执行 ==========
        action = np.clip(action, -1.0, 1.0)
        
        p_eh_in_mw = 0.0
        p_st_out_mw = 0.0
        p_curtail_mw = 0.0
        
        if action > 0:  # 充电模式
            # 计算最大可充电功率（受SOC上限约束）
            soc_headroom = self.soc_max - self.soc
            max_charge_energy = soc_headroom * self.cap_ts  # MWh_th
            max_charge_power_th = max_charge_energy / self.dt_hours  # MW_th
            max_charge_power_e = max_charge_power_th / self.eta_eh  # MW_e
            
            # 实际充电功率（不超过设备容量）
            max_eh_power = self.cap_eh / self.eta_eh
            p_eh_target = action * min(max_charge_power_e, max_eh_power)
            
            # 电加热器实际消耗的电功率
            p_eh_in_mw = min(p_eh_target, ren_mw + load_mw)  # 不能超过可用电力
            
            # 转换为热功率并更新SOC
            p_eh_th = p_eh_in_mw * self.eta_eh
            delta_energy = p_eh_th * self.dt_hours
            self.soc = min(self.soc_max, self.soc + delta_energy / self.cap_ts)
            
        elif action < 0:  # 放电模式
            # 计算最大可放电功率（受SOC下限约束）
            soc_available = self.soc - self.soc_min
            max_discharge_energy = soc_available * self.cap_ts  # MWh_th
            max_discharge_power_th = max_discharge_energy / self.dt_hours  # MW_th
            max_discharge_power_e = max_discharge_power_th * self.eta_st  # MW_e
            
            # 实际放电功率（不超过设备容量）
            p_st_target = abs(action) * min(max_discharge_power_e, self.cap_st)
            
            # 汽轮机实际输出电功率
            p_st_out_mw = p_st_target
            
            # 消耗的热功率并更新SOC
            p_st_th = p_st_out_mw / self.eta_st
            delta_energy = p_st_th * self.dt_hours
            self.soc = max(self.soc_min, self.soc - delta_energy / self.cap_ts)
        
        # 热损失
        if self.loss_rate > 0:
            loss_energy = self.soc * self.cap_ts * self.loss_rate * self.dt_hours
            self.soc = max(0.0, self.soc - loss_energy / self.cap_ts)
        
        # ========== 3. 功率平衡 ==========
        p_supply = ren_mw + p_st_out_mw
        p_demand = load_mw + p_eh_in_mw
        p_grid = p_supply - p_demand
        
        p_export = max(0.0, p_grid)
        p_import = max(0.0, -p_grid)
        
        # 弃电：可再生过剩且无法充储
        if ren_mw > load_mw + p_eh_in_mw:
            p_curtail_mw = ren_mw - load_mw - p_eh_in_mw
        
        # ========== 4. 计算奖励 ==========
        # (1) 运行成本项（只计购电成本，不计卖电收益）
        cost_buy = p_import * price * 1000  # 元/h
        cost_net = cost_buy
        
        # (2) 电网波动惩罚项
        ramp = abs(p_grid - self.p_grid_prev)
        ramp_penalty = ramp * self.ramp_scale
        
        # (3) 弃风弃光惩罚项
        curtail_penalty = p_curtail_mw * self.curtail_scale
        
        # (4) SOC越界惩罚（软约束）
        soc_penalty = 0.0
        if self.soc < self.soc_min - 0.01 or self.soc > self.soc_max + 0.01:
            soc_penalty = 1000.0  # 严重惩罚
        
        # 综合reward（归一化）
        reward = -(
            self.w_cost * cost_net +
            self.w_ramp * ramp_penalty +
            self.w_curt * curtail_penalty +
            soc_penalty
        ) / self.cost_scale
        
        # ========== 5. 更新状态 ==========
        self.p_grid_prev = p_grid
        self.global_idx += 1
        self.episode_step += 1
        
        # 统计信息
        self.episode_cost += cost_net
        self.episode_curtail += p_curtail_mw * self.dt_hours
        
        done = self.episode_step >= self.max_steps
        
        # ========== 6. 信息记录 ==========
        info = {
            "load_mw": load_mw,
            "pv_mw": pv_mw,
            "wind_mw": wind_mw,
            "ren_mw": ren_mw,
            "st_out_mw": p_st_out_mw,
            "eh_in_mw": p_eh_in_mw,
            "grid_mw": p_grid,
            "export_mw": p_export,
            "import_mw": p_import,
            "curtail_mw": p_curtail_mw,
            "soc": self.soc,
            "cost_net": cost_net,
            "ramp": ramp,
            "price": price,
            "episode_cost": self.episode_cost,
            "episode_curtail": self.episode_curtail,
        }
        
        return self._get_state(), reward, done, info
    
    def get_state_dim(self) -> int:
        """返回状态空间维度"""
        return 6
    
    def get_action_dim(self) -> int:
        """返回动作空间维度"""
        return 1


if __name__ == "__main__":
    """测试环境"""
    import yaml
    
    # 加载配置
    config_path = Path(__file__).parent / "phase2_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 加载测试数据
    data_path = Path(__file__).parent / "clustered_training_data.csv"
    if not data_path.exists():
        print("警告：测试数据不存在，请先运行 prepare_training_data.py")
        exit(1)
    
    df = pd.read_csv(data_path)
    df_expert0 = df[df["Day_Label"] == 0]
    
    print("=" * 60)
    print("环境测试")
    print("=" * 60)
    print(f"数据形状: {df_expert0.shape}")
    print(f"列名: {df_expert0.columns.tolist()}")
    
    # 创建环境
    env = MicrogridEnv(df_expert0, config, expert_id=0)
    
    print(f"\n状态维度: {env.get_state_dim()}")
    print(f"动作维度: {env.get_action_dim()}")
    print(f"Episode长度: {env.max_steps} 步")
    
    # 测试一个episode
    print(f"\n开始测试episode...")
    state = env.reset()
    print(f"初始状态: {state}")
    
    total_reward = 0
    for step in range(10):  # 只测试10步
        action = np.random.uniform(-1, 1)  # 随机动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step < 3:  # 只打印前3步
            print(f"\n步骤 {step + 1}:")
            print(f"  动作: {action:.3f}")
            print(f"  奖励: {reward:.3f}")
            print(f"  SOC: {info['soc']:.3f}")
            print(f"  电网功率: {info['grid_mw']:.2f} MW")
            print(f"  成本: {info['cost_net']:.2f} 元")
        
        if done:
            break
    
    print(f"\n累计奖励: {total_reward:.3f}")
    print(f"\n✓ 环境测试通过！")
