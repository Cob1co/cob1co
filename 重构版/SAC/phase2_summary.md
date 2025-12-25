# 第二阶段（SAC多专家强化学习调度）总结文档

## 📋 目标与架构

### 目标
基于第一阶段确定的设备容量，使用SAC（Soft Actor-Critic）强化学习算法，为5种典型天气类型分别训练专家策略，实现日前调度优化。

### 优化目标
- **最小化运行成本**：购电成本 - 售电收益
- **最小化电网功率波动**：减少ramping惩罚
- **控制弃电量**：隐含在成本优化中

### 架构设计
```
天气聚类(K-Means, k=5) → 训练5个SAC专家 → 评估专家性能 → 部署应用
```

---

## 📁 主要文件与模块

### 核心文件
| 文件名 | 功能 | 关键类/函数 |
|--------|------|------------|
| `phase2_config.yaml` | 配置文件 | 环境、SAC、训练参数 |
| `prepare_training_data.py` | 数据预处理 | `compute_renewable_output()`, `cluster_days_by_weather()` |
| `microgrid_env.py` | 强化学习环境 | `MicrogridEnv` |
| `sac_agent.py` | SAC智能体 | `SACAgent`, `PolicyNet`, `QValueNet` |
| `replay_buffer.py` | 经验回放 | `ReplayBuffer` |
| `train_expert.py` | 单专家训练 | `train_expert()` |
| `train_all_experts.py` | 批量训练 | `main()` |
| `eval_expert.py` | 单专家评估 | `evaluate_expert()` |
| `eval_all_experts.py` | 批量评估 | `evaluate_all_experts()`, `generate_comparison_report()` |

---

## 🔧 核心API接口

### 1. 数据预处理 (`prepare_training_data.py`)

#### `compute_renewable_output(df, wind, pv)`
```python
def compute_renewable_output(df: pd.DataFrame, wind: WindFarm, pv: PVPlant) -> pd.DataFrame:
    """
    计算风光出力
    
    输入：
        df: 原始数据，包含 ['Wind_Speed_m_s', 'Solar_W_m2', 'Temperature_C']
        wind: WindFarm模型实例
        pv: PVPlant模型实例
    
    输出：
        df: 添加列 ['Wind_Gen_MW', 'PV_Gen_MW', 'REN_Gen_MW']
    """
```

#### `cluster_days_by_weather(df, k_typical=5)`
```python
def cluster_days_by_weather(df: pd.DataFrame, k_typical: int = 5) -> pd.DataFrame:
    """
    天气聚类，为每天分配标签
    
    输入：
        df: 带有风光出力的数据
        k_typical: 聚类数量
    
    输出：
        df: 添加列 ['Date', 'Day_Index', 'Day_Label', 'Hour']
    
    聚类特征：
        - ghi_mean: 日均光照
        - wind_mean: 日均风速
        - temp_mean: 日均温度
        - price_diff: 日内电价波动
    """
```

---

### 2. 强化学习环境 (`microgrid_env.py`)

#### `MicrogridEnv` 类

##### 初始化
```python
def __init__(self, df: pd.DataFrame, config: dict, expert_id: int = 0):
    """
    参数：
        df: 训练数据，必须包含专家对应的天气类型数据
        config: phase2_config.yaml配置字典
        expert_id: 专家ID (0-4)
    
    关键属性：
        self.state_dim = 6   # 状态维度
        self.action_dim = 1  # 动作维度
        self.max_steps = episode_days × 24  # Episode长度
    """
```

##### 状态空间（6维）
```python
state = [
    load_norm,      # 归一化负荷 [0, 1]
    pv_norm,        # 归一化光伏出力 [0, 1]
    wind_norm,      # 归一化风电出力 [0, 1]
    soc,            # 储能SOC [0.1, 0.9]
    price_norm,     # 归一化电价 [0, 1]
    grid_prev_norm  # 归一化上一时刻电网功率 [-1, 1]
]
```

##### 动作空间（1维）
```python
action = storage_cmd  # 储能充放电指令 [-1, 1]
# -1: 最大充电（电加热器满功率）
#  0: 不充不放
# +1: 最大放电（汽轮机满功率）
```

##### 核心方法
```python
def reset(self) -> np.ndarray:
    """
    重置环境，开始新的episode
    
    返回：
        state: 初始状态 (6,)
    """

def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
    """
    执行动作，推进一步
    
    输入：
        action: 储能指令 [-1, 1]
    
    返回：
        next_state: 下一状态 (6,)
        reward: 奖励值（负成本 + 负ramping惩罚）
        done: 是否结束
        info: 详细信息字典
    
    info字典字段：
        'soc': 当前SOC
        'grid_mw': 电网功率 (MW)
        'curtail_mw': 弃电 (MW)
        'import_mw': 购电 (MW，grid>0时)
        'export_mw': 售电 (MW，grid<0时)
        'cost': 本步成本 (元)
        'episode_cost': 累计成本 (元，done=True时)
        'episode_curtail': 累计弃电 (MWh，done=True时)
    """

def get_state_dim(self) -> int:
    """返回状态维度 6"""

def get_action_dim(self) -> int:
    """返回动作维度 1"""
```

##### 奖励函数
```python
reward = -(w_cost × cost + w_ramp × ramp_penalty) / scale
# cost = grid_import_cost - grid_export_revenue  (元)
# ramp_penalty = |grid_t - grid_{t-1}|  (MW)
# 归一化：cost_scale = 10000, ramp_scale = 50
```

---

### 3. SAC智能体 (`sac_agent.py`)

#### `SACAgent` 类

##### 初始化
```python
def __init__(self, state_dim: int, action_dim: int, config: dict, device: str = "cuda"):
    """
    参数：
        state_dim: 状态维度（固定为6）
        action_dim: 动作维度（固定为1）
        config: phase2_config.yaml配置
        device: 计算设备 "cuda" 或 "cpu"
    
    网络结构：
        actor: PolicyNet(state_dim, hidden_dim, action_dim)
        critic1/2: QValueNet(state_dim, hidden_dim, action_dim)
        critic1/2_target: 目标网络
    
    超参数（来自config["sac"]）：
        lr_actor, lr_critic, lr_alpha: 学习率
        gamma: 折扣因子
        tau: 软更新系数
        alpha_init: 初始熵系数
        auto_alpha: 是否自动调整熵系数
    """
```

##### 核心方法
```python
def take_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
    """
    根据当前状态选择动作
    
    输入：
        state: 状态向量 (state_dim,)
        deterministic: True=确定性策略（评估用），False=随机策略（训练用）
    
    返回：
        action: 动作向量 (action_dim,)，范围[-1, 1]
    """

def update(self, transition_dict: dict) -> Tuple[float, float, float]:
    """
    更新网络参数
    
    输入：
        transition_dict: {
            'states': (batch_size, state_dim),
            'actions': (batch_size, action_dim),
            'rewards': (batch_size, 1),
            'next_states': (batch_size, state_dim),
            'dones': (batch_size, 1)
        }
    
    返回：
        critic_loss, actor_loss, alpha_loss
    """

def save(self, path: Path, expert_id: int):
    """
    保存模型权重
    
    输入：
        path: 保存目录
        expert_id: 专家ID
    
    保存文件：
        expert_{expert_id}_actor.pth
        expert_{expert_id}_critic1.pth
        expert_{expert_id}_critic2.pth
    """

def load(self, path: Path, expert_id: int):
    """
    加载模型权重
    
    输入：
        path: 模型目录
        expert_id: 专家ID
    """
```

---

### 4. 训练接口 (`train_expert.py`)

#### `train_expert(expert_id, config, show_progress=True)`
```python
def train_expert(expert_id: int, config: dict, show_progress: bool = True):
    """
    训练单个专家策略
    
    输入：
        expert_id: 专家ID (0-4)
        config: phase2_config.yaml配置
        show_progress: 是否显示进度条
    
    训练流程：
        1. 加载对应专家的数据（Day_Label == expert_id）
        2. 创建环境和智能体
        3. warmup阶段：随机探索填充replay buffer
        4. 训练循环：
           - 运行episode收集数据
           - 每步从buffer采样并更新网络
           - 每50个episode保存模型
        5. 保存最终模型和训练曲线
    
    输出：
        模型文件: models/expert_{expert_id}_*.pth
        训练日志: logs/expert_{expert_id}/training_log.json
        训练曲线: logs/expert_{expert_id}/training_curves.png
    """
```

---

### 5. 评估接口 (`eval_expert.py`, `eval_all_experts.py`)

#### `evaluate_expert(expert_id, config, episodes=20, deterministic=True)`
```python
def evaluate_expert(expert_id: int, config: dict, episodes: int = 20, deterministic: bool = True):
    """
    评估单个专家策略
    
    输入：
        expert_id: 专家ID (0-4)
        config: 配置字典
        episodes: 评估episode数量
        deterministic: 是否使用确定性策略
    
    输出：
        eval_results/expert_{expert_id}_eval.json: {
            "return": 平均回报,
            "cost": 平均成本(元),
            "curtail": 平均弃电(MWh),
            "import": 平均购电(MW),
            "export": 平均售电(MW),
            "ramp": 平均电网波动(MW),
            "episodes": 评估次数,
            "deterministic": 是否确定性
        }
    """
```

#### `evaluate_all_experts(config, episodes=20)`
```python
def evaluate_all_experts(config: dict, episodes: int = 20):
    """
    批量评估所有专家并生成对比报告
    
    输出文件：
        eval_results/all_experts_comparison.csv: 详细对比数据
        eval_results/all_experts_comparison.png: 可视化对比图表
    
    终端输出：
        - 各专家性能对比表格
        - 统计汇总（均值、标准差）
        - 最佳专家（最高回报/收益/最低弃电）
    """
```

---

## ⚙️ 配置文件说明 (`phase2_config.yaml`)

### 结构
```yaml
capacity:                    # 设备容量（来自Phase 1）
  wind_mw: 20.0             # 风电装机 (MW)
  pv_mw: 50.0               # 光伏装机 (MW)
  ts_mwh: 100.0             # 储能容量 (MWh)
  eh_mw_th: 40.0            # 电加热器功率 (MW_th)
  st_mw_e: 15.0             # 汽轮机功率 (MW_e)

data:                        # 数据路径
  source: data/data2023.csv
  clustered: clustered_training_data.csv  # 预处理后数据

environment:                 # 环境参数
  state_dim: 6              # 状态维度
  action_dim: 1             # 动作维度
  dt_hours: 1.0             # 时间步长(小时)
  episode_days: 4           # Episode长度(天)
  initial_soc: 0.5          # 初始SOC
  soc_min: 0.1              # SOC下限
  soc_max: 0.9              # SOC上限

objective:                   # 目标函数权重
  w_cost: 1.0               # 成本权重
  w_ramp: 0.15              # 电网波动权重
  cost_scale: 10000.0       # 成本归一化尺度
  ramp_scale: 50.0          # 波动归一化尺度

sac:                         # SAC超参数
  hidden_dim: 256           # 隐藏层维度
  lr_actor: 0.0003          # Actor学习率
  lr_critic: 0.0003         # Critic学习率
  lr_alpha: 0.0003          # 熵系数学习率
  gamma: 0.99               # 折扣因子
  tau: 0.005                # 软更新系数
  alpha_init: 0.2           # 初始熵系数
  auto_alpha: true          # 自动调整熵
  target_entropy: -1.0      # 目标熵（-action_dim）
  batch_size: 256           # 批次大小
  buffer_size: 100000       # 经验池大小
  warmup_steps: 1000        # 预热步数
  max_episodes: 1000        # 最大训练轮数
  update_frequency: 1       # 更新频率
  save_frequency: 50        # 保存频率
  eval_frequency: 20        # 评估频率

training:                    # 训练设置
  num_experts: 5            # 专家数量
  model_dir: models         # 模型保存目录
  log_dir: logs             # 日志保存目录
  use_gpu: true             # 是否使用GPU
  random_seed: 42           # 随机种子
  parallel: false           # 是否并行训练
  num_workers: 5            # 并行worker数量

evaluation:                  # 评估设置
  eval_episodes: 10         # 评估episode数
  eval_deterministic: true  # 使用确定性策略
```

---

## 🔄 完整使用流程

### 步骤1：数据预处理
```bash
cd SAC
python prepare_training_data.py
```

**输出**：
- `clustered_training_data.csv`：包含风光出力和天气标签的完整数据
- 自动更新 `phase2_config.yaml` 中的容量配置

**生成的数据字段**：
```python
['Time', 'Temperature_C', 'Solar_W_m2', 'Wind_Speed_m_s', 'Price_CNY_kWh', 
 'Load_kW', 'Wind_Gen_MW', 'PV_Gen_MW', 'REN_Gen_MW', 'Date', 'Day_Index', 
 'Day_Label', 'Hour', 'Load_MW']
```

### 步骤2：训练专家策略

**选项A：训练单个专家**
```bash
python train_expert.py --expert_id 0
```

**选项B：批量训练所有专家**
```bash
python train_all_experts.py
```

**输出**：
- `models/expert_{0-4}_actor.pth`：Actor网络权重
- `models/expert_{0-4}_critic1.pth`：Critic1网络权重
- `models/expert_{0-4}_critic2.pth`：Critic2网络权重
- `logs/expert_{0-4}/training_log.json`：训练指标数据
- `logs/expert_{0-4}/training_curves.png`：训练曲线图

### 步骤3：评估专家性能

**选项A：评估单个专家**
```bash
python eval_expert.py --expert_id 0 --episodes 20
```

**选项B：批量评估所有专家**
```bash
python eval_all_experts.py --episodes 20
```

**输出**：
- `eval_results/expert_{0-4}_eval.json`：单个专家评估结果
- `eval_results/all_experts_comparison.csv`：所有专家对比数据
- `eval_results/all_experts_comparison.png`：对比图表

---

## 📊 关键数据结构

### 训练数据格式
```python
# clustered_training_data.csv
{
    'Time': datetime,           # 时间戳
    'Load_MW': float,          # 负荷 (MW)
    'Price_CNY_kWh': float,    # 电价 (元/kWh)
    'Wind_Gen_MW': float,      # 风电出力 (MW)
    'PV_Gen_MW': float,        # 光伏出力 (MW)
    'REN_Gen_MW': float,       # 总可再生出力 (MW)
    'Temperature_C': float,    # 温度 (℃)
    'Solar_W_m2': float,       # 太阳辐射 (W/m²)
    'Wind_Speed_m_s': float,   # 风速 (m/s)
    'Day_Label': int,          # 天气类型标签 (0-4)
    'Day_Index': int,          # 天序号 (0-364)
    'Hour': int                # 小时 (0-23)
}
```

### Transition字典（经验回放）
```python
transition = {
    'states': np.ndarray,      # (batch_size, 6)
    'actions': np.ndarray,     # (batch_size, 1)
    'rewards': np.ndarray,     # (batch_size, 1)
    'next_states': np.ndarray, # (batch_size, 6)
    'dones': np.ndarray        # (batch_size, 1)
}
```

### 评估结果字典
```python
eval_result = {
    'return': float,           # 平均回报
    'cost': float,            # 平均成本(元) 负数=盈利
    'curtail': float,         # 平均弃电(MWh)
    'import': float,          # 平均购电(MW)
    'export': float,          # 平均售电(MW)
    'ramp': float,            # 平均电网波动(MW)
    'episodes': int,          # 评估次数
    'deterministic': bool     # 是否确定性策略
}
```

---

## 🎯 关键设计要点

### 1. 为什么是1维动作空间？
- **简化问题**：从6维（风光×2+储能×2+电网×2）简化为1维（仅储能）
- **物理约束**：风光输出不可控，电网功率由功率平衡自动确定
- **训练效率**：1维动作空间训练速度快10倍以上

### 2. Episode为什么是4天？
- **跨日优化**：储能SOC需要跨日优化，单日episode无法学习长期策略
- **计算效率**：4天=96步，既能保证学习效果，又不会太慢
- **连续性**：SOC在episode内连续，episode间独立

### 3. 为什么训练5个专家？
- **天气聚类**：K-Means将365天聚为5类典型天气
- **策略差异化**：不同天气下的最优策略差异巨大
- **泛化能力**：单一策略难以适应所有天气类型

### 4. 奖励函数设计原理
```python
reward = -(w_cost × cost + w_ramp × ramp_penalty) / scale
```
- **负奖励**：最小化成本 → 最大化负成本（即奖励）
- **多目标**：同时优化经济性（成本）和电网友好性（波动）
- **归一化**：使用scale参数将不同量级的指标统一到相近范围

---

## 🔍 调试与故障排查

### 常见问题

**1. 训练不收敛**
- 检查学习率是否过大：`lr_actor`, `lr_critic`
- 增加warmup_steps：从1000增到5000
- 检查奖励scale是否合理

**2. GPU内存不足**
- 减小batch_size：从256降到128
- 减小buffer_size：从100000降到50000
- 减小hidden_dim：从256降到128

**3. 弃电量过高**
- 增大储能容量：`ts_mwh`, `eh_mw_th`, `st_mw_e`
- 增加弃电惩罚权重（需修改代码）
- 提高售电价格

**4. 模型路径错误**
- 确保在SAC目录下运行脚本
- 检查phase2_config.yaml中的路径配置
- 使用绝对路径或修改`_resolve_path()`函数

---

## 📈 性能基准

### 训练性能（RTX 2070, 8GB）
- 单个episode：约1.2秒
- 单个专家（1000 episodes）：约20分钟
- 全部5个专家：约1.5-2小时

### 评估性能
- 单个expert评估（20 episodes）：约2-3分钟
- 全部5个专家评估：约10-15分钟

### 经济性能（实际测试结果）
- 专家0（低可再生天）：4天收益30万元
- 专家4（高可再生天）：4天收益114万元
- **平均年化收益**：约7000万元

---

## 🚀 后续扩展

### 第三阶段：实时调度（Learning MPC）
- 在线天气识别 → 选择对应专家
- MPC微调专家决策到15分钟粒度
- 实时反馈修正

### 可能的改进
1. **增加弃电约束**：在奖励中显式惩罚弃电
2. **多目标优化**：使用Pareto前沿选择策略
3. **迁移学习**：用已训练专家初始化新专家
4. **集成学习**：融合多个专家的决策

---

## 📚 参考资料

### 论文算法
- SAC: Soft Actor-Critic (Haarnoja et al., 2018)
- 连续动作空间的off-policy RL算法
- 自动熵调整机制

### 代码来源
- 基础SAC实现：开源SAC-OS项目（6维动作版本）
- 改造要点：简化为1维动作 + 微电网物理模型

---

**文档版本**：v1.0  
**最后更新**：2024年（训练完成后）  
**维护者**：项目团队
