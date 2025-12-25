# 第二阶段：SAC日前调度策略训练

## 文件结构

```
SAC/
├── phase2_config.yaml          # 配置文件
├── prepare_training_data.py    # 数据预处理脚本
├── microgrid_env.py            # 微电网环境（1维动作+6维状态）
├── sac_agent.py                # SAC智能体
├── replay_buffer.py            # 经验回放缓冲区
├── train_expert.py             # 训练单个专家
├── train_all_experts.py        # 批量训练5个专家
├── models/                     # 模型保存目录
│   ├── expert_0_actor.pth
│   ├── expert_0_critic1.pth
│   ├── expert_0_critic2.pth
│   └── ...
├── logs/                       # 训练日志
│   ├── expert_0/
│   │   ├── training_log.json
│   │   └── training_curves.png
│   └── ...
└── clustered_training_data.csv # 预处理后的训练数据
```

---

## 使用流程

### **步骤1：数据预处理**

运行数据预处理脚本，生成训练数据：

```bash
python prepare_training_data.py
```

**输出**：
- `clustered_training_data.csv`：包含天气聚类标签和预计算的风光出力
- 更新 `phase2_config.yaml` 中的容量配置

**注意**：运行前需确保：
- `data/data2023.csv` 存在
- 第一阶段的容量配置已填入 `prepare_training_data.py`

---

### **步骤2：训练专家策略**

#### 方式A：批量训练所有5个专家（推荐）

```bash
python train_all_experts.py
```

训练完成后，会在 `models/` 目录生成：
- `expert_0_actor.pth` ~ `expert_4_actor.pth`（5个Actor网络）
- `expert_0_critic1.pth` ~ `expert_4_critic1.pth`（5个Critic1网络）
- `expert_0_critic2.pth` ~ `expert_4_critic2.pth`（5个Critic2网络）

#### 方式B：单独训练某个专家

```bash
python train_expert.py --expert_id 0
```

---

### **步骤3：检查训练结果**

查看训练日志和曲线：

```bash
# 日志保存在
logs/expert_0/training_log.json
logs/expert_0/training_curves.png
```

---

## 配置说明

### **核心参数（phase2_config.yaml）**

```yaml
environment:
  episode_days: 4               # 每个episode天数（4天=96步）
  state_dim: 6                  # 状态维度
  action_dim: 1                 # 动作维度
  soc_min: 0.1                  # SOC下限
  soc_max: 0.9                  # SOC上限

objective:
  w_cost: 1.0                   # 成本权重
  w_ramp: 0.15                  # 波动权重

sac:
  hidden_dim: 256               # 隐藏层维度
  lr_actor: 0.0003              # Actor学习率
  lr_critic: 0.0003             # Critic学习率
  gamma: 0.99                   # 折扣因子
  tau: 0.005                    # 软更新系数
  buffer_size: 100000           # 回放池大小
  batch_size: 256               # 批次大小
  max_episodes: 1000            # 最大训练轮数

training:
  num_experts: 5                # 专家数量
  use_gpu: true                 # 是否使用GPU
```

---

## 设计要点

### **1. 状态空间（6维）**
```python
[0] load / max_load           # 负荷（归一化）
[1] pv / max_pv               # 光伏出力（归一化）
[2] wind / max_wind           # 风电出力（归一化）
[3] soc                       # 储能SOC
[4] price / max_price         # 电价（归一化）
[5] p_grid_prev / max_grid    # 上时刻电网功率（归一化）
```

### **2. 动作空间（1维）**
```python
action ∈ [-1, 1]  # 储能充放电指令
# > 0: 充电模式（启动电加热器）
# < 0: 放电模式（启动汽轮机）
# 电网功率自动平衡
```

### **3. 奖励函数（2项加权）**
```python
reward = -(w_cost * 运行成本 + w_ramp * 电网波动) / cost_scale
```

**为什么不单独优化弃电？**
- 弃电会损失售电收益，已包含在成本中
- 优化成本自然会减少弃电

---

## 常见问题

### Q1: 训练时显存不足怎么办？
A: 修改 `phase2_config.yaml`：
```yaml
sac:
  batch_size: 128  # 减小批次大小（默认256）
  hidden_dim: 128  # 减小网络规模（默认256）
```

### Q2: 训练速度慢怎么办？
A: 
1. 确保 `use_gpu: true`
2. 减少 `max_episodes`（如改为500）
3. 减少 `episode_days`（如改为2天）

### Q3: 如何评估训练好的模型？
A: 查看训练曲线 `logs/expert_*/training_curves.png`，重点关注：
- Return曲线是否上升并收敛
- Cost曲线是否下降
- 最后100个episode的平均表现

### Q4: 如何调参？
A: 优先调整这些参数：
1. `w_cost` 和 `w_ramp` 的相对权重
2. `lr_actor` 和 `lr_critic`（学习率）
3. `gamma`（折扣因子，影响长期规划）

---

## 依赖安装

```bash
pip install torch numpy pandas pyyaml scikit-learn tqdm matplotlib
```

---

## 下一步

训练完成后，可以：
1. 运行全年评估脚本（第三阶段）
2. 可视化5个专家的调度策略差异
3. 与规则型调度对比性能
