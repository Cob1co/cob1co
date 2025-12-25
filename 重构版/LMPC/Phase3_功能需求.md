# Phase 3：Learning MPC 功能需求说明

## 1. 外部依赖与输入输出

### 1.1 依赖前两阶段

- **Phase1**：容量与物理参数
  - 从 `models/config.yaml` 读取设备效率、SOC 上下限、容量等。
- **Phase2**：SAC 专家策略
  - 5 个专家 Actor 网络：
    - 路径：`SAC/models/expert_{0-4}_actor.pth`
  - 训练数据：`SAC/clustered_training_data.csv`（用于归一化与天气聚类一致）。

### 1.2 数据输入

- 历史真实数据：
  - `data/data2023.csv`（小时）  
  - `data/realtime2024.csv` 或 `generate_realtime_2024.py` 生成的 15min 数据

- 预测数据：
  - `LMPC/data/forecast_2023_8h_training.pkl`  
  - `LMPC/data/forecast_2024_8h_testing.pkl`

### 1.3 输出

- 训练阶段：
  - `LMPC/data/transformer_training_data.pkl`：包含 `(state_seq (24,12), optimal_alphas (3,))` 样本。
  - `LMPC/models/transformer_weights.pth`：训练好的 Transformer 权重控制器。

- 评估 / 运行阶段：
  - `LMPC/logs/realtime_results/`：7 天连续仿真结果（SOC、电网功率、成本、权重轨迹等）。

---

## 2. 功能模块需求

### 2.1 天气分类器（WeatherClassifier）

- **输入**：
  - 未来 24h 的预测序列（光照、风速、负荷、电价），或过去 24h 实测数据。
- **输出**：
  - 专家 ID ∈ {0,1,2,3,4}。
- **要求**：
  - 聚类特征与 Phase2 K-means 特征完全一致（10 维：均值/标准差/峰值 + 电价）。
  - 提供 `classify_from_forecast`、`classify_from_history` 两种接口。

### 2.2 专家接口（ExpertInterface）

- **输入**：
  - `expert_id`，当前 `soc`，上一时刻 `grid_power`；  
  - 未来 8 小时预测：`load(kW), pv(MW), wind(MW), price(元/kWh)`，长度 32（15min 分辨率）。
- **输出**：
  - 参考计划：`{'soc': np.array(32,), 'grid_power': np.array(32,)}`。
- **功能需求**：
  1. 调用对应专家 Actor（确定性策略），逐步滚动模拟 32 步；
  2. 物理模型（充放电、热损、功率平衡）与 Phase2 `MicrogridEnv` 逻辑一致；
  3. 使用从 `SAC/clustered_training_data.csv` 统计得到的归一化边界（避免魔术数字）；
  4. 专家 ID 切换时，对前 4 步参考轨迹做线性插值平滑。

### 2.3 MPC 控制器（MPCController）

- **输入**：
  - 当前状态：`{'soc', 'grid_power_prev'}`；  
  - 预测窗口 H 步：`{'load', 'pv', 'wind', 'price'}`，H≈16；  
  - 专家参考计划 H 步：`{'soc', 'grid_power'}`；  
  - 动态权重：`{'alpha_soc', 'alpha_grid', 'alpha_cost'}`。
- **输出**：
  - `{'action': float, 'soc_plan': (H,), 'grid_plan': (H,), 'cost': float, 'status': str}`。
- **目标函数需求**（对齐需求文档）：
  - 最小化：
    - `α_soc * w_base_soc * ||soc - soc_ref||²`  
    - `α_grid * w_base_grid * ||grid - grid_ref||²`  
    - `α_cost * w_base_cost * 运行成本`  
    - `w_ramp * ||Δgrid||²`  
    - `w_curtail * curtailment`（若显式建模弃电）  
  - 基础权重从 `phase3_config.yaml/mpc/base_weights` 读取。

- **约束需求**：
  - SOC 动力学：`soc[k+1] = soc[k] + (充电/放电能量变化) / 容量`  
  - SOC ∈ [0.1, 0.9]；动作、设备功率、网侧功率在容量范围内；  
  - 功率平衡：`P_grid = Load - PV - Wind + P_charge + P_curtail`。

### 2.4 特征提取器（FeatureExtractor）

- **输入**：
  - 历史 24 步（6 小时）状态序列和预测/实测序列。
- **输出**：
  - 单步 12 维特征向量。
- **特征要求**：
  - 4 维系统状态：`soc`, `grid_power`, `soc_deviation`, `grid_deviation`；  
  - 3 维预测误差：`load_error`, `pv_error`, `wind_error`（过去4步平均，因果性）  
  - 2 维时间特征：`hour`, `is_peak`;  
  - 3 维专家特征：`expert_id`, `expert_confidence`, `time_since_switch`。

### 2.5 数据收集器（DataCollector）

- **输入**：
  - 2023 年真实数据（1h 或 15min），及对应预测数据（含误差）。
- **输出**：
  - `LMPC/data/transformer_training_data.pkl`，每个样本包含：
    - `state_sequence`: (24,12)  
    - `optimal_weights`: (3,)  
    - 可选：记录 `reward`, 单项成本/误差指标。

- **算法需求**：
  1. 对每个采样时刻 t（可每小时采样一次）：
     - 提取最近 24 步 12 维特征 → `state_seq`；
     - 基于预测 H 步 + 专家计划 H 步，构造 MPC 输入；
  2. 对候选权重组合（27 组）：
     - 调用 MPCController，计算**和在线完全一致的目标值 J**；
     - 按 `reward = -J` 选取 reward 最大的一组作为 `optimal_weights`；
  3. 支持并行或分段运行，便于长时间离线计算。

### 2.6 Transformer 训练模块

- **输入**：
  - `transformer_training_data.pkl`。
- **输出**：
  - 最佳模型：`LMPC/models/transformer_weights.pth`；
  - 训练日志和曲线（可选）。

- **训练需求**：
  - 模型结构：4 层 Transformer Encoder，8 头注意力，MLP 输出 3 维；  
  - 损失函数：MSE；  
  - 优化器：AdamW + CosineAnnealingLR；  
  - 支持加权采样（权重偏离 1.0 越大，采样权重越高）；  
  - 支持 early stopping（patience≈10）。

### 2.7 实时运行脚本

- **目标脚本**：`LMPC/scripts/run_lmpc_control.py`
- **需求**：
  - 统一入口，读取 `phase3_config.yaml`；
  - 支持指定测试天数（例如 7 天）；  
  - 内部完整跑通：
    - 读 2024 实时数据 + 预测数据；  
    - 每 15min：天气分类 → 专家计划 → 特征提取 → Transformer 权重 → MPC 解 → 状态更新；  
    - 记录关键指标到指定日志目录。

---
