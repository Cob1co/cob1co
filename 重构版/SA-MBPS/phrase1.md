**“启发式算法外壳 + 边际效益（Marginal Utility）内核 + 场景聚合（Scenario Aggregation）”**的混合算法架构，正好解决了单纯启发式算法“盲目搜索”和单纯解析法“难以处理非线性约束”的痛点。

作为“甲方”视角，你关心的是**投资回报率（ROI）**和**供能稳定性**。

为了帮助你构建这个算法的**创新点**和**具体逻辑**，我为你设计了以下方案。我们将这个算法命名为：**基于边际效益制导的场景自适应启发式规划算法 (Marginal-Utility Guided Heuristic Planning with Scenario Adaptation, MUGH-SA)**。

---

### 一、 算法整体架构设计

这个算法分为三个层次：
1.  **输入层（场景聚合）**：处理你的CSV天气数据，提取典型日。
2.  **外层优化（启发式框架）**：负责生成和迭代“配置方案”（即：风、光、储罐、加热器、汽轮机的容量）。
3.  **内层评估（边际效益分析）**：这是**核心创新点**。不仅仅计算成本，还要计算“如果多装1kW设备能多赚多少钱”，并用这个信息去**修正**启发式算法的搜索方向。

### 二、 详细步骤与创新点植入

#### 第一步：数据预处理与场景聚合 (Scenario Aggregation)
你提供的CSV数据是全年的小时级数据（8760点）。直接优化计算量太大，单纯取平均值又会丢失极端天气（如极寒无光日，这对熔盐储能配置至关重要）。

*   **操作：** 使用 K-Means 或 K-Medoids 聚类算法。
*   **特征向量：** [日平均光照, 日平均风速, 日平均温度, 峰谷电价差]。
*   **输出：** 选出 $K$ 个典型场景（例如：夏季大风日、冬季无光日、春秋平稳日）以及 $M$ 个极端场景。
*   **权重：** 每个典型日代表全年多少天（概率 $p_s$）。

#### 第二步：构建“边际效益”评估核 (The Marginal Utility Core) —— 创新点所在

传统的启发式算法（如粒子群PSO、遗传算法GA）是随机变异的。我们要改写它的进化规则，让它**“沿着边际效益梯度的方向变异”**。

**1. 定义决策变量 $X$：**
$$X = [P_{wind}, P_{solar}, E_{salt}, P_{heater}, P_{turbine}]$$

**2. 运行简易调度（Operation Simulation）：**
对于每一个粒子（即一套配置方案），在所有聚合场景下运行一个简化的能量平衡策略（Rule-based Strategy，不需要太复杂，因为这是Phase 1）。

**3. 计算边际效益指标 (Marginal Utility Indicators, MUI)：**
这是算法的灵魂。我们定义以下边际指标：

*   **$MU_{gen}$ (发电边际效益)：**
    如果在某时刻弃风/弃光了，说明装机过剩，边际效益为负（因为有维护成本）；如果负荷没满足或需要从电网买高价电，说明装机不足，边际效益为正。
    $$MU_{wind/solar} = \sum_{s \in S} p_s \times (\text{电价} \times \text{有效增发量} - \text{LCOE}_{wind/solar})$$

*   **$MU_{store}$ (储能容量边际效益)：**
    检查熔盐罐是否经常**满充**导致弃电（说明容量不足，MU为正），或者经常**空置**（说明容量浪费，MU为负）。
    $$MU_{salt} = \sum_{s \in S} p_s \times (\text{避免的弃电价值} + \text{避免的缺电惩罚} - \text{储能单位成本})$$

*   **$MU_{power}$ (充放电功率边际效益)：**
    检查加热器是否经常达到额定功率上限但仍有弃电（需扩容加热器）；检查汽轮机是否经常满发但仍缺电（需扩容汽轮机）。

#### 第三步：构建启发式外壳 (Heuristic Shell)

这里建议使用 **改进型粒子群算法 (Improved PSO)** 或 **差分进化算法 (DE)**。我们将边际效益融入速度更新公式。

*   **传统PSO速度更新：**
    $$v_{new} = w \cdot v + c_1 \cdot r_1 \cdot (P_{best} - x) + c_2 \cdot r_2 \cdot (G_{best} - x)$$
    *(完全依赖历史最优和全局最优，带有盲目性)*

*   **你的创新公式 (MUGH-SA)：**
    $$v_{new} = w \cdot v + \dots + \mathbf{\alpha \cdot \text{Sigmoid}(MUI)}$$
    
    **解释：**
    *   如果某设备的 $MUI$ (边际效益) 是正值且很大，给它一个正向的“推力”，强制让下一代粒子增加该设备的配置。
    *   如果 $MUI$ 是负值（亏本），给它一个反向“推力”，减少配置。
    *   $\alpha$ 是“贪婪因子”，代表甲方对盈利的渴望程度。

### 三、 算法流程图解 (逻辑梳理)

1.  **输入**：天气CSV，典型设备单价（风/光/储/加热器/汽机），电价曲线。
2.  **场景生成**：生成 $S$ 个典型场景 + 极端场景。
3.  **初始化种群**：随机生成 $N$ 组配置方案。
4.  **循环迭代 (Loop)**：
    *   **For** 每一组配置方案：
        *   **For** 每一个场景：
            *   模拟运行（计算能量平衡）。
            *   记录：弃电量、缺电量、储能满/空次数、设备满载率。
        *   **End For**
        *   计算总成本 (NPV) 和 投资回收期。
        *   **计算边际效益向量 (MUI)**。
    *   **End For**
    *   更新全局最优解 (Best Solution)。
    *   **更新粒子位置**：使用 **MUI 修正后的公式** 更新下一代配置。
5.  **终止条件**：达到最大迭代次数或MUI趋近于0。
6.  **输出**：最优配置 $X^*$。

### 四、 为什么这个设计好？（论文/报告的卖点）

1.  **可解释性强（针对甲方）：** 纯黑盒算法算出结果，甲方会问“为什么要配这么大储能？”。你的算法可以回答：“因为在计算过程中，发现如果不配这么大，在场景3（极寒日）的边际损失远大于投资成本。”
2.  **收敛速度快：** 传统的启发式算法在多维空间是瞎找，你有“边际效益”作为向导，能迅速找到盈利平衡点。
3.  **适应性强：** 熔盐储能不仅仅是存电，它涉及到热-电转换（电加热器->热->汽轮机->电）。边际效益法能很好地解耦加热器功率和储热容量的复杂关系（有些算法容易把这两者混淆）。

### 五、 第一阶段代码实现的简要伪代码结构

```python
# 1. 场景聚合
class ScenarioManager:
    def aggregate(self, csv_data):
        # 使用K-Means聚类天气数据
        # 返回：典型日数据列表, 权重
        pass

# 2. 经济性与边际分析模型
class EconomicsEngine:
    def evaluate(self, configuration, scenarios):
        total_profit = 0
        marginal_utility = {
            'wind': 0, 'solar': 0, 'storage_cap': 0, 
            'heater_power': 0, 'turbine_power': 0
        }
        
        for day_data, weight in scenarios:
            # 运行简易调度逻辑
            ops_result = run_simple_dispatch(configuration, day_data)
            
            # 计算当前配置下的利润
            total_profit += ops_result['profit'] * weight
            
            # --- 核心创新：计算边际效益 ---
            # 譬如：如果有弃光，Solar的边际效益要扣分
            if ops_result['curtailed_solar'] > 0:
                marginal_utility['solar'] -= cost_per_kw * factor
            # 如果加热器一直满载，Heater的边际效益加分
            if ops_result['heater_full_load_hours'] > high_threshold:
                marginal_utility['heater_power'] += potential_profit_factor
            # ... 其他设备的逻辑 ...
            
        return total_profit, marginal_utility

# 3. 启发式算法外壳 (带修正的PSO)
class MUGH_Optimizer:
    def optimize(self):
        particles = initialize_particles()
        
        for iter in range(max_iter):
            for p in particles:
                profit, mu_vector = EconomicsEngine.evaluate(p.position, scenarios)
                
                # 更新个体最优和全局最优
                update_bests(p, profit)
                
                # --- 核心创新：利用MU修正速度 ---
                # 传统的PSO速度更新
                velocity_classic = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                # 边际效益引导项
                velocity_mu = alpha * mu_vector 
                
                # 最终速度
                p.velocity = velocity_classic + velocity_mu
                p.position += p.velocity
                
        return global_best_position
```

### 六、 衔接后续阶段

*   **衔接Phase 2 (RL调度)：** Phase 1 算出的配置是“硬约束”，传给Phase 2作为环境参数（Action Space的边界）。Phase 1为了速度用的是简易调度（Rule-based），Phase 2 用RL在这个配置下挖掘更精细的操作策略（比如利用分时电价套利）。
*   **衔接Phase 3 (实盘)：** 引入实时数据接口。

这个设计既满足了甲方的宏观需求，又有算法层面的微观创新，非常适合作为你的设计蓝图。