# 第三阶段：基于 Transformer 动态权重的 Learning MPC（开题报告大纲）

> 本节作为研究生开题报告中“第三阶段：Learning MPC”的独立部分，重点说明：
>
> - 问题建模与优化目标的物理和工程合理性；
> - 三个核心目标（购电成本、弃风弃光、净电荷波动）及其权重设计思路；
> - 储能充放电功率与 SOC 寿命等约束的扩展建模；
> - Transformer 动态权重模型与 MPC 优化器之间的耦合关系；
> - 与前两阶段（改进 PSO、SAC 多专家）的整体架构衔接与创新点。

---

## 1 研究背景与阶段定位

- **阶段一（PSO 容量配置）**：在给定风光资源与负荷曲线的前提下，利用改进 PSO 搜索最优设备容量向量
  $x^* = (P_{\text{wind}}^{\max}, P_{\text{pv}}^{\max}, E_{\text{ts}}^{\max}, P_{\text{EH}}^{\max}, P_{\text{ST}}^{\max})$，保证全年运行成本最小、弃风弃光可控。
- **阶段二（SAC 多专家日内调度）**：在容量 $x^*$ 固定的条件下，针对 $K=5$ 类典型天气（$\text{Day\_Label}\in\{0,\dots,4\}$）分别训练 SAC 专家策略 $\pi^{(k)}_\theta(a_t\mid s_t)$，输出 **小时级参考轨迹**：
  - 参考 SOC：$\{SOC^{\text{ref}}_{t}\}$；
  - 参考电网功率：$\{P^{\text{ref}}_{\text{grid},t}\}$。
- **阶段三（学习型 MPC，Learning MPC）**：在 15 min 时间尺度上，引入基于 Transformer 的动态权重控制器 $f_\phi(\cdot)$，结合预测数据与专家参考轨迹，在线求解 MPC 问题：
  - **决策变量**：当前与预测窗口内的储能充放电功率、SOC、网侧功率等；
  - **目标函数**：综合权衡“跟踪专家参考轨迹”“降低运行成本”“抑制网侧波动与弃电”；
  - **动态权重**：由 Transformer 根据历史运行状态与预测误差自适应给出。

从控制层次上看，三个阶段构成了一个“**容量规划 → 日前策略 → 实时滚动优化**”的三层闭环：

1. 改进 PSO 决定 **系统边界与物理约束**；
2. SAC 专家给出 **在该边界内的优良策略先验**；
3. Learning MPC 在预测误差和实时扰动下，对 SAC 策略进行 **局部修正与细化**。

---

## 2 物理系统与变量定义（15 min 级别）

时间离散为 $\Delta t = 15\,\text{min} = 0.25\,\text{h}$，第 $t$ 步对应的主要变量定义如下：

- **状态变量**：
  - 储能能量 $E_t$，对应 SOC 为
    $$
    SOC_t = \frac{E_t}{E_{\max}}\,,\quad SOC_t \in [SOC_{\min},\ SOC_{\max}]\,.
    $$
  - 上一时刻网侧有功功率 $P^{\text{grid}}_{t-1}$（MW）。
- **可控变量**：
  - 储能充电功率 $P^{\text{ch}}_t \ge 0$（MW）；
  - 储能放电功率 $P^{\text{dis}}_t \ge 0$（MW）；
  - 为避免同时充放电，可通过互斥约束或等效变量化简（见第 4 节）。
- **外部输入**：
  - 预测负荷功率 $\hat P^{\text{load}}_{t+k}$；
  - 预测风电功率 $\hat P^{\text{wind}}_{t+k}$；
  - 预测光伏功率 $\hat P^{\text{pv}}_{t+k}$；
  - 预测电价 $\hat \pi_{t+k}$（元/kWh）。
- **派生变量**：
  - 网侧功率（正为购电、负为上网）：
    $$
    P^{\text{grid}}_t = P^{\text{load}}_t - P^{\text{pv}}_t - P^{\text{wind}}_t + P^{\text{ch}}_t - P^{\text{dis}}_t - P^{\text{curt}}_t\,.
    $$
  - 弃风弃光功率：
    $$
    P^{\text{curt}}_t = \max\Bigl\{0,\ P^{\text{pv}}_t + P^{\text{wind}}_t - P^{\text{load}}_t - P^{\text{ch}}_t\Bigr\} \,.
    $$

> 说明：在实际实现中，$P^{\text{load}}_t$、$P^{\text{pv}}_t$、$P^{\text{wind}}_t$ 由预测值 $\hat P_{\cdot}$ 代替；上述功率平衡与弃电定义与阶段一、二保持一致，保证三阶段模型物理自洽。

---

## 3 Learning MPC 优化模型

### 3.1 决策变量向量

在每个 15 min 时刻 $t$，MPC 以预测时域长度 $H$（例如 $H=32$，即 8 小时）为窗口，求解如下决策变量集合：

$$
\boldsymbol x_t = \Bigl\{ P^{\text{ch}}_{t+k},\ P^{\text{dis}}_{t+k},\ SOC_{t+k},\ P^{\text{grid}}_{t+k},\ P^{\text{curt}}_{t+k} \Bigr\}_{k=0}^{H-1}\,.
$$

其中第一步 $k=0$ 的充放电功率 $P^{\text{ch}}_{t}, P^{\text{dis}}_t$ 将作为当前控制输入下发，余下步骤用于构建滚动计划。

### 3.2 目标函数结构

阶段三的核心思想是：

- 用 MPC 在约束条件下优化运行；
- 用 Transformer 输出的 **动态权重** $\boldsymbol{\alpha}_t = (\alpha^{\text{soc}}_t,\ \alpha^{\text{grid}}_t,\ \alpha^{\text{cost}}_t)$ 调节不同目标的相对重要性；
- 固定权重 $w_\cdot$ 则由工程经验与配置文件 `phase3_config.yaml/mpc/base_weights` 给出。

给定 SAC 专家参考轨迹 $SOC^{\text{ref}}_{t+k}, P^{\text{ref}}_{\text{grid},t+k}$，以及电价 $\hat \pi_{t+k}$，MPC 目标函数写为：

$$
\begin{aligned}
J_t(\boldsymbol x_t; \boldsymbol{\alpha}_t)
= \sum_{k=0}^{H-1} \Bigl[&
\underbrace{\alpha^{\text{soc}}_t\, w^{\text{soc}}\bigl(SOC_{t+k} - SOC^{\text{ref}}_{t+k}\bigr)^2}_{\text{SOC 跟踪项}} \\
&+ \underbrace{\alpha^{\text{grid}}_t\, w^{\text{grid}}\bigl(P^{\text{grid}}_{t+k} - P^{\text{ref}}_{\text{grid},t+k}\bigr)^2}_{\text{电网功率跟踪项}} \\
&+ \underbrace{\alpha^{\text{cost}}_t\, w^{\text{cost}}\, C_{t+k}}_{\text{运行成本项}} \\
&+ \underbrace{w^{\text{ramp}}\bigl(P^{\text{grid}}_{t+k} - P^{\text{grid}}_{t+k-1}\bigr)^2}_{\text{功率爬坡惩罚}} \\
&+ \underbrace{w^{\text{curt}}\, P^{\text{curt}}_{t+k}}_{\text{弃风弃光惩罚}} \\
&+ \underbrace{w^{\text{smooth}}\bigl(P^{\text{ch}}_{t+k} - P^{\text{ch}}_{t+k-1}\bigr)^2}_{\text{充放电平滑项}}\Bigr] \,.
\end{aligned}
$$

其中：

- $C_{t+k}$ 为第 $t+k$ 步的购电成本（可写为）
  $$
  C_{t+k} = \hat \pi_{t+k} \cdot \max\{0, P^{\text{grid}}_{t+k}\} \cdot \Delta t\,,
  $$
  若存在售电，可在此基础上减去上网收益项；
- $w^{\text{soc}}, w^{\text{grid}}, w^{\text{cost}}, w^{\text{ramp}}, w^{\text{curt}}, w^{\text{smooth}}$ 为 **基础权重**，在开题阶段可直接引用 `phase3_config.yaml` 中给出的经验值，例如：
  $$
  w^{\text{soc}}=5.0,\ w^{\text{grid}}=2.0,\ w^{\text{cost}}=1.0,\ w^{\text{ramp}}=1.0,\ w^{\text{curt}}=2.0,\ w^{\text{smooth}}=1.0\,;
  $$
- $\alpha^{\text{soc}}_t, \alpha^{\text{grid}}_t, \alpha^{\text{cost}}_t$ 为 Transformer 输出的 **动态权重因子**，满足
  $$
  \alpha^{(i)}_t \in [\alpha_{\min}, \alpha_{\max}] = [0.5, 2.0]\,,\quad i\in\{\text{soc},\text{grid},\text{cost}\}\,.
  $$

### 3.3 目标函数与权重设计的合理性

1. **目标函数构建逻辑**：
   - SOC 跟踪项保证 LMPC 不偏离 SAC 专家给出的“长期策略”，避免反复深充深放；
   - 网侧功率跟踪项体现对电网友好性，尽量遵循专家给出的“平滑负荷曲线”；
   - 成本项直接反映购电开销，是经济性指标；
   - 爬坡与弃电惩罚项兼顾电网安全和可再生能源利用率；
   - 充放电平滑项则限制指令剧烈变化，有利于设备寿命与实际可实现性。

2. **三个动态权重的工程含义**：
   - $\alpha^{\text{soc}}_t$：当 SOC 偏离安全区间或预测误差较大时，提高该权重，使控制器更“保守”，优先保证储能安全与长周期策略；
   - $\alpha^{\text{grid}}_t$：在电网约束严格（例如峰时段、并网点限额较低）时提高该权重，抑制网侧波动；
   - $\alpha^{\text{cost}}_t$：在电价高企或可再生发电充裕时提高该权重，从而更激进地利用储能削峰填谷、减少购电成本。

3. **相对重要性的选择依据**：
   - 在典型工业微电网场景下，**供电安全与电网友好性** 往往是硬约束，其次才是经济性；因此基础权重设置为
     $$
     w^{\text{soc}} > w^{\text{grid}} > w^{\text{cost}}\,.
     $$
   - 动态权重 $\boldsymbol{\alpha}_t$ 在 $[0.5, 2.0]$ 的相对窄区间内调节，保证优化导向不会完全颠倒，而是在“安全优先”的框架内做局部折中。

4. **与三目标优化的一致性**：
   - 购电成本最低、弃风弃光率最低与净负荷波动最小三个目标，在上述目标函数中分别由 $C_{t+k}$、$P^{\text{curt}}_{t+k}$ 与 $\bigl(P^{\text{grid}}_{t+k} - P^{\text{grid}}_{t+k-1}\bigr)^2$ 体现；
   - 通过基础权重 $w_\cdot$ 确定“长期的工程优先级”，再由 $\boldsymbol{\alpha}_t$ 在不同运行工况（电价时段、天气类型、预测误差大小等）下做 **局部重加权**，从而使优化结果既符合工程常识，又能体现“情景自适应”。

---

## 4 约束条件与储能功率特性建模

### 4.1 SOC 动力学与安全约束

假设储能等效容量为 $E_{\max}$，电加热器效率为 $\eta_{\text{ch}}$，汽轮机效率为 $\eta_{\text{dis}}$，热损失率为 $\lambda$，则第 $t+k$ 步到 $t+k+1$ 步的能量平衡可写为：

$$
E_{t+k+1} = E_{t+k} + \bigl(\eta_{\text{ch}} P^{\text{ch}}_{t+k} - \frac{1}{\eta_{\text{dis}}} P^{\text{dis}}_{t+k}\bigr) \Delta t - \lambda E_{t+k} \Delta t\,.
$$

对应的 SOC 更新为：

$$
SOC_{t+k+1} = SOC_{t+k} + \frac{\eta_{\text{ch}} P^{\text{ch}}_{t+k} - P^{\text{dis}}_{t+k}/\eta_{\text{dis}}}{E_{\max}}\,\Delta t - \lambda SOC_{t+k} \Delta t\,.
$$

安全约束：

$$
SOC_{\min} \le SOC_{t+k} \le SOC_{\max}\,,\quad \forall k=0,\dots,H\,.
$$

### 4.2 充放电功率与工况相关约束

传统模型通常只考虑恒定的充放电功率上限：

$$
0 \le P^{\text{ch}}_{t+k} \le \bar P^{\text{ch}}\,,\quad 0 \le P^{\text{dis}}_{t+k} \le \bar P^{\text{dis}}\,.
$$

为更贴近实际运行安全与寿命管理，在开题设计阶段可以引入 **工况相关的功率限值**：

1. **SOC 深度影响（DoD 约束）**：在 SOC 接近上下限时，允许的充放电功率应降低：
   $$
   P^{\text{dis}}_{t+k} \le \bar P^{\text{dis}} \cdot f_{\text{DoD}}(SOC_{t+k})\,,\quad
   P^{\text{ch}}_{t+k} \le \bar P^{\text{ch}} \cdot f_{\text{DoD}}(SOC_{t+k})\,,
   $$
   其中 $f_{\text{DoD}}(\cdot) \in (0,1]$ 是随 SOC 变化的折减系数，例如分段线性函数，在 $SOC$ 接近 $SOC_{\min}$ 或 $SOC_{\max}$ 时逐渐减小。

2. **温度工况影响**：若可获得设备温度 $T_t$，可进一步引入温度折减系数：
   $$
   P^{\text{dis}}_{t+k} \le \bar P^{\text{dis}} \cdot f_T(T_{t+k})\,,\quad
   P^{\text{ch}}_{t+k} \le \bar P^{\text{ch}} \cdot f_T(T_{t+k})\,,
   $$
   以反映高温或低温下电池/储热系统可安全输出的功率降低。

3. **互斥与方向切换约束**：
   - 可以通过二进制变量 $u_{t+k} \in \{0,1\}$ 表示充放电模式：
     $$
     P^{\text{ch}}_{t+k} \le u_{t+k}\, \bar P^{\text{ch}}\,,\quad
     P^{\text{dis}}_{t+k} \le (1-u_{t+k})\, \bar P^{\text{dis}}\,;
     $$
   - 或在工程实现中采用“等效单一功率变量 $P^{\text{ts}}_{t+k} \in [-\bar P^{\text{dis}}, \bar P^{\text{ch}}]$” 的方式简化模型，互斥关系由控制逻辑保证。

> 虽然当前代码实现中未显式建模温度与 DoD 对功率的影响，但在开题报告中给出上述扩展形式，可以体现模型在安全性与寿命管理层面的**可扩展性与工程完整性**。

### 4.3 功率平衡与弃电约束

功率平衡关系：

$$
P^{\text{grid}}_{t+k} = P^{\text{load}}_{t+k} - P^{\text{pv}}_{t+k} - P^{\text{wind}}_{t+k} + P^{\text{ch}}_{t+k} - P^{\text{dis}}_{t+k} - P^{\text{curt}}_{t+k}\,.
$$

弃风弃光功率与网侧功率均需满足：

$$
P^{\text{curt}}_{t+k} \ge 0\,,\quad P^{\text{grid}}_{\min} \le P^{\text{grid}}_{t+k} \le P^{\text{grid}}_{\max}\,.
$$

其中 $P^{\text{grid}}_{\min}, P^{\text{grid}}_{\max}$ 由并网点能力或合同约定给出。

---

## 5 Transformer 动态权重模型

### 5.1 输入特征构造

在时间 $t$，特征提取器将最近 $L=24$ 步（约 6 小时）的历史数据与预测信息编码为矩阵
$$
\boldsymbol Z_t \in \mathbb{R}^{L \times d}\,, \quad d=12\,.
$$

每一行对应一步时刻的 12 维特征向量，例如：

1. **系统状态相关特征（4 维）**：
   - $z^{(1)} = SOC$
   - $z^{(2)} = P^{\text{grid}}$
   - $z^{(3)} = SOC - SOC^{\text{ref}}$
   - $z^{(4)} = P^{\text{grid}} - P^{\text{ref}}_{\text{grid}}$
2. **预测误差相关特征（3 维）**：
   - $z^{(5)} = e^{\text{load}}$：负荷预测误差的滚动平均；
   - $z^{(6)} = e^{\text{pv}}$：光伏预测误差；
   - $z^{(7)} = e^{\text{wind}}$：风电预测误差；
3. **时间与工况特征（2 维）**：
   - $z^{(8)} = \text{hour}/24$：当前小时归一化；
   - $z^{(9)} = \mathbf{1}_{\text{peak}}$：是否处于电价高峰时段；
4. **专家相关特征（3 维）**：
   - $z^{(10)} = \text{expert\_id}/4$：当前选用的 SAC 专家编号；
   - $z^{(11)} = \text{expert\_confidence}$：根据专家在该天气类型上的长期表现估计的“信心度”；
   - $z^{(12)} = \text{time\_since\_switch}$：距离上一次专家切换已经过去的步数。

> 这些特征保证模型既“看见”物理状态与参考轨迹的偏差，又“感知”预测质量、时间段与专家切换情况，从而有足够信息去学习如何调整权重。

### 5.2 标签生成：离线网格搜索

由于现实中没有“理想权重”作为直接监督信号，本阶段采用 **离线 MPC 网格搜索** 生成标签：

1. 给定某一时刻 $t$ 的特征矩阵 $\boldsymbol Z_t$，枚举一组候选权重
   $$
   \mathcal{A} = \bigl\{ (\alpha^{\text{soc}},\alpha^{\text{grid}},\alpha^{\text{cost}}) : \alpha^{(i)} \in \{0.5, 1.0, 1.5\} \bigr\}\,,
   $$
   共 $3^3 = 27$ 组；
2. 对于每一组 $\boldsymbol{\alpha} \in \mathcal{A}$，调用 MPC 求解器，计算统一定义的 $H$ 步累计代价
   $$
   J_t(\boldsymbol x_t; \boldsymbol{\alpha})\,;
   $$
3. 将代价最小对应的权重
   $$
   \boldsymbol{\alpha}_t^{\ast} = \arg\min_{\boldsymbol{\alpha} \in \mathcal{A}} J_t(\boldsymbol x_t; \boldsymbol{\alpha})
   $$
   作为该样本的“最优权重标签”。

通过在 2023 年全年数据上离线滚动仿真，可以构造出训练集：

$$
\mathcal{D} = \bigl\{ (\boldsymbol Z_t, \boldsymbol{\alpha}_t^{\ast}) \bigr\}_{t=1}^N\,.
$$

### 5.3 Transformer 网络结构与损失函数

采用标准 Transformer Encoder 结构：

- 输入序列长度 $L=24$，特征维度 $d=12$，通过线性投影到模型维度 $d_{\text{model}}=128$；
- 堆叠 $N_\ell=4$ 个 Encoder 层，每层包含多头自注意力（$n_{\text{head}}=8$）和前馈网络；
- 对最终序列做池化（例如取 $[\text{CLS}]$ token 或平均池化）得到全局表示 $h_t \in \mathbb{R}^{d_{\text{model}}}$；
- 通过多层感知机输出三维权重向量：
  $$
  \hat{\boldsymbol{\alpha}}_t = g_\phi(h_t) \in \mathbb{R}^3\,,\quad \hat{\boldsymbol{\alpha}}_t \in [0.5,2.0]^3 \text{ 通过 Sigmoid+线性变换约束。}
  $$

训练目标为最小化均方误差损失：

$$
\mathcal{L}(\phi) = \frac{1}{N} \sum_{t=1}^{N} \bigl\| \hat{\boldsymbol{\alpha}}_t - \boldsymbol{\alpha}_t^{\ast} \bigr\|_2^2\,.
$$

为提升对“极端工况”的关注度，可对样本加权，例如权重偏离 $1.0$ 越大的样本给予更高采样权重，以强调 Transformer 对“非常态情况”的学习能力。

### 5.4 训练数据的物理一致性与覆盖性

- **物理一致性**：所有训练样本均来自“专家 + MPC”在真实历史数据上的闭环仿真，MPC 内部严格满足功率平衡与 SOC 约束，因此标签 $\boldsymbol{\alpha}_t^{\ast}$ 所对应的轨迹天然符合物理约束；
- **覆盖性**：通过在全年数据上以一定时间步长（例如每小时）采样，并在配置文件中设置足够多的采样天数 `num_days`，保证不同季节、不同负荷水平、不同天气条件下的工况都被覆盖；
- **代表性**：在构造训练集时，可以对具有大预测误差或约束紧张的时段进行过采样，使模型更关注“困难场景”，这在工程上更有价值。

---

## 6 三阶段算法耦合关系与系统结构

### 6.1 三种算法的角色分工

1. **改进 PSO（Phase1）**：
   - 决定容量向量 $x^*$ 及相关物理参数，是后续两个阶段的**约束边界与物理基础**；
   - 输出：风光装机、储热容量、电加热与汽轮机功率上限等。

2. **SAC 多专家（Phase2）**：
   - 在给定容量 $x^*$ 及天气标签 $\text{Day\_Label}$ 下，学习不同典型日的“**日内最优调度策略**”；
   - 输出：按专家编号 $k$ 区分的参考 SOC 轨迹 $SOC^{\text{ref},(k)}_t$ 与网侧功率轨迹 $P^{\text{ref},(k)}_{\text{grid},t}$；
   - 为 Phase3 提供**长期趋势与人造“专家经验”**。

3. **Transformer + MPC（Phase3）**：
   - 以 SAC 参考轨迹为“软目标”，在实际运行与预测误差的约束下做 **短周期滚动优化**；
   - Transformer 学习“在什么情况下需要偏重安全/电网/成本”，MPC 则在给定权重下求解凸优化问题；
   - 共同实现对专家策略的 **实时修正与细化**。

### 6.2 系统结构框图（文字描述，便于绘图）

从数据与信号流角度，可画出如下结构（供 PPT/论文绘制）：

1. **输入层**：
   - 历史测量：$\{SOC, P^{\text{grid}}, P^{\text{pv}}, P^{\text{wind}}, P^{\text{load}}\}$；
   - 预测序列：$\hat P^{\text{pv}}, \hat P^{\text{wind}}, \hat P^{\text{load}}, \hat \pi$；
   - 天气聚类模型 $k$-means 与 SAC 专家模型集合 $\{\pi^{(k)}_\theta\}_{k=0}^4$。

2. **专家层（小时级）**：
   - WeatherClassifier 根据预测/历史序列给出天气标签 $\text{Day\_Label}$；
   - 选取对应 SAC 专家，生成未来 8 小时的参考轨迹 $SOC^{\text{ref}}$, $P^{\text{ref}}_{\text{grid}}$。

3. **战术层（Transformer 权重控制器）**：
   - FeatureExtractor 从过去 6 小时数据构造 $\boldsymbol Z_t$；
   - Transformer Controller 输出动态权重 $\boldsymbol{\alpha}_t$。

4. **执行层（MPC 控制器）**：
   - 接收 $\boldsymbol{\alpha}_t$、预测序列与专家参考轨迹；
   - 求解 MPC，输出当前步最优控制量 $P^{\text{ch}}_t, P^{\text{dis}}_t$；
   - 更新微电网物理状态，进入下一步循环。

5. **评估与日志层**：
   - 记录运行成本、SOC/网侧功率跟踪误差、约束违反率、权重轨迹等指标；
   - 与基线策略（仅专家、固定权重 MPC）进行对比。

在 PPT 中可以将这三层画成自上而下的框图，用不同颜色标注三种算法（PSO、SAC、Transformer+MPC），突出“**先做容量 → 再学策略 → 最后学权重**”的层次结构。

---

## 7 本阶段创新点总结（供开题使用）

1. **提出“专家策略 + 学习型权重 + MPC”的三层协同控制框架**：
   - 与传统“固定权重 MPC”相比，引入 Transformer 学习权重调节规律，使 MPC 目标函数能根据运行工况自适应调整。

2. **利用 SAC 多专家作为“策略先验”，Learning MPC 只做局部修正**：
   - 避免直接在 15 min 级别上训练大规模 RL，降低样本需求与训练难度；
   - 通过跟踪 SAC 参考轨迹，保持整体操作风格的一致性。

3. **构建基于离线 MPC 网格搜索的监督标签生成方法**：
   - 在保证物理约束的前提下，用统一目标函数 $J_t$ 从历史数据中推导“最优权重”，从而将复杂的多目标调度问题转化为标准的监督学习任务。

4. **引入预测误差、专家切换等高维特征，丰富权重决策依据**：
   - 不仅依赖 SOC 与电网功率，还显式考虑预测质量、时间段、专家信心等信息，使权重调节更贴近实际运行经验。

5. **在建模层面预留了与设备寿命、安全工况相关的拓展接口**：
   - 通过引入 DoD 与温度相关的功率约束函数 $f_{\text{DoD}}(\cdot)$、$f_T(\cdot)$，为后续将寿命模型融入优化提供基础。

6. **以统一的模型和指标串联三阶段工作**：
   - 同一套功率平衡方程、SOC 约束与成本定义贯穿三阶段，确保阶段间逻辑统一、易于在开题报告中形成完整叙事链条。

> 以上内容可直接作为“第三阶段：基于 Transformer 动态权重的 Learning MPC”章节的提纲，在正式 PPT 中可进一步配合框图、示意曲线与对比表格进行展示。
