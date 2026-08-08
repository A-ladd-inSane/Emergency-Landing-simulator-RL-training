# Emergency-Landing-simulator-RL-training

# JAX 紧急迫降智能决策系统 / JAX Emergency Landing Intelligent Decision System

> 基于飞行汽车紧急迫降技术方案，使用 JAX 实现的三层分离架构（感知→决策→执行）仿真与训练框架。
> 小球（质点）模型作为飞行汽车的简化动力学替身，演示完整的故障检测、在线辨识、强化学习安全决策和着陆控制链路。
>
> A three-layer (Perception → Decision → Execution) simulation and training framework for flight-vehicle emergency landing,
> implemented in JAX. A point-mass ball model serves as a simplified flight-vehicle surrogate, demonstrating the full
> fault detection, online identification, reinforcement learning safety decision, and landing control pipeline.

---

## 目录 / Table of Contents

- [中文文档](#中文文档)
  - [项目概述](#项目概述)
  - [系统架构](#系统架构)
  - [目录结构](#目录结构)
  - [核心模块详解](#核心模块详解)
  - [快速开始](#快速开始)
  - [公式映射表](#公式映射表)
  - [性能基准](#性能基准)
  - [技术栈](#技术栈)
- [English Documentation](#english-documentation)
  - [Project Overview](#project-overview)
  - [System Architecture](#system-architecture)
  - [Directory Structure](#directory-structure)
  - [Core Module Details](#core-module-details)
  - [Quick Start](#quick-start)
  - [Equation Mapping](#equation-mapping)
  - [Performance Benchmarks](#performance-benchmarks)
  - [Tech Stack](#tech-stack)

---

# 中文文档

## 项目概述

本项目将一份**飞行汽车紧急迫降智能决策系统技术方案**落地为可运行的 JAX 代码框架。技术方案定义了三层分离架构（感知层、决策层、执行层），涵盖从故障检测到安全着陆的完整链路。由于完整飞行汽车仿真涉及 6DOF 刚体动力学、多旋翼气动、伞系统耦合等复杂模型，本项目使用**2D 质点（小球）模型**作为动力学替身，在保留架构完整性的前提下实现快速原型验证。

**核心价值：**
- 每个模块对应技术方案中的具体公式编号，可逐条追溯
- 全部物理计算基于 JAX JIT 编译，支持 `vmap` 大规模并行仿真
- 端到端流水线一键运行，验证感知→决策→执行全链路
- 适航安全评估指标（Clopper-Pearson 置信区间、FMEA）已实现

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    紧急迫降决策系统                         │
├─────────────────────────────────────────────────────────┤
│  感知层 (Perception)                                      │
│  ├── FDI: GLR 故障检测 + 残差隔离 (Eq 7-10)              │
│  │     滑动窗口广义似然比检验，χ² 阈值检测突变故障          │
│  │     结构化残差方向隔离：动力/伞/复合故障分类             │
│  ├── RLS: 双遗忘因子在线辨识 B̂(t) (Eq 11-15)            │
│  │     λ_f=0.98 快变通道跟踪阶跃故障                      │
│  │     λ_s=0.999 慢变通道跟踪渐进退化                      │
│  ├── 退化度: SVD截断 Δμ_eff(t) (Eq 4, 14)               │
│  │     三路融合：FDI + RLS + 直接传感器                    │
│  │     SVD 数值秩确定有效控制维度                          │
│  ├── 续航评估: T_remain 估计 (Eq 5)                      │
│  │     基于剩余能量和当前功耗的解析估计                     │
│  └── 降落点检索: KD-Tree + 多目标评分 (Eq 16-17)        │
│        地形坡度、风场适配、可达性、障碍距离加权评分           │
├─────────────────────────────────────────────────────────┤
│  决策层 (Decision)                                       │
│  ├── RL策略: PPO + CMDP Lagrangian (Eq 6, 20)          │
│  │     策略网络：6维观测 → 64×64 MLP → 2维连续动作         │
│  │     价值头 + 安全价值头（约束代价估计）                  │
│  │     拉格朗日乘子 λ 自适应对偶上升                       │
│  ├── 安全过滤: 鲁棒 CBF/CLF QP (Eq 22-23)               │
│  │     对RL动作做投影修正，保证安全集不变                   │
│  │     地面/天花板/障碍物三类约束                          │
│  └── 可行运动包络: FME (Eq 19)                           │
│        控制空间网格前向仿真，计算可达状态包络                │
├─────────────────────────────────────────────────────────┤
│  执行层 (Execution)                                      │
│  ├── 控制分配: 加权伪逆 (Eq 26)                          │
│  │     B†_W = W⁻¹ Bᵀ (B W⁻¹ Bᵀ)⁻¹                     │
│  │     健康度权重反映执行器故障程度                        │
│  ├── 轨迹跟踪: 离散LQR (Eq 27)                           │
│  │     求解离散代数 Riccati 方程获得反馈增益 K              │
│  └── 着陆程序: 三段式下降 (Eq 28)                        │
│        进场 → 拉平 → 接地，分段速度约束                    │
├─────────────────────────────────────────────────────────┤
│  评估层 (Evaluation)                                     │
│  ├── 安全指标: p_safe, p_casualty (§5.6)               │
│  ├── 统计验证: Clopper-Pearson 精确二项分布置信区间       │
│  └── 蒙特卡洛: 150次试验，7种故障场景                     │
└─────────────────────────────────────────────────────────┘
```

## 目录结构

```
emergency_landing_jax/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── config.py                         # 全局参数配置（物理/感知/决策/执行/评估）
├── run_full_pipeline.py              # 端到端流水线入口（7个模块一键验证）
│
├── src/
│   ├── physics/                      # 物理层
│   │   ├── ball_sim_jax.py          #   原始小球仿真（JAX JIT + Matplotlib HUD）
│   │   ├── dynamics.py              #   扩展动力学：故障掩码B(t)、伞扰动d_chute (Eq 1-3)
│   │   └── envelope.py              #   可行运动包络 FME (Eq 19)
│   │
│   ├── perception/                   # 感知层
│   │   ├── fdi.py                   #   GLR故障检测 + 结构化残差隔离 (Eq 7-10)
│   │   ├── rls.py                   #   双遗忘因子RLS在线辨识 (Eq 11-15)
│   │   ├── degradation.py          #   组合退化度 + SVD截断 (Eq 4, 14)
│   │   ├── endurance.py             #   剩余可飞时间估计 (Eq 5)
│   │   └── landing_points.py       #   KD-Tree降落点检索 + 多目标评分 (Eq 16-17)
│   │
│   ├── decision/                     # 决策层
│   │   ├── policy.py                #   策略网络定义（共享特征层 + 策略/价值/安全头）
│   │   ├── cmdp.py                  #   CMDP拉格朗日训练 (Eq 6, 20)
│   │   └── cbf.py                   #   鲁棒CBF/CLF QP安全过滤 (Eq 22-23)
│   │
│   ├── execution/                    # 执行层
│   │   ├── control_alloc.py         #   加权伪逆控制分配 (Eq 26)
│   │   ├── tracker.py               #   离散LQR轨迹跟踪 (Eq 27)
│   │   └── landing.py               #   三段式着陆程序 (Eq 28)
│   │
│   ├── env/                          # 环境层
│   │   ├── fault_injection.py       #   故障注入系统（域随机化 + 课程学习）
│   │   ├── gym_env.py               #   Gymnasium环境封装（SB3兼容）
│   │   └── jax_env.py              #   纯JAX向量化环境（vmap并行）
│   │
│   └── training/                     # 训练层
│       └── train_jax.py             #   纯JAX REINFORCE训练（vmap + autograd）
│
├── evaluation/
│   ├── metrics.py                    #   适航安全指标 + Clopper-Pearson置信区间
│   └── monte_carlo.py               #   蒙特卡洛150次试验评估
│
└── scripts/
    ├── run_sim.py                    #   交互式仿真（Matplotlib HUD）
    ├── run_pipeline.py              #   端到端流水线
    └── run_mc.py                    #   蒙特卡洛安全评估
```

## 核心模块详解

### 1. 物理层 — 扩展动力学 (`src/physics/dynamics.py`)

在原始 `ball_sim_jax.py` 的半隐式 Euler 积分基础上，增加两个关键故障项：

**Eq 1 — 组合动力学方程：**
```
ṡ = f(s) + B(t)·u + d_chute(v, φ)
```
- `f(s)` = 重力 + 线性阻力（标称动力学）
- `B(t)·u` = 经故障掩码后的实际控制力
- `d_chute` = 伞系统产生的加性扰动

**Eq 2 — 故障掩码矩阵：**
```
B(t) = B₀ ⊙ M(φ, δ, t)
```
- `φ ∈ [0,1]`：伞失效模式（0=完好，1=完全失效）
- `δ ∈ [0,1]`：动力降级程度
- `M` 为对角矩阵，各通道独立降级

**Eq 3 — 伞系统加性扰动：**
```
d_chute = ½ ρ ‖v‖² C_D(φ) A(φ) · ℓ_arm
```
- 阻力随伞失效而减小（φ↑ → C_D↓, A↓）
- 部分展开时产生不对称侧向力

**性能：** JIT 编译后单步 ~0.1ms，连续仿真 ~38,000 steps/s

### 2. 感知层

#### 2.1 GLR 故障检测 (`src/perception/fdi.py`)

**原理：** 滑动窗口内计算广义似然比统计量，超过卡方阈值则判定故障。

```
H₀: 残差 ~ N(0, Σ)       （正常）
H₁: 残差 ~ N(μ_fault, Σ)  （故障，均值偏移）

GLR = max_μ [Σ rᵢᵀ Σ⁻¹ μ - n/2 μᵀ Σ⁻¹ μ]  ~ χ²(d) under H₀
```

- 虚警概率 α = 10⁻⁶，卡方自由度 d = 2
- 阈值 ≈ 27.63
- 检测到故障后，根据残差方向向量隔离故障类型：
  - 垂直方向残差大 → 动力降级
  - 水平方向残差大 → 伞失效
  - 两方向均大 → 复合故障

#### 2.2 双遗忘因子 RLS (`src/perception/rls.py`)

同时运行两个 RLS 通道：

| 通道 | 遗忘因子 | 用途 |
|------|---------|------|
| 快变 λ_f = 0.98 | 灵敏但噪声大 | 跟踪阶跃故障（如动力突变） |
| 慢变 λ_s = 0.999 | 稳定但滞后 | 跟踪渐进退化（如电池老化） |

融合策略：基于预测误差自适应调整权重 w_f，误差大时偏向快变通道。

输出：控制有效性矩阵的在线估计 B̂(t)，供决策层使用。

#### 2.3 组合退化度 (`src/perception/degradation.py`)

三路信息融合给出标量退化度量 Δμ_eff(t) ∈ [0,1]：

```
Δμ_eff = α·μ_FDI + β·μ_RLS + γ·μ_sensor
```

- `μ_FDI`：FDI 检测到的故障严重度
- `μ_RLS`：B̂(t) vs B₀ 的相对偏差
- `μ_sensor`：直接传感器读数（能量、推力）

SVD 截断（Eq 14）：对 B̂ 做奇异值分解，截断低于噪声等效奇异值 δ_sensor 的项，确定有效控制维度（数值秩）。

#### 2.4 降落点检索 (`src/perception/landing_points.py`)

基于 KD-Tree 的空间检索 + 多目标评分：

```
score = w₁·(1 - d/d₀) + w₂·(1 - ρ/ρ₀) + w₃·(1 - σ/σ₀) + w₄·reachability
```

- 距离 d：距当前位置的欧氏距离
- 人口密度 ρ：地面人员密度估计
- 粗糙度 σ：地形坡度
- 可达性：是否在 FME 包络内

输出 Top-K 候选降落点，按综合得分排序。

### 3. 决策层

#### 3.1 策略网络 (`src/decision/policy.py`)

纯 JAX 实现的 MLP 策略网络：

```
观测(6D) → [64] → [64] → 策略头(2D) + 价值头(1D) + 安全价值头(1D)
```

- 共享特征层：两层 tanh 激活
- 策略头：输出动作均值（tanh压缩到[-1,1]）+ 可学习 log_std
- 价值头：状态价值 V(s) 估计
- 安全价值头：安全约束代价 C(s) 估计（CMDP专用）

全部参数手工管理（不依赖 Flax），使用 `jax.vmap` 支持批量推理。

#### 3.2 CMDP 拉格朗日训练 (`src/decision/cmdp.py`)

约束马尔可夫决策过程的拉格朗日方法：

```
max_θ  E[Σ γᵗ rₜ]
s.t.   E[Σ γᵗ cₜ] ≤ d_safety

L(θ, λ) = J_reward(θ) - λ·(J_cost(θ) - d_safety)
θ ← θ + ∇_θ L     （策略梯度上升）
λ ← λ + αλ·(J_cost - d_safety)   （对偶上升）
```

- λ 初始为 0，当安全代价超限时自动增大
- Adam 优化器，学习率 3e-4
- GAE λ = 0.95 的优势函数估计

#### 3.3 CBF 安全过滤 (`src/decision/cbf.py`)

对 RL 策略输出的原始动作做投影修正：

```
min ||u - u_rl||²
s.t. Lf h(x) + Lg h(x)·u ≥ -α·h(x) - margin   （CBF约束）
     u ∈ U                                       （执行器约束）
```

- `h(x) ≥ 0` 定义安全集（距障碍物/地面的有符号距离）
- `margin` 为鲁棒裕度，对应模型不确定度 Σ_cb
- 使用 SLSQP 求解器（scipy.optimize.minimize）
- 预置三类约束：地面碰撞、天花板、障碍物

### 4. 执行层

#### 4.1 控制分配 (`src/execution/control_alloc.py`)

加权伪逆方法，考虑故障掩码：

```
B†_W = W⁻¹ Bᵀ (B W⁻¹ Bᵀ)⁻¹
u_command = B†_W · F_desired
```

- 权重矩阵 W 为各执行器健康度（故障后降低）
- 条件数 κ(B) > 100 时触发降级警告
- 输出饱和度（各执行器是否达到物理极限）

#### 4.2 LQR 轨迹跟踪 (`src/execution/tracker.py`)

离散代数 Riccati 方程求解 LQR 增益：

```
P = AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q     （DARE）
K = (R + BᵀPB)⁻¹BᵀPA
u = -K(x - x_ref) + u_ff
```

- 使用 `scipy.linalg.solve_discrete_are` 求解
- Q = diag(10, 10, 1, 1)：位置权重大于速度
- R = diag(0.1, 0.1)：控制能量惩罚

#### 4.3 着陆程序 (`src/execution/landing.py`)

三段式下降策略：

| 阶段 | 触发条件 | 下降率 | 控制目标 |
|------|---------|--------|---------|
| 进场 (Approach) | y > flare_height | 2.0 m/s | 朝目标点水平移动 |
| 拉平 (Flare) | y ≤ flare_height | 0.5 m/s | 减速，浅角度 |
| 接地 (Touchdown) | y ≤ 1m | →0 m/s | 垂直速度趋零 |

### 5. 评估层

#### 5.1 适航安全指标 (`src/evaluation/metrics.py`)

**Clopper-Pearson 精确二项分布置信区间：**
```
p_lower = Beta(α/2; k, n-k+1)
p_upper = Beta(1-α/2; k+1, n-k)
```
- n = 150 次蒙特卡洛试验
- k 次成功
- 95% 置信水平

**适航标准：**
- p_safe ≥ 0.95（安全着陆概率）
- p_casualty ≤ 10⁻⁴（地面伤亡概率）

**FMEA（故障模式与影响分析）：**
- 严重度(S) × 发生度(O) × 检测度(D) = 风险优先数(RPN)
- RPN > 125 为高风险项

#### 5.2 蒙特卡洛评估 (`src/evaluation/monte_carlo.py`)

150 次试验覆盖 7 种故障场景：
- 正常条件
- 动力降级 30%/50%/70%
- 伞完全失效 / 伞部分失效
- 复合故障（伞50% + 动力50% + 不对称30%）

每次试验：随机初始状态 → 故障注入 → 物理仿真 → 判定着陆成功/失败/伤亡。

## 快速开始

### 安装依赖

```bash
cd emergency_landing_jax
pip install -r requirements.txt
```

### 一键运行端到端流水线

```bash
python run_full_pipeline.py
```

输出示例：
```
============================================================
  1. 物理引擎验证
============================================================
  10k steps: 38,627 steps/s

============================================================
  2. 感知层
============================================================
  FDI: GLR阈值=27.6310
  退化度: μ_eff=0.4473 (rank=2)
  续航: T_remain=48.2s, range=168.7m

============================================================
  3. 可行运动包络 (FME)
============================================================
  正常: x=[-2.9, 5.3], y=[35.2, 43.5]
  故障: x=[0.0, 3.9], y=[37.1, 40.6]

============================================================
  4. 降落点检索
============================================================
  #1: (+80.0, 60.0) score=0.892

============================================================
  5. CBF 安全过滤
============================================================

============================================================
  6. 控制分配
============================================================

============================================================
  7. 蒙特卡洛安全评估
============================================================
  总计: 0/30
  p_safe: 0.0000  95% CI: [0.0000, 0.1157]
```

### 交互式仿真（带 HUD）

```bash
python scripts/run_sim.py
```

控制方式：
- `↑/↓` 推力增减
- `←/→` 旋转引擎角度
- `W/A/S/D` 快速设置引擎方向
- `SPACE` 暂停/恢复
- `R` 重置

### 纯 JAX RL 训练

```bash
python src/training/train_jax.py --n_envs 500 --steps 200 --iters 200
```

无需 PyTorch，纯 JAX 实现的 REINFORCE 策略梯度，500 并行环境 × 200 步 × 200 轮 = 20M env-steps。

### 蒙特卡洛安全评估

```bash
python scripts/run_mc.py
```

7 种故障场景 × 21 次试验 = 147 次，输出 Clopper-Pearson 置信区间。

### 单模块独立运行

每个模块均可独立运行验证：

```bash
cd emergency_landing_jax

# 物理引擎测试
python src/physics/dynamics.py

# FDI 故障检测测试
python src/perception/fdi.py

# RLS 在线辨识测试
python src/perception/rls.py

# CBF 安全过滤测试
python src/decision/cbf.py

# 控制分配测试
python src/execution/control_alloc.py

# LQR 跟踪测试
python src/execution/tracker.py

# 着陆程序测试
python src/execution/landing.py
```

## 公式映射表

| 技术方案公式 | 代码位置 | 说明 |
|:---:|:---|:---|
| Eq 1 | `src/physics/dynamics.py` `faulty_physics_step()` | 组合动力学 ṡ=f(s)+B(t)u+d_chute |
| Eq 2 | `src/physics/dynamics.py` `control_effectiveness_mask()` | 故障掩码 B(t)=B₀⊙M(φ,δ,t) |
| Eq 3 | `src/physics/dynamics.py` `parachute_disturbance()` | 伞扰动 d_chute=½ρ‖v‖²C_D(φ)A(φ) |
| Eq 4 | `src/perception/degradation.py` `compute_degradation()` | 组合退化度 Δμ_eff(t) |
| Eq 5 | `src/perception/endurance.py` `estimate_endurance()` | 剩余可飞时间 T_remain |
| Eq 6 | `src/decision/cmdp.py` `cmdp_update()` | CMDP 优化目标 |
| Eq 7-8 | `src/perception/fdi.py` `GLRDetector.update()` | GLR 检验统计量 + χ²阈值 |
| Eq 9 | `src/perception/fdi.py` `GLRDetector._isolate()` | 结构化残差隔离 |
| Eq 10 | `src/perception/fdi.py` `FDIResult` | 严重度+置信度估计 |
| Eq 11-13 | `src/perception/rls.py` `DualForgettingRLS.update()` | 双遗忘因子 RLS 更新 |
| Eq 14 | `src/perception/degradation.py` `svd_truncation()` | SVD 截断 + 数值秩 |
| Eq 15 | `src/perception/rls.py` `get_B_hat()` | B̂(t) 输出 |
| Eq 16-17 | `src/perception/landing_points.py` `search()` | KD-Tree 检索 + 多目标评分 |
| Eq 19 | `src/physics/envelope.py` `compute_fme()` | 可行运动包络 FME |
| Eq 20 | `src/decision/cmdp.py` `cmdp_update()` | CMDP 拉格朗日对偶上升 |
| Eq 22-23 | `src/decision/cbf.py` `CBFSafetyFilter.filter()` | 鲁棒 CBF QP 安全过滤 |
| Eq 26 | `src/execution/control_alloc.py` `allocate_control()` | 加权伪逆控制分配 |
| Eq 27 | `src/execution/tracker.py` `compute_lqr_gain()` | 离散 LQR (DARE) |
| Eq 28 | `src/execution/landing.py` `LandingController` | 三段式着陆程序 |
| §5.6.1 | `src/evaluation/metrics.py` `compute_safety_metrics()` | p_safe, p_casualty |
| §5.6.2 | `src/evaluation/metrics.py` `clopper_pearson_ci()` | Clopper-Pearson 置信区间 |
| §5.6.5 | `src/evaluation/monte_carlo.py` `run_monte_carlo()` | 蒙特卡洛评估 |

## 性能基准

| 指标 | 数值 | 说明 |
|------|------|------|
| 物理引擎单步 | ~0.026ms | JAX JIT，含故障掩码+伞扰动 |
| 连续仿真吞吐 | 38,600 steps/s | 单 CPU 核 |
| JAX vmap 并行 | 425,900 env-steps/s | 1,000 并行环境 |
| 端到端 RL训练 | 17,274 env-steps/s | 200 envs × 100 steps × 30 iters |
| 流水线全流程 | ~3 秒 | 7个模块依次执行 |
| 代码规模 | 4,712 行 / 27 个 .py 文件 | 含全部模块和测试 |

## 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| JAX | 物理引擎、策略网络、自动微分 | 0.4.30 |
| NumPy | 数值计算、FDI/RLS | ≥1.22 |
| SciPy | 卡方分布、KD-Tree、LQR(DARE)、QP求解 | ≥1.9 |
| Matplotlib | 交互式仿真可视化 | ≥3.5 |
| Gymnasium | RL 环境标准接口 | ≥1.0 |

**可选依赖：**
- Stable-Baselines3：MuJoCo-like PPO 训练工作流
- Brax：纯 JAX PPO/SAC 实现
- Flax：更高级的神经网络框架（本项目使用手工 MLP 避免额外依赖）

---

# English Documentation

## Project Overview

This project implements a **flight-vehicle emergency landing intelligent decision system** as a runnable JAX codebase. The technical proposal defines a three-layer architecture (Perception → Decision → Execution) covering the full pipeline from fault detection to safe landing. Since a full flight-vehicle simulation involves 6DOF rigid body dynamics, multi-rotor aerodynamics, and parachute system coupling, this project uses a **2D point-mass (ball) model** as a dynamics surrogate, preserving architectural completeness while enabling rapid prototyping.

**Key Value:**
- Every module maps to a specific equation number in the technical proposal, enabling traceability
- All physics computations use JAX JIT compilation, supporting `vmap` for massive parallel simulation
- End-to-end pipeline runs with a single command, validating the full Perception → Decision → Execution chain
- Airworthiness safety metrics (Clopper-Pearson confidence intervals, FMEA) are implemented

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Emergency Landing Decision System            │
├─────────────────────────────────────────────────────────┤
│  Perception Layer                                        │
│  ├── FDI: GLR fault detection + residual isolation (Eq 7-10)│
│  │     Sliding-window generalized likelihood ratio test  │
│  │     χ² threshold for abrupt fault detection            │
│  │     Structured residual isolation: power/chute/combined│
│  ├── RLS: Dual-forgetting-factor online ID B̂(t) (Eq 11-15)│
│  │     λ_f=0.98 fast channel for step faults              │
│  │     λ_s=0.999 slow channel for gradual degradation     │
│  ├── Degradation: SVD truncation Δμ_eff(t) (Eq 4, 14)   │
│  │     Three-source fusion: FDI + RLS + direct sensors   │
│  │     SVD numerical rank for effective control dimension │
│  ├── Endurance: T_remain estimation (Eq 5)               │
│  │     Analytical estimate from remaining energy & power  │
│  └── Landing Points: KD-Tree + multi-objective scoring   │
│        (Eq 16-17)                                        │
│        Terrain slope, wind alignment, reachability,       │
│        obstacle distance weighted scoring                 │
├─────────────────────────────────────────────────────────┤
│  Decision Layer                                          │
│  ├── RL Policy: PPO + CMDP Lagrangian (Eq 6, 20)       │
│  │     Policy network: 6D obs → 64×64 MLP → 2D action    │
│  │     Value head + safety value head (constraint cost)  │
│  │     Lagrange multiplier λ adaptive dual ascent        │
│  ├── Safety Filter: Robust CBF/CLF QP (Eq 22-23)       │
│  │     Projects RL actions to safe set                    │
│  │     Ground/ceiling/obstacle constraints              │
│  └── Feasible Motion Envelope: FME (Eq 19)              │
│        Control-space grid forward simulation              │
├─────────────────────────────────────────────────────────┤
│  Execution Layer                                         │
│  ├── Control Allocation: Weighted pseudo-inverse (Eq 26)│
│  │     B†_W = W⁻¹ Bᵀ (B W⁻¹ Bᵀ)⁻¹                      │
│  │     Health-weighted (faulty actuators get low weight)  │
│  ├── Trajectory Tracking: Discrete LQR (Eq 27)          │
│  │     Solves Discrete Algebraic Riccati Equation (DARE)│
│  └── Landing Procedure: Three-phase descent (Eq 28)    │
│        Approach → Flare → Touchdown                      │
├─────────────────────────────────────────────────────────┤
│  Evaluation Layer                                        │
│  ├── Safety Metrics: p_safe, p_casualty (§5.6)         │
│  ├── Statistical Validation: Clopper-Pearson CI         │
│  └── Monte Carlo: 150 trials, 7 fault scenarios          │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
emergency_landing_jax/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── config.py                         # Global parameters (physics/perception/decision/execution/eval)
├── run_full_pipeline.py              # End-to-end pipeline entry point
│
├── src/
│   ├── physics/                      # Physics Layer
│   │   ├── ball_sim_jax.py          #   Original ball simulation (JAX JIT + Matplotlib HUD)
│   │   ├── dynamics.py              #   Extended dynamics: fault mask B(t), chute disturbance (Eq 1-3)
│   │   └── envelope.py              #   Feasible Motion Envelope FME (Eq 19)
│   │
│   ├── perception/                   # Perception Layer
│   │   ├── fdi.py                   #   GLR fault detection + structured residual isolation (Eq 7-10)
│   │   ├── rls.py                   #   Dual-forgetting-factor RLS online identification (Eq 11-15)
│   │   ├── degradation.py          #   Combined degradation + SVD truncation (Eq 4, 14)
│   │   ├── endurance.py             #   Remaining flight time estimation (Eq 5)
│   │   └── landing_points.py       #   KD-Tree landing point search + multi-objective scoring (Eq 16-17)
│   │
│   ├── decision/                     # Decision Layer
│   │   ├── policy.py                #   Policy network (shared features + policy/value/safety heads)
│   │   ├── cmdp.py                  #   CMDP Lagrangian training (Eq 6, 20)
│   │   └── cbf.py                   #   Robust CBF/CLF QP safety filter (Eq 22-23)
│   │
│   ├── execution/                    # Execution Layer
│   │   ├── control_alloc.py         #   Weighted pseudo-inverse control allocation (Eq 26)
│   │   ├── tracker.py               #   Discrete LQR trajectory tracking (Eq 27)
│   │   └── landing.py               #   Three-phase landing procedure (Eq 28)
│   │
│   ├── env/                          # Environment Layer
│   │   ├── fault_injection.py       #   Fault injection (domain randomization + curriculum)
│   │   ├── gym_env.py               #   Gymnasium wrapper (SB3-compatible)
│   │   └── jax_env.py              #   Pure-JAX vectorized environment (vmap parallel)
│   │
│   └── training/                     # Training
│       └── train_jax.py             #   Pure-JAX REINFORCE training (vmap + autograd)
│
├── evaluation/
│   ├── metrics.py                    #   Airworthiness safety metrics + Clopper-Pearson CI
│   └── monte_carlo.py               #   Monte Carlo 150-trial evaluation
│
└── scripts/
    ├── run_sim.py                    #   Interactive simulation (Matplotlib HUD)
    ├── run_pipeline.py              #   End-to-end pipeline
    └── run_mc.py                    #   Monte Carlo safety evaluation
```

## Core Module Details

### 1. Physics Layer — Extended Dynamics (`src/physics/dynamics.py`)

Extends the original `ball_sim_jax.py` semi-implicit Euler integration with two critical fault terms:

**Eq 1 — Combined dynamics equation:**
```
ṡ = f(s) + B(t)·u + d_chute(v, φ)
```
- `f(s)` = gravity + linear drag (nominal dynamics)
- `B(t)·u` = actual control force after fault masking
- `d_chute` = additive disturbance from parachute system

**Eq 2 — Fault mask matrix:**
```
B(t) = B₀ ⊙ M(φ, δ, t)
```
- `φ ∈ [0,1]`: parachute failure mode (0=intact, 1=total failure)
- `δ ∈ [0,1]`: power degradation level
- `M` is diagonal, allowing per-channel degradation

**Eq 3 — Parachute additive disturbance:**
```
d_chute = ½ ρ ‖v‖² C_D(φ) A(φ) · ℓ_arm
```
- Drag decreases as parachute fails (φ↑ → C_D↓, A↓)
- Partial deployment generates asymmetric lateral force

**Performance:** ~0.026ms/step after JIT compilation, ~38,000 steps/s continuous

### 2. Perception Layer

#### 2.1 GLR Fault Detection (`src/perception/fdi.py`)

**Principle:** Compute generalized likelihood ratio statistic over a sliding window; if it exceeds a chi-squared threshold, declare a fault.

```
H₀: residual ~ N(0, Σ)       (nominal)
H₁: residual ~ N(μ_fault, Σ)  (faulted, mean shift)

GLR = max_μ [Σ rᵢᵀ Σ⁻¹ μ - n/2 μᵀ Σ⁻¹ μ]  ~ χ²(d) under H₀
```

- False alarm probability α = 10⁻⁶, chi-squared DOF d = 2
- Threshold ≈ 27.63
- After detection, fault type isolated via residual direction:
  - Large vertical residual → power degradation
  - Large horizontal residual → parachute failure
  - Both significant → combined fault

#### 2.2 Dual-Forgetting-Factor RLS (`src/perception/rls.py`)

Runs two RLS channels simultaneously:

| Channel | Forgetting Factor | Purpose |
|---------|-------------------|---------|
| Fast λ_f = 0.98 | Sensitive but noisy | Track step faults (sudden power loss) |
| Slow λ_s = 0.999 | Stable but lagging | Track gradual degradation (battery aging) |

Fusion: adaptively adjusts weight w_f based on prediction error; biases toward fast channel when error is large.

Output: online estimate of control effectiveness matrix B̂(t) for the decision layer.

#### 2.3 Combined Degradation (`src/perception/degradation.py`)

Three-source fusion into a scalar degradation measure Δμ_eff(t) ∈ [0,1]:

```
Δμ_eff = α·μ_FDI + β·μ_RLS + γ·μ_sensor
```

- `μ_FDI`: fault severity from FDI
- `μ_RLS`: relative deviation of B̂(t) from B₀
- `μ_sensor`: direct sensor readings (energy, thrust)

SVD truncation (Eq 14): decompose B̂, truncate singular values below noise-equivalent threshold δ_sensor, determine effective control dimension (numerical rank).

#### 2.4 Landing Point Search (`src/perception/landing_points.py`)

KD-Tree spatial search + multi-objective scoring:

```
score = w₁·(1 - d/d₀) + w₂·(1 - ρ/ρ₀) + w₃·(1 - σ/σ₀) + w₄·reachability
```

- Distance d: Euclidean distance from current position
- Population density ρ: ground personnel density estimate
- Roughness σ: terrain slope
- Reachability: whether within FME envelope

Returns Top-K candidate landing points, sorted by composite score.

### 3. Decision Layer

#### 3.1 Policy Network (`src/decision/policy.py`)

Pure JAX MLP policy network:

```
Obs(6D) → [64] → [64] → Policy head(2D) + Value head(1D) + Safety value head(1D)
```

- Shared feature layers: two tanh-activated layers
- Policy head: action mean (tanh squashed to [-1,1]) + learnable log_std
- Value head: state value V(s) estimate
- Safety value head: safety constraint cost C(s) estimate (CMDP-specific)

All parameters managed manually (no Flax dependency), supports `jax.vmap` for batch inference.

#### 3.2 CMDP Lagrangian Training (`src/decision/cmdp.py`)

Constrained Markov Decision Process via Lagrangian method:

```
max_θ  E[Σ γᵗ rₜ]
s.t.   E[Σ γᵗ cₜ] ≤ d_safety

L(θ, λ) = J_reward(θ) - λ·(J_cost(θ) - d_safety)
θ ← θ + ∇_θ L     (policy gradient ascent)
λ ← λ + αλ·(J_cost - d_safety)   (dual ascent)
```

- λ starts at 0, automatically increases when safety cost exceeds limit
- Adam optimizer, learning rate 3e-4
- GAE λ = 0.95 advantage estimation

#### 3.3 CBF Safety Filter (`src/decision/cbf.py`)

Projects raw RL actions onto the safe set:

```
min ||u - u_rl||²
s.t. Lf h(x) + Lg h(x)·u ≥ -α·h(x) - margin   (CBF constraint)
     u ∈ U                                       (actuator bounds)
```

- `h(x) ≥ 0` defines the safe set (signed distance to obstacles/ground)
- `margin` is the robustness margin for model uncertainty Σ_cb
- SLSQP solver (scipy.optimize.minimize)
- Three pre-built constraint types: ground collision, ceiling, obstacles

### 4. Execution Layer

#### 4.1 Control Allocation (`src/execution/control_alloc.py`)

Weighted pseudo-inverse with fault awareness:

```
B†_W = W⁻¹ Bᵀ (B W⁻¹ Bᵀ)⁻¹
u_command = B†_W · F_desired
```

- Weight matrix W encodes per-actuator health (reduced after faults)
- Condition number κ(B) > 100 triggers degradation warning
- Outputs saturation indicators (whether actuators hit physical limits)

#### 4.2 LQR Trajectory Tracking (`src/execution/tracker.py`)

Discrete Algebraic Riccati Equation for LQR gain:

```
P = AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q     (DARE)
K = (R + BᵀPB)⁻¹BᵀPA
u = -K(x - x_ref) + u_ff
```

- Uses `scipy.linalg.solve_discrete_are`
- Q = diag(10, 10, 1, 1): position weighted higher than velocity
- R = diag(0.1, 0.1): control energy penalty

#### 4.3 Landing Procedure (`src/execution/landing.py`)

Three-phase descent strategy:

| Phase | Trigger | Descent Rate | Control Target |
|-------|---------|-------------|----------------|
| Approach | y > flare_height | 2.0 m/s | Move horizontally toward target |
| Flare | y ≤ flare_height | 0.5 m/s | Decelerate, shallow angle |
| Touchdown | y ≤ 1m | →0 m/s | Zero vertical velocity |

### 5. Evaluation Layer

#### 5.1 Airworthiness Safety Metrics (`src/evaluation/metrics.py`)

**Clopper-Pearson exact binomial confidence interval:**
```
p_lower = Beta(α/2; k, n-k+1)
p_upper = Beta(1-α/2; k+1, n-k)
```
- n = 150 Monte Carlo trials
- k successes
- 95% confidence level

**Airworthiness standards:**
- p_safe ≥ 0.95 (safe landing probability)
- p_casualty ≤ 10⁻⁴ (ground casualty probability)

**FMEA (Failure Mode and Effects Analysis):**
- Severity(S) × Occurrence(O) × Detection(D) = Risk Priority Number (RPN)
- RPN > 125 classified as high-risk

#### 5.2 Monte Carlo Evaluation (`src/evaluation/monte_carlo.py`)

150 trials covering 7 fault scenarios:
- Normal conditions
- Power degradation 30%/50%/70%
- Parachute total failure / partial failure
- Combined faults (chute 50% + power 50% + asymmetry 30%)

Each trial: random initial state → fault injection → physics simulation → landing success/failure/casualty determination.

## Quick Start

### Install Dependencies

```bash
cd emergency_landing_jax
pip install -r requirements.txt
```

### Run End-to-End Pipeline

```bash
python run_full_pipeline.py
```

Sample output:
```
============================================================
  1. Physics Engine Verification
============================================================
  10k steps: 38,627 steps/s

============================================================
  2. Perception Layer
============================================================
  FDI: GLR threshold = 27.6310
  Degradation: μ_eff = 0.4473 (rank=2)
  Endurance: T_remain = 48.2s, range = 168.7m

============================================================
  3. Feasible Motion Envelope (FME)
============================================================
  Normal: x=[-2.9, 5.3], y=[35.2, 43.5]
  Fault:  x=[0.0, 3.9], y=[37.1, 40.6]

============================================================
  4. Landing Point Search
============================================================
  #1: (+80.0, 60.0) score=0.892

============================================================
  5. CBF Safety Filter
============================================================

============================================================
  6. Control Allocation
============================================================

============================================================
  7. Monte Carlo Safety Evaluation
============================================================
  Total: 0/30
  p_safe: 0.0000  95% CI: [0.0000, 0.1157]
```

### Interactive Simulation (with HUD)

```bash
python scripts/run_sim.py
```

Controls:
- `↑/↓` Thrust increase/decrease
- `←/→` Rotate engine angle
- `W/A/S/D` Quick set engine direction
- `SPACE` Pause/Resume
- `R` Reset

### Pure-JAX RL Training

```bash
python src/training/train_jax.py --n_envs 500 --steps 200 --iters 200
```

No PyTorch required. Pure JAX REINFORCE policy gradient, 500 parallel envs × 200 steps × 200 iterations = 20M env-steps.

### Monte Carlo Safety Evaluation

```bash
python scripts/run_mc.py
```

7 fault scenarios × 21 trials = 147 total, outputs Clopper-Pearson confidence interval.

### Run Individual Modules

Each module can run standalone for verification:

```bash
cd emergency_landing_jax

# Physics engine test
python src/physics/dynamics.py

# FDI fault detection test
python src/perception/fdi.py

# RLS online identification test
python src/perception/rls.py

# CBF safety filter test
python src/decision/cbf.py

# Control allocation test
python src/execution/control_alloc.py

# LQR tracking test
python src/execution/tracker.py

# Landing procedure test
python src/execution/landing.py
```

## Equation Mapping

| Tech Proposal Equation | Code Location | Description |
|:---:|:---|:---|
| Eq 1 | `src/physics/dynamics.py` `faulty_physics_step()` | Combined dynamics ṡ=f(s)+B(t)u+d_chute |
| Eq 2 | `src/physics/dynamics.py` `control_effectiveness_mask()` | Fault mask B(t)=B₀⊙M(φ,δ,t) |
| Eq 3 | `src/physics/dynamics.py` `parachute_disturbance()` | Chute disturbance d_chute=½ρ‖v‖²C_D(φ)A(φ) |
| Eq 4 | `src/perception/degradation.py` `compute_degradation()` | Combined degradation Δμ_eff(t) |
| Eq 5 | `src/perception/endurance.py` `estimate_endurance()` | Remaining flight time T_remain |
| Eq 6 | `src/decision/cmdp.py` `cmdp_update()` | CMDP optimization objective |
| Eq 7-8 | `src/perception/fdi.py` `GLRDetector.update()` | GLR test statistic + χ² threshold |
| Eq 9 | `src/perception/fdi.py` `GLRDetector._isolate()` | Structured residual isolation |
| Eq 10 | `src/perception/fdi.py` `FDIResult` | Severity + confidence estimation |
| Eq 11-13 | `src/perception/rls.py` `DualForgettingRLS.update()` | Dual-forgetting RLS update |
| Eq 14 | `src/perception/degradation.py` `svd_truncation()` | SVD truncation + numerical rank |
| Eq 15 | `src/perception/rls.py` `get_B_hat()` | B̂(t) output |
| Eq 16-17 | `src/perception/landing_points.py` `search()` | KD-Tree search + multi-objective scoring |
| Eq 19 | `src/physics/envelope.py` `compute_fme()` | Feasible Motion Envelope |
| Eq 20 | `src/decision/cmdp.py` `cmdp_update()` | CMDP Lagrangian dual ascent |
| Eq 22-23 | `src/decision/cbf.py` `CBFSafetyFilter.filter()` | Robust CBF QP safety filter |
| Eq 26 | `src/execution/control_alloc.py` `allocate_control()` | Weighted pseudo-inverse allocation |
| Eq 27 | `src/execution/tracker.py` `compute_lqr_gain()` | Discrete LQR (DARE) |
| Eq 28 | `src/execution/landing.py` `LandingController` | Three-phase landing procedure |
| §5.6.1 | `src/evaluation/metrics.py` `compute_safety_metrics()` | p_safe, p_casualty |
| §5.6.2 | `src/evaluation/metrics.py` `clopper_pearson_ci()` | Clopper-Pearson confidence interval |
| §5.6.5 | `src/evaluation/monte_carlo.py` `run_monte_carlo()` | Monte Carlo evaluation |

## Performance Benchmarks

| Metric | Value | Description |
|--------|-------|-------------|
| Physics step latency | ~0.026ms | JAX JIT, including fault mask + chute disturbance |
| Continuous simulation | 38,600 steps/s | Single CPU core |
| JAX vmap parallel | 425,900 env-steps/s | 1,000 parallel environments |
| End-to-end RL training | 17,274 env-steps/s | 200 envs × 100 steps × 30 iters |
| Full pipeline runtime | ~3 seconds | All 7 modules sequentially |
| Code size | 4,712 lines / 27 .py files | Including all modules and tests |

## Tech Stack

| Component | Purpose | Version |
|-----------|---------|---------|
| JAX | Physics engine, policy network, autodiff | 0.4.30 |
| NumPy | Numerical computation, FDI/RLS | ≥1.22 |
| SciPy | Chi-squared distribution, KD-Tree, LQR (DARE), QP solver | ≥1.9 |
| Matplotlib | Interactive simulation visualization | ≥3.5 |
| Gymnasium | RL environment standard interface | ≥1.0 |

**Optional dependencies:**
- Stable-Baselines3: MuJoCo-like PPO training workflow
- Brax: Pure-JAX PPO/SAC implementation
- Flax: Higher-level neural network framework (this project uses manual MLP to avoid extra dependencies)
