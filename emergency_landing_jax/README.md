# JAX 紧急迫降智能决策系统

> 基于飞行汽车紧急迫降技术方案，使用 JAX 实现的三层分离架构仿真与训练框架。
> 小球（质点）模型作为飞行汽车的简化动力学替身，演示完整的感知→决策→执行链路。

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    紧急迫降决策系统                         │
├─────────────────────────────────────────────────────────┤
│  感知层 (Perception)                                      │
│  ├── FDI: GLR 故障检测 + 残差隔离 (Eq 7-10)              │
│  ├── RLS: 双遗忘因子在线辨识 B̂(t) (Eq 11-15)            │
│  ├── 退化度: SVD截断 Δμ_eff(t) (Eq 4, 14)               │
│  ├── 续航评估: T_remain 估计 (Eq 5)                      │
│  └── 降落点检索: KD-Tree + 评分 (Eq 16-17)              │
├─────────────────────────────────────────────────────────┤
│  决策层 (Decision)                                       │
│  ├── RL策略: PPO + CMDP Lagrangian (Eq 6, 20)           │
│  ├── 安全过滤: 鲁棒 CBF/CLF QP (Eq 22-23)               │
│  ├── 路径规划: A* + RRT* 混合策略 (Eq 24)               │
│  └── 动态重规划: 在线 MPC (Eq 25)                        │
├─────────────────────────────────────────────────────────┤
│  执行层 (Execution)                                      │
│  ├── 控制分配: 伪逆 B̂⁺ (Eq 26)                          │
│  ├── 轨迹跟踪: LQR (Eq 27)                              │
│  └── 着陆程序: 分段下降 (Eq 28)                          │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
emergency_landing_jax/
├── README.md                        # 本文件
├── requirements.txt                 # 依赖
├── config.py                        # 全局参数配置（对应技术方案各公式参数）
├── src/
│   ├── physics/
│   │   ├── dynamics.py              # 扩展动力学：B(t)掩码 + 伞扰动 (Eq 1-3)
│   │   ├── ball_sim_jax.py          # 原始JAX物理引擎
│   │   └── envelope.py              # FME 可行运动包络 (Eq 19)
│   ├── perception/
│   │   ├── fdi.py                   # GLR故障检测 + 残差隔离 (Eq 7-10)
│   │   ├── rls.py                   # 双遗忘因子RLS (Eq 11-15)
│   │   ├── degradation.py           # 退化度Δμ_eff + SVD截断 (Eq 4, 14)
│   │   ├── endurance.py             # 续航评估 T_remain (Eq 5)
│   │   ├── landing_points.py        # KD-Tree降落点检索 (Eq 16-17)
│   │   └── structured_log.py        # 结构化日志 (Eq 18)
│   ├── decision/
│   │   ├── policy.py                # 策略网络 (JAX/Flax MLP)
│   │   ├── cmdp.py                  # CMDP Lagrangian 训练 (Eq 6, 20)
│   │   ├── cbf.py                   # 鲁棒CBF/CLF QP安全过滤 (Eq 22-23)
│   │   ├── path_planning.py         # A*+RRT* 路径规划 (Eq 24)
│   │   └── mpc.py                   # 在线MPC重规划 (Eq 25)
│   ├── execution/
│   │   ├── control_alloc.py         # 伪逆控制分配 (Eq 26)
│   │   ├── tracker.py               # LQR轨迹跟踪 (Eq 27)
│   │   └── landing.py               # 着陆程序 (Eq 28)
│   ├── env/
│   │   ├── fault_injection.py       # 故障模式：伞失效φ + 动力降级δ
│   │   ├── gym_env.py               # Gymnasium 环境（感知层+决策层闭环）
│   │   └── jax_env.py               # 纯JAX向量化环境（vmappable）
│   └── training/
│       ├── train_ppo.py             # SB3 PPO + CMDP训练
│       ├── train_jax.py             # 纯JAX REINFORCE/CMDP训练
│       └── curriculum.py            # 课程学习调度
├── evaluation/
│   ├── metrics.py                   # p_safe, p_casualty, Clopper-Pearson (Eq 9)
│   └── monte_carlo.py              # 150次蒙特卡洛评估
├── scripts/
│   ├── run_sim.py                   # 交互式仿真
│   ├── train.py                     # 训练入口
│   ├── evaluate.py                  # 评估入口
│   └── demo_fault.py               # 故障注入演示
└── tests/
    ├── test_physics.py
    ├── test_perception.py
    └── test_decision.py
```

## 快速开始

```bash
# 安装依赖
pip install jax jaxlib gymnasium numpy scipy matplotlib

# 运行原始仿真（无故障）
python scripts/run_sim.py

# 故障注入演示
python scripts/demo_fault.py

# 训练 RL 策略（纯JAX）
python scripts/train.py --algo jax --env ball_hover

# 评估
python scripts/evaluate.py --model models/best --trials 150
```

## 公式映射

| 技术方案公式 | 代码位置 | 说明 |
|-------------|---------|------|
| Eq 1: ṡ=f(s)+B(t)u+d_chute | `physics/dynamics.py` | 扩展动力学 |
| Eq 2: B(t)=B₀⊙M(φ,δ,t) | `physics/dynamics.py` | 故障掩码 |
| Eq 3: d_chute | `physics/dynamics.py` | 伞扰动 |
| Eq 4: Δμ_eff | `perception/degradation.py` | 组合退化度 |
| Eq 5: T_remain | `perception/endurance.py` | 剩余可飞时间 |
| Eq 6: CMDP | `decision/cmdp.py` | 约束MDP |
| Eq 7-10: GLR/FDI | `perception/fdi.py` | 故障检测隔离 |
| Eq 11-15: RLS | `perception/rls.py` | 在线辨识 |
| Eq 14: SVD截断 | `perception/degradation.py` | 数值秩 |
| Eq 16-17: 降落点 | `perception/landing_points.py` | KD-Tree |
| Eq 19: FME | `physics/envelope.py` | 可行运动包络 |
| Eq 20: CMDP Lagrangian | `decision/cmdp.py` | 拉格朗日训练 |
| Eq 22-23: CBF/CLF QP | `decision/cbf.py` | 安全过滤 |
| Eq 24: 路径代价 | `decision/path_planning.py` | 多目标路径 |
| Eq 25: MPC | `decision/mpc.py` | 滚动优化 |
| Eq 26: 控制分配 | `execution/control_alloc.py` | 伪逆 |
| Eq 27: LQR | `execution/tracker.py` | 轨迹跟踪 |
| Eq 28: 着陆 | `execution/landing.py` | 分段下降 |
