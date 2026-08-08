#!/usr/bin/env python3
"""
scripts/run_mc.py — 蒙特卡洛安全评估

150次试验，覆盖多故障场景，
输出 Clopper-Pearson 置信区间。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.monte_carlo import run_monte_carlo
from src.env.fault_injection import FaultScenario

if __name__ == "__main__":
    scenarios = [
        ("normal",      FaultScenario()),
        ("power_30",    FaultScenario(delta=0.3)),
        ("power_50",    FaultScenario(delta=0.5)),
        ("power_70",    FaultScenario(delta=0.7)),
        ("chute_fail",  FaultScenario(phi=1.0)),
        ("chute_half",  FaultScenario(phi=0.5)),
        ("combined",    FaultScenario(phi=0.5, delta=0.5, asym=0.3)),
    ]
    run_monte_carlo(n_trials=150 // len(scenarios), scenarios=scenarios)
