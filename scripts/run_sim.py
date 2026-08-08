#!/usr/bin/env python3
"""
scripts/run_sim.py — 交互式仿真（原始小球仿真）

运行带有 HUD 的交互式仿真：
  python scripts/run_sim.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics.ball_sim_jax import BallSimulator

if __name__ == "__main__":
    BallSimulator().run()
