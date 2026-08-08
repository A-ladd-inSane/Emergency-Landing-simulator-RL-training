#!/usr/bin/env python3
"""
scripts/run_pipeline.py — 端到端流水线演示

依次执行全部7个模块的验证测试。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_full_pipeline import main

if __name__ == "__main__":
    main()
