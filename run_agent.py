#!/usr/bin/env python3
"""
Run a single pipeline agent (convenience entrypoint). Delegates to run_pipeline.py --agent.

Usage:
  python3 run_agent.py --project subjective_randomness --run 1 --agent 1_theory
  python3 run_agent.py --project X --run 2 --agent 2_design --state-from-run 1
  python3 run_agent.py --project X --run 1 --agent 1_theory --use-fixtures

Equivalently: python3 run_pipeline.py --project subjective_randomness --run 1 --agent 1_theory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pipeline import main

if __name__ == "__main__":
    main()
