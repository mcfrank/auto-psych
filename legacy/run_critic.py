#!/usr/bin/env python3
"""
Run validators on a run (or a single agent) and write results to the run directory.
Exits with non-zero status if any validation fails.

Usage:
  python3 run_critic.py --project subjective_randomness --run 1
  python3 run_critic.py --project subjective_randomness --run 1 --agent 1_theory
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import run_dir
from src.validation.validators import (
    AGENT_VALIDATORS,
    Validated,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate agent outputs for a run")
    parser.add_argument("--project", required=True, help="Project id")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument(
        "--agent",
        default=None,
        choices=list(AGENT_VALIDATORS),
        help="Validate only this agent (default: all)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write validation.json files; only print and exit code",
    )
    args = parser.parse_args()

    project_id = args.project
    run_id = args.run
    rdir = run_dir(project_id, run_id)

    if not rdir.exists():
        print(f"Error: run directory not found: {rdir}", file=sys.stderr)
        sys.exit(2)

    agents_to_run = [args.agent] if args.agent else list(AGENT_VALIDATORS)
    results = {}
    any_fail = False

    for agent_key in agents_to_run:
        validator_fn = AGENT_VALIDATORS[agent_key]
        v = validator_fn(rdir)
        results[agent_key] = {"ok": v.ok, "message": v.message, "details": v.details}
        if not v.ok:
            any_fail = True
        print(f"{agent_key}: {'PASS' if v.ok else 'FAIL'} - {v.message}", file=sys.stderr)
        if not args.no_write:
            out_path = rdir / agent_key / "validation.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps({"ok": v.ok, "message": v.message, "details": v.details}, indent=2),
                encoding="utf-8",
            )

    if any_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
