#!/usr/bin/env python3
"""
Entrypoint for the auto-psych LangGraph pipeline.

Usage:
  python run_pipeline.py --project subjective_randomness --run 1 --mode simulated_participants
  python run_pipeline.py --project subjective_randomness --runs 3 --mode simulated_participants
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import problem_definition_path, run_dir, prompts_used_dir
from src.console_log import run_banner
from src.prompts import resolve_prompts, archive_prompts_for_run
from src.graph import build_graph

AGENT_SUBDIRS = [
    "1theorist", "2experiment_designer", "3experiment_implementer", "4deployer",
    "5simulated_participant", "6data_analyst", "7interpreter",
]


def _build_initial_state(
    project_id: str,
    run_id: int,
    mode: str,
    prob_path: Path,
    interpreter_report_path: str | None = None,
) -> dict:
    rdir = run_dir(project_id, run_id)
    registry_path = rdir / "model_registry.yaml"
    state = {
        "project_id": project_id,
        "run_id": run_id,
        "mode": mode,
        "problem_definition_path": str(prob_path),
        "registry_path": str(registry_path),
    }
    if interpreter_report_path:
        state["interpreter_report_path"] = interpreter_report_path
    return state


def _run_single(
    project_id: str,
    run_id: int,
    mode: str,
    prob_path: Path,
    interpreter_report: str | None,
    total_runs: int | None = None,
) -> None:
    run_banner(run_id, total_runs)
    rdir = run_dir(project_id, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    for key in AGENT_SUBDIRS:
        (rdir / key).mkdir(exist_ok=True)
    resolved = resolve_prompts(project_id)
    archive_prompts_for_run(project_id, run_id, resolved)
    prev_report = None
    if run_id > 1:
        prev = run_dir(project_id, run_id - 1) / "7interpreter" / "report.md"
        if prev.exists():
            prev_report = str(prev)
    if interpreter_report:
        prev_report = interpreter_report
    initial_state = _build_initial_state(project_id, run_id, mode, prob_path, prev_report)
    if total_runs is not None:
        initial_state["total_runs"] = total_runs
    graph = build_graph()
    result = graph.invoke(initial_state)
    print(f"Run {run_id} completed.", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auto-psych experiment pipeline")
    parser.add_argument("--project", required=True, help="Project id (e.g. subjective_randomness)")
    parser.add_argument("--run", type=int, default=None, help="Single run number (e.g. 1). Ignored if --runs is set.")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of runs to execute (run 1 through N). If set, overrides --run.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated_participants", "live"],
        default="simulated_participants",
        help="Run mode",
    )
    parser.add_argument(
        "--interpreter-report",
        type=str,
        default=None,
        help="Path to interpreter report from previous run (for theorist context; single-run only)",
    )
    args = parser.parse_args()

    project_id = args.project
    mode = args.mode

    prob_path = problem_definition_path(project_id)
    if not prob_path.exists():
        print(f"Error: problem definition not found at {prob_path}", file=sys.stderr)
        sys.exit(1)

    if args.runs is not None:
        num_runs = args.runs
        if num_runs < 1:
            print("Error: --runs must be >= 1", file=sys.stderr)
            sys.exit(1)
        for run_id in range(1, num_runs + 1):
            _run_single(project_id, run_id, mode, prob_path, interpreter_report=None, total_runs=num_runs)
        print("All runs completed.", file=sys.stderr, flush=True)
        return

    run_id = args.run
    if run_id is None:
        print("Error: specify either --run N or --runs N", file=sys.stderr)
        sys.exit(1)
    _run_single(project_id, run_id, mode, prob_path, args.interpreter_report, total_runs=None)


if __name__ == "__main__":
    main()
