#!/usr/bin/env python3
"""
Entrypoint for the auto-psych pipeline. Run full graph or a single agent.

Usage:
  python3 run_pipeline.py --project subjective_randomness --run 1 --mode simulated_participants
  python3 run_pipeline.py --project subjective_randomness --runs 3 --mode simulated_participants
  python3 run_pipeline.py --project subjective_randomness --run 1 --agent 1_theory   # single agent
  python3 run_pipeline.py --project X --run 1 --agent 2_design --state-from-run 1
  python3 run_pipeline.py --project X --run 1 --n-participants 10 --max-retries 5
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    problem_definition_path,
    run_dir,
    DEFAULT_SIMULATED_N_PARTICIPANTS,
    DEFAULT_MAX_VALIDATION_RETRIES,
)
from src.console_log import run_banner, agent_header, log_status
from src.prompts import resolve_prompts, archive_prompts_for_run
from src.graph import build_graph
from src.state_loader import load_state_from_run, minimal_state_for_agent

AGENT_SUBDIRS = [
    "1_theory", "2_design", "3_implement", "4_collect", "5_analyze", "6_interpret",
]

def _get_agent_fn(agent_key: str):
    """Resolve agent key to runnable function (lazy import)."""
    mod_map = {
        "1_theory": ("src.agents.theorist", "run_theorist"),
        "2_design": ("src.agents.experiment_designer", "run_experiment_designer"),
        "3_implement": ("src.agents.experiment_implementer", "run_experiment_implementer"),
        "4_collect": ("src.agents.simulated_participant", "run_simulated_participant"),
        "5_analyze": ("src.agents.data_analyst", "run_data_analyst"),
        "6_interpret": ("src.agents.interpreter", "run_interpreter"),
    }
    mod_name, attr = mod_map[agent_key]
    mod = __import__(mod_name, fromlist=[attr])
    return getattr(mod, attr)


def _build_initial_state(
    project_id: str,
    run_id: int,
    mode: str,
    prob_path: Path,
    interpreter_report_path: str | None = None,
    simulated_n_participants: int = DEFAULT_SIMULATED_N_PARTICIPANTS,
    max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES,
) -> dict:
    rdir = run_dir(project_id, run_id)
    registry_path = rdir / "model_registry.yaml"
    state = {
        "project_id": project_id,
        "run_id": run_id,
        "mode": mode,
        "problem_definition_path": str(prob_path),
        "registry_path": str(registry_path),
        "simulated_n_participants": simulated_n_participants,
        "max_validation_retries": max_validation_retries,
    }
    if interpreter_report_path:
        state["interpreter_report_path"] = interpreter_report_path
    return state


def _run_single_agent(
    project_id: str,
    run_id: int,
    agent_key: str,
    state: dict,
) -> None:
    run_banner(run_id)
    agent_header(agent_key, run_id, total_runs=None, mode=state.get("mode", "simulated_participants"))
    fn = _get_agent_fn(agent_key)
    result = fn(state)
    outputs = {
        "1_theory": ["theorist_manifest_path", "theorist_rationale_path"],
        "2_design": ["stimuli_path", "design_rationale_path"],
        "3_implement": ["experiment_path", "deployment_config_path"],
        "4_collect": ["simulated_data_path"],
        "5_analyze": ["summary_stats_path", "aggregate_csv_path"],
        "6_interpret": ["interpreter_report_path"],
    }.get(agent_key, [])
    written = [result.get(k) for k in outputs if result.get(k)]
    if written:
        log_status(f"Wrote: {written[0]}" + (f" (+{len(written)-1} more)" if len(written) > 1 else ""))
    print("Done.", file=sys.stderr, flush=True)


def _run_full_pipeline(
    project_id: str,
    run_id: int,
    mode: str,
    prob_path: Path,
    interpreter_report: str | None,
    total_runs: int | None,
    simulated_n_participants: int,
    max_validation_retries: int,
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
        prev = run_dir(project_id, run_id - 1) / "6_interpret" / "report.md"
        if prev.exists():
            prev_report = str(prev)
    if interpreter_report:
        prev_report = interpreter_report
    initial_state = _build_initial_state(
        project_id, run_id, mode, prob_path, prev_report,
        simulated_n_participants=simulated_n_participants,
        max_validation_retries=max_validation_retries,
    )
    if total_runs is not None:
        initial_state["total_runs"] = total_runs
    graph = build_graph()
    graph.invoke(initial_state)
    print(f"Run {run_id} completed.", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auto-psych pipeline (full or single agent)")
    parser.add_argument("--project", required=True, help="Project id (e.g. subjective_randomness)")
    parser.add_argument("--run", type=int, default=None, help="Run number (e.g. 1). Required unless --runs set.")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Execute runs 1 through N. Overrides --run.",
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENT_SUBDIRS),
        default=None,
        help="Run only this agent (e.g. 1_theory). Omit to run full pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated_participants", "live"],
        default="simulated_participants",
        help="Run mode",
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=DEFAULT_SIMULATED_N_PARTICIPANTS,
        metavar="N",
        help=f"Number of simulated participants (default: {DEFAULT_SIMULATED_N_PARTICIPANTS})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_VALIDATION_RETRIES,
        metavar="N",
        help=f"Max validation retries per agent (default: {DEFAULT_MAX_VALIDATION_RETRIES})",
    )
    parser.add_argument(
        "--state-from-run",
        type=int,
        default=None,
        help="Load artifact paths from this run (single-agent only); outputs still go to --run.",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Use tests/fixtures for missing inputs (single-agent only).",
    )
    parser.add_argument(
        "--interpreter-report",
        type=str,
        default=None,
        help="Path to interpreter report from previous run (theorist context; single-run only)",
    )
    args = parser.parse_args()

    project_id = args.project
    mode = args.mode
    prob_path = problem_definition_path(project_id)

    if not prob_path.exists() and not (args.agent and args.use_fixtures):
        print(f"Error: problem definition not found at {prob_path}", file=sys.stderr)
        sys.exit(1)

    # Single-agent mode (same behavior as legacy run_agent.py)
    if args.agent is not None:
        if args.runs is not None:
            print("Error: --agent and --runs cannot be used together.", file=sys.stderr)
            sys.exit(1)
        run_id = args.run
        if run_id is None:
            print("Error: --run N required when using --agent.", file=sys.stderr)
            sys.exit(1)
        rdir = run_dir(project_id, run_id)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / args.agent).mkdir(exist_ok=True)
        resolved = resolve_prompts(project_id)
        archive_prompts_for_run(project_id, run_id, resolved)
        if args.state_from_run is not None:
            state = load_state_from_run(project_id, run_id, reference_run_id=args.state_from_run, mode=mode)
            state["run_id"] = run_id
        elif args.use_fixtures:
            from src.config import REPO_ROOT
            state = minimal_state_for_agent(args.agent, project_id, run_id, fixtures_dir=REPO_ROOT / "tests" / "fixtures")
        else:
            state = load_state_from_run(project_id, run_id, mode=mode)
        state["mode"] = mode
        state["simulated_n_participants"] = args.n_participants
        state["max_validation_retries"] = args.max_retries
        _run_single_agent(project_id, run_id, args.agent, state)
        return

    # Full pipeline
    if args.runs is not None:
        num_runs = args.runs
        if num_runs < 1:
            print("Error: --runs must be >= 1", file=sys.stderr)
            sys.exit(1)
        for run_id in range(1, num_runs + 1):
            _run_full_pipeline(
                project_id, run_id, mode, prob_path,
                interpreter_report=None, total_runs=num_runs,
                simulated_n_participants=args.n_participants,
                max_validation_retries=args.max_retries,
            )
        print("All runs completed.", file=sys.stderr, flush=True)
        return

    run_id = args.run
    if run_id is None:
        print("Error: specify --run N or --runs N", file=sys.stderr)
        sys.exit(1)
    _run_full_pipeline(
        project_id, run_id, mode, prob_path,
        args.interpreter_report, total_runs=None,
        simulated_n_participants=args.n_participants,
        max_validation_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
