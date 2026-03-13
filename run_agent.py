#!/usr/bin/env python3
"""
Run a single agent in isolation for debugging. Does not trigger the next agent.

Usage:
  python run_agent.py --project subjective_randomness --run 1 --agent 1_theory
  python run_agent.py --project subjective_randomness --run 2 --agent 2_design --state-from-run 1
  python run_agent.py --project subjective_randomness --run 1 --agent 1_theory --use-fixtures
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import problem_definition_path, run_dir
from src.console_log import run_banner, agent_header
from src.prompts import resolve_prompts, archive_prompts_for_run
from src.state_loader import load_state_from_run, minimal_state_for_agent
from src.agents.theorist import run_theorist
from src.agents.experiment_designer import run_experiment_designer
from src.agents.experiment_implementer import run_experiment_implementer
from src.agents.simulated_participant import run_simulated_participant
from src.agents.data_analyst import run_data_analyst
from src.agents.interpreter import run_interpreter


AGENT_FUNCS = {
    "1_theory": run_theorist,
    "2_design": run_experiment_designer,
    "3_implement": run_experiment_implementer,
    "4_collect": run_simulated_participant,
    "5_analyze": run_data_analyst,
    "6_interpret": run_interpreter,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single pipeline agent (no next agent)")
    parser.add_argument("--project", required=True, help="Project id")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--agent", required=True, choices=list(AGENT_FUNCS), help="Agent to run (e.g. 1_theory)")
    parser.add_argument(
        "--state-from-run",
        type=int,
        default=None,
        help="Load artifact paths from this run id (e.g. 1); current run id is still used for writing outputs",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Use minimal state with tests/fixtures for missing inputs (for running without a prior full run)",
    )
    parser.add_argument("--mode", choices=["simulated_participants", "live"], default="simulated_participants")
    args = parser.parse_args()

    project_id = args.project
    run_id = args.run
    agent_key = args.agent

    # Validate project and problem definition
    prob_path = problem_definition_path(project_id)
    if not prob_path.exists() and not args.use_fixtures:
        print(f"Error: problem definition not found at {prob_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure run dir and agent dir exist
    rdir = run_dir(project_id, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / agent_key).mkdir(exist_ok=True)

    # Resolve and archive prompts so agent can load its prompt
    resolved = resolve_prompts(project_id)
    archive_prompts_for_run(project_id, run_id, resolved)

    # Build state
    if args.state_from_run is not None:
        state = load_state_from_run(project_id, run_id, reference_run_id=args.state_from_run, mode=args.mode)
        # Ensure we write to current run, not reference
        state["run_id"] = run_id
    elif args.use_fixtures:
        state = minimal_state_for_agent(agent_key, project_id, run_id)
    else:
        state = load_state_from_run(project_id, run_id, mode=args.mode)

    state["mode"] = args.mode

    run_banner(run_id)
    agent_header(agent_key, run_id, total_runs=None, mode=state["mode"])

    try:
        result = AGENT_FUNCS[agent_key](state)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

    from src.console_log import log_status
    written = [k for k in _outputs_for_agent(agent_key) if result.get(k)]
    paths_written = [result[k] for k in written if isinstance(result.get(k), str)]
    if paths_written:
        log_status(f"Wrote: {paths_written[0]}" + (f" (+{len(paths_written)-1} more)" if len(paths_written) > 1 else ""))
    print("Done.", file=sys.stderr, flush=True)


def _inputs_for_agent(agent_key: str) -> list:
    return {
        "1_theory": ["problem_definition_path", "interpreter_report_path"],
        "2_design": ["problem_definition_path", "theorist_manifest_path"],
        "3_implement": ["problem_definition_path", "stimuli_path"],
        "4_collect": ["stimuli_path", "theorist_manifest_path", "deployment_config_path"],
        "5_analyze": ["simulated_data_path", "deployment_config_path", "theorist_manifest_path"],
        "6_interpret": ["summary_stats_path", "aggregate_csv_path", "theorist_manifest_path"],
    }.get(agent_key, [])


def _outputs_for_agent(agent_key: str) -> list:
    return {
        "1_theory": ["theorist_manifest_path", "theorist_rationale_path"],
        "2_design": ["stimuli_path", "design_rationale_path"],
        "3_implement": ["experiment_path", "deployment_config_path"],
        "4_collect": ["simulated_data_path"],
        "5_analyze": ["summary_stats_path", "aggregate_csv_path"],
        "6_interpret": ["interpreter_report_path"],
    }.get(agent_key, [])


if __name__ == "__main__":
    main()
