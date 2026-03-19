#!/usr/bin/env python3
"""
Claude Code agent pipeline for auto-psych.

Each agent runs as a full Claude Code instance (read/write files, run bash, multi-step reasoning).
Outputs go to cc_pipeline/projects/<project>/experimentN/ directories.

Usage:
  python3 cc_pipeline/run_pipeline_cc.py --project subjective_randomness --experiment 1
  python3 cc_pipeline/run_pipeline_cc.py --project subjective_randomness --experiments 3
  python3 cc_pipeline/run_pipeline_cc.py --project subjective_randomness --experiments 4-6
  python3 cc_pipeline/run_pipeline_cc.py --project subjective_randomness --experiment 1 --agent 1_theory
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cc_pipeline.orchestrator import (
    cc_projects_dir,
    ensure_experiment_dirs,
    experiment_dir,
    init_registry,
    run_analyze_programmatic,
    run_collect_programmatic,
    spawn_cc_agent,
    update_registry_from_interpretation,
    validate_cc_output,
    write_context,
)

AGENT_KEYS_STANDARD = ["1_theory", "2_design", "3_implement", "4_collect", "5_analyze", "6_interpret"]
AGENT_KEYS_CRITIQUE = ["1_theory", "2_design", "3_implement", "4_collect", "5_critique"]
PROGRAMMATIC_KEYS = {"4_collect", "5_analyze"}

DEFAULT_N_PARTICIPANTS = 5


def _parse_experiments(value: str) -> list[int]:
    """Parse --experiments: N (1..N) or A-B (A through B inclusive)."""
    value = value.strip()
    m = re.match(r"^(\d+)-(\d+)$", value)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start < 1 or end < 1 or start > end:
            raise ValueError("--experiments range must have start <= end, both >= 1")
        return list(range(start, end + 1))
    if value.isdigit():
        n = int(value)
        if n < 1:
            raise ValueError("--experiments must be >= 1")
        return list(range(1, n + 1))
    raise ValueError("--experiments must be N (1..N) or A-B (e.g. 4-6)")


def _run_agent(
    agent_key: str,
    exp_dir: Path,
    project_id: str,
    exp_num: int,
    mode: str,
    n_participants: int,
    prev_exp_dir: Optional[Path],
    validate: bool,
    ground_truth_model: Optional[str] = None,
) -> None:
    """Run one agent. Raises SystemExit if --validate and output is invalid."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment {exp_num} / Agent {agent_key}", flush=True)
    print(f"{'='*60}", flush=True)

    if agent_key in PROGRAMMATIC_KEYS:
        if agent_key == "4_collect":
            run_collect_programmatic(exp_dir, mode, n_participants,
                                     project_id=project_id,
                                     ground_truth_model=ground_truth_model)
        elif agent_key == "5_analyze":
            run_analyze_programmatic(exp_dir)
    else:
        write_context(
            exp_dir=exp_dir,
            agent_key=agent_key,
            project_id=project_id,
            exp_num=exp_num,
            prev_exp_dir=prev_exp_dir,
        )
        ok_spawn, output = spawn_cc_agent(agent_key=agent_key, exp_dir=exp_dir)
        if not ok_spawn:
            print(f"  [warn] Agent {agent_key} exited with non-zero status", flush=True)

    if validate:
        ok, msg = validate_cc_output(agent_key, exp_dir)
        if ok:
            print(f"  [ok] {agent_key}: {msg}", flush=True)
        else:
            print(f"  [error] Validation failed for {agent_key}: {msg}", file=sys.stderr)
            sys.exit(1)


def _run_experiment(
    project_id: str,
    exp_num: int,
    mode: str,
    n_participants: int,
    validate: bool,
    critique: bool,
    ground_truth_model: Optional[str] = None,
    agent_filter: Optional[str] = None,
) -> None:
    """Run all (or one) agents for a single experiment."""
    exp_dir_path = experiment_dir(project_id, exp_num)
    ensure_experiment_dirs(exp_dir_path, critique=critique)
    init_registry(exp_dir_path)

    prev_exp_dir = experiment_dir(project_id, exp_num - 1) if exp_num > 1 else None
    if prev_exp_dir and not prev_exp_dir.exists():
        prev_exp_dir = None

    agent_keys = AGENT_KEYS_CRITIQUE if critique else AGENT_KEYS_STANDARD
    keys_to_run = [agent_filter] if agent_filter else agent_keys

    for agent_key in keys_to_run:
        _run_agent(
            agent_key=agent_key,
            exp_dir=exp_dir_path,
            project_id=project_id,
            exp_num=exp_num,
            mode=mode,
            n_participants=n_participants,
            prev_exp_dir=prev_exp_dir,
            validate=validate,
            ground_truth_model=ground_truth_model,
        )

    if "6_interpret" in keys_to_run:
        update_registry_from_interpretation(exp_dir_path, critique=False)
    elif "5_critique" in keys_to_run:
        update_registry_from_interpretation(exp_dir_path, critique=True)

    print(f"\nExperiment {exp_num} complete. Outputs: {exp_dir_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude Code agent pipeline for auto-psych"
    )
    parser.add_argument("--project", required=True, help="Project ID (e.g. subjective_randomness)")
    parser.add_argument("--experiment", type=int, default=None, help="Single experiment number")
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        metavar="N|A-B",
        help="Experiments to run: N (1..N) or A-B (e.g. 4-6). Overrides --experiment.",
    )
    parser.add_argument(
        "--agent",
        choices=sorted(set(AGENT_KEYS_STANDARD) | set(AGENT_KEYS_CRITIQUE)),
        default=None,
        help="Run only this agent. Omit for full pipeline.",
    )
    parser.add_argument(
        "--critique",
        action="store_true",
        help="Use critique pipeline (5_critique) instead of standard analyze+interpret.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated_participants", "live"],
        default="simulated_participants",
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=DEFAULT_N_PARTICIPANTS,
        metavar="N",
    )
    parser.add_argument(
        "--ground-truth-model",
        default=None,
        metavar="MODEL",
        help="Generate synthetic participant data from this ground-truth model "
             "(must be in projects/<project>/ground_truth_models.py). "
             "If omitted, data is sampled from the theorist's models.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each agent's output; error and stop on failure.",
    )
    args = parser.parse_args()

    project_id = args.project
    prob_path = REPO_ROOT / "projects" / project_id / "problem_definition.md"
    if not prob_path.exists():
        print(f"Error: problem definition not found at {prob_path}", file=sys.stderr)
        sys.exit(1)

    if args.ground_truth_model is not None:
        sys.path.insert(0, str(REPO_ROOT))
        from src.models.ground_truth import get_ground_truth_model_names  # type: ignore
        allowed = get_ground_truth_model_names(project_id)
        if args.ground_truth_model not in allowed:
            print(f"Error: --ground-truth-model must be one of {allowed}; got {args.ground_truth_model!r}", file=sys.stderr)
            sys.exit(1)

    # Resolve experiment IDs
    if args.experiments is not None:
        try:
            exp_ids = _parse_experiments(args.experiments)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.experiment is not None:
        exp_ids = [args.experiment]
    else:
        print("Error: specify --experiment N or --experiments N", file=sys.stderr)
        sys.exit(1)

    pipeline = "critique" if args.critique else "standard"
    print(f"CC Pipeline: project={project_id} experiments={exp_ids} mode={args.mode} pipeline={pipeline} validate={args.validate}", flush=True)
    print(f"Outputs: {cc_projects_dir() / project_id}", flush=True)

    for exp_num in exp_ids:
        _run_experiment(
            project_id=project_id,
            exp_num=exp_num,
            mode=args.mode,
            n_participants=args.n_participants,
            validate=args.validate,
            critique=args.critique,
            ground_truth_model=args.ground_truth_model,
            agent_filter=args.agent,
        )

    print("\nAll experiments complete.", flush=True)


if __name__ == "__main__":
    main()
