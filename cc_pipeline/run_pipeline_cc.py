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

AGENT_KEYS = ["1_theory", "2_design", "3_implement", "4_collect", "5_analyze", "6_interpret"]
# These agents run as CC agents (Claude Code CLI)
CC_AGENT_KEYS = {"1_theory", "2_design", "3_implement", "6_interpret"}
# These run programmatically (existing Python logic)
PROGRAMMATIC_KEYS = {"4_collect", "5_analyze"}

DEFAULT_N_PARTICIPANTS = 5
DEFAULT_MAX_RETRIES = 3


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
    max_retries: int,
    prev_exp_dir: Path | None,
) -> bool:
    """
    Run one agent. Returns True if successful (or retries exhausted — pipeline continues).
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment {exp_num} / Agent {agent_key}", flush=True)
    print(f"{'='*60}", flush=True)

    if agent_key in PROGRAMMATIC_KEYS:
        # Run programmatically, no CC agent needed
        if agent_key == "4_collect":
            run_collect_programmatic(exp_dir, mode, n_participants)
        elif agent_key == "5_analyze":
            run_analyze_programmatic(exp_dir)
        ok, msg = validate_cc_output(agent_key, exp_dir)
        if not ok:
            print(f"  [warn] Validation failed for {agent_key}: {msg}", flush=True)
        return True

    # CC agent path
    write_context(
        exp_dir=exp_dir,
        agent_key=agent_key,
        project_id=project_id,
        exp_num=exp_num,
        prev_exp_dir=prev_exp_dir,
    )

    validation_feedback = ""
    for attempt in range(1, max_retries + 2):
        ok_spawn, output = spawn_cc_agent(
            agent_key=agent_key,
            exp_dir=exp_dir,
            validation_feedback=validation_feedback,
        )
        if not ok_spawn:
            print(f"  [warn] Agent {agent_key} spawn failed (attempt {attempt}): {output[:200]}", flush=True)

        ok_val, val_msg = validate_cc_output(agent_key, exp_dir)
        if ok_val:
            print(f"  [ok] {agent_key} validated: {val_msg}", flush=True)
            return True

        if attempt > max_retries:
            print(f"  [warn] {agent_key} failed validation after {max_retries} retries: {val_msg}", flush=True)
            print(f"  [warn] Continuing pipeline despite validation failure.", flush=True)
            return False

        print(f"  [retry] {agent_key} attempt {attempt}/{max_retries} failed: {val_msg}", flush=True)
        validation_feedback = val_msg

    return False


def _run_experiment(
    project_id: str,
    exp_num: int,
    mode: str,
    n_participants: int,
    max_retries: int,
    agent_filter: Optional[str] = None,
) -> None:
    """Run all (or one) agents for a single experiment."""
    exp_dir_path = experiment_dir(project_id, exp_num)
    ensure_experiment_dirs(exp_dir_path)
    init_registry(exp_dir_path)

    prev_exp_dir = experiment_dir(project_id, exp_num - 1) if exp_num > 1 else None
    if prev_exp_dir and not prev_exp_dir.exists():
        prev_exp_dir = None

    keys_to_run = [agent_filter] if agent_filter else AGENT_KEYS

    for agent_key in keys_to_run:
        _run_agent(
            agent_key=agent_key,
            exp_dir=exp_dir_path,
            project_id=project_id,
            exp_num=exp_num,
            mode=mode,
            n_participants=n_participants,
            max_retries=max_retries,
            prev_exp_dir=prev_exp_dir,
        )

    # After interpretation, sync registry
    if "6_interpret" in keys_to_run:
        update_registry_from_interpretation(exp_dir_path)

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
        choices=AGENT_KEYS,
        default=None,
        help="Run only this agent (e.g. 1_theory). Omit for full pipeline.",
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
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        metavar="N",
    )
    args = parser.parse_args()

    project_id = args.project
    prob_path = REPO_ROOT / "projects" / project_id / "problem_definition.md"
    if not prob_path.exists():
        print(f"Error: problem definition not found at {prob_path}", file=sys.stderr)
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

    print(f"CC Pipeline: project={project_id} experiments={exp_ids} mode={args.mode}", flush=True)
    print(f"Outputs: {cc_projects_dir() / project_id}", flush=True)

    for exp_num in exp_ids:
        _run_experiment(
            project_id=project_id,
            exp_num=exp_num,
            mode=args.mode,
            n_participants=args.n_participants,
            max_retries=args.max_retries,
            agent_filter=args.agent,
        )

    print("\nAll experiments complete.", flush=True)


if __name__ == "__main__":
    main()
