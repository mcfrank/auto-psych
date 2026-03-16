#!/usr/bin/env python3
"""
Entrypoint for the auto-psych pipeline. Run full graph or a single agent.

Usage:
  python3 run_pipeline.py --project subjective_randomness --run 1 --mode simulated_participants
  python3 run_pipeline.py --project subjective_randomness --runs 3 --mode simulated_participants
  python3 run_pipeline.py --project subjective_randomness --runs 4-6 --mode simulated_participants
  python3 run_pipeline.py --project subjective_randomness --run 1 --agent 1_theory   # single agent
  python3 run_pipeline.py --project X --run 1 --agent 2_design --state-from-run 1
  python3 run_pipeline.py --project X --run 1 --n-participants 10 --max-retries 5
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    REPO_ROOT,
    batches_dir,
    problem_definition_path,
    run_dir,
    run_dir_for_state,
    DEFAULT_SIMULATED_N_PARTICIPANTS,
    DEFAULT_MAX_VALIDATION_RETRIES,
)
from src.console_log import run_banner, agent_header, log_status
from src.prompts import resolve_prompts, archive_prompts_for_run
from src.graph import build_graph
from src.state_loader import load_state_from_run, minimal_state_for_agent
from src.batch_plots import append_correlations_to_batch_csv, plot_correlations_by_run

AGENT_SUBDIRS = [
    "1_theory", "2_design", "3_implement", "4_collect", "5_analyze", "6_interpret",
]

def _git_commit_hash() -> tuple[str | None, str]:
    """Return (full_hash, short_for_dir). short is 7 chars or 'nogit'/'nogit_dirty'."""
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    full = r.stdout.strip() if r.returncode == 0 and r.stdout else None
    d = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    dirty = d.returncode == 0 and bool(d.stdout.strip())
    if not full:
        return None, "nogit_dirty" if dirty else "nogit"
    short = full[:7] + ("_dirty" if dirty else "")
    return full, short


def _create_batch_dir(project_id: str) -> Path:
    """Create batch_YYYYMMDD-HHMM_shortHash under project batches; write commit_hash.txt. Return batch path."""
    bdir = batches_dir(project_id)
    bdir.mkdir(parents=True, exist_ok=True)
    full_hash, short = _git_commit_hash()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = f"batch_{stamp}_{short}"
    path = bdir / name
    path.mkdir(parents=True, exist_ok=True)
    meta = f"commit={full_hash or 'none'}\ndirty={full_hash is None or '_dirty' in short}\ntimestamp={stamp}\n"
    (path / "commit_hash.txt").write_text(meta, encoding="utf-8")
    return path


def _latest_batch_dir(project_id: str) -> Path | None:
    """Return path to the latest batch directory (by name sort), or None if none exist."""
    bdir = batches_dir(project_id)
    if not bdir.exists():
        return None
    subdirs = [d for d in bdir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    if not subdirs:
        return None
    subdirs.sort(key=lambda d: d.name, reverse=True)
    return subdirs[0]


def _parse_runs(value: str) -> list[int]:
    """
    Parse --runs value: either N (runs 1..N) or A-B (runs A through B inclusive).
    Returns list of run IDs.
    """
    value = value.strip()
    # Range: one or more digits, optional hyphen, one or more digits
    m = re.match(r"^(\d+)-(\d+)$", value)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start < 1 or end < 1:
            raise ValueError("--runs range bounds must be >= 1")
        if start > end:
            raise ValueError("--runs range must be start-end with start <= end")
        return list(range(start, end + 1))
    # Single number N → runs 1..N
    if value.isdigit():
        n = int(value)
        if n < 1:
            raise ValueError("--runs must be >= 1")
        return list(range(1, n + 1))
    raise ValueError("--runs must be a number N (runs 1..N) or a range A-B (e.g. 4-6)")


def _get_agent_fn(agent_key: str):
    """Resolve agent key to runnable function (lazy import)."""
    mod_map = {
        "1_theory": ("src.agents.theorist", "run_theorist"),
        "2_design": ("src.agents.experiment_designer", "run_experiment_designer"),
        "3_implement": ("src.agents.experiment_implementer", "run_experiment_implementer"),
        "4_collect": ("src.agents.collect", "run_collect"),
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
    batch_dir: Path | str | None = None,
) -> dict:
    state = {"batch_dir": str(batch_dir) if batch_dir else None}
    rdir = run_dir_for_state(project_id, run_id, state) if state.get("batch_dir") else run_dir(project_id, run_id)
    registry_path = rdir / "model_registry.yaml"
    state.update({
        "project_id": project_id,
        "run_id": run_id,
        "mode": mode,
        "problem_definition_path": str(prob_path),
        "registry_path": str(registry_path),
        "simulated_n_participants": simulated_n_participants,
        "max_validation_retries": max_validation_retries,
    })
    if not state.get("batch_dir"):
        del state["batch_dir"]
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
    run_index: int | None = None,
    batch_dir: Path | None = None,
) -> None:
    run_banner(run_id, total_runs, run_index)
    initial_state = _build_initial_state(
        project_id, run_id, mode, prob_path, None,
        simulated_n_participants=simulated_n_participants,
        max_validation_retries=max_validation_retries,
        batch_dir=batch_dir,
    )
    rdir = run_dir_for_state(project_id, run_id, initial_state)
    rdir.mkdir(parents=True, exist_ok=True)
    for key in AGENT_SUBDIRS:
        (rdir / key).mkdir(exist_ok=True)
    resolved = resolve_prompts(project_id)
    archive_prompts_for_run(project_id, run_id, resolved, run_dir_base=rdir)
    prev_report = None
    if run_id > 1:
        prev_dir = run_dir_for_state(project_id, run_id - 1, initial_state)
        prev = prev_dir / "6_interpret" / "report.md"
        if prev.exists():
            prev_report = str(prev)
    if interpreter_report:
        prev_report = interpreter_report
    initial_state["interpreter_report_path"] = prev_report or ""
    if total_runs is not None:
        initial_state["total_runs"] = total_runs
    if batch_dir:
        log_status(f"batch_dir={batch_dir}", indent=False)
    graph = build_graph()
    graph.invoke(initial_state)
    if batch_dir:
        corr_path = rdir / "6_interpret" / "model_correlations.yaml"
        if corr_path.exists():
            import yaml
            try:
                data = yaml.safe_load(corr_path.read_text(encoding="utf-8")) or {}
                corr = data.get("correlations") or {}
                if corr:
                    append_correlations_to_batch_csv(batch_dir, run_id, corr)
            except Exception:
                pass
        plot_path = plot_correlations_by_run(batch_dir)
        if plot_path.exists():
            log_status(f"Updated {plot_path}", indent=False)
    print(f"Run {run_id} completed.", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auto-psych pipeline (full or single agent)")
    parser.add_argument("--project", required=True, help="Project id (e.g. subjective_randomness)")
    parser.add_argument("--run", type=int, default=None, help="Run number (e.g. 1). Required unless --runs set.")
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        metavar="N|A-B",
        help="Runs to execute: N (runs 1..N) or A-B (runs A through B, e.g. 4-6). Overrides --run.",
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENT_SUBDIRS),
        default=None,
        help="Run only this agent (e.g. 1_theory). Omit to run full pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated_participants", "live", "test_prolific"],
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
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add runs to the latest batch (use with --run or --runs). Do not create a new batch.",
    )
    args = parser.parse_args()

    if args.append and args.agent is not None:
        print("Error: --append cannot be used with --agent.", file=sys.stderr)
        sys.exit(1)
    if args.append and args.run is None and args.runs is None:
        print("Error: --append requires --run N or --runs N.", file=sys.stderr)
        sys.exit(1)

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

    # Full pipeline (multi-run with optional batch)
    if args.runs is not None:
        try:
            run_ids = _parse_runs(args.runs)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        total_runs = len(run_ids)
        batch_dir = None
        if args.append:
            batch_dir = _latest_batch_dir(project_id)
            if batch_dir is None:
                print("Error: --append used but no existing batch found.", file=sys.stderr)
                sys.exit(1)
            log_status(f"Appending to batch: {batch_dir}", indent=False)
        else:
            batch_dir = _create_batch_dir(project_id)
            full_hash, short = _git_commit_hash()
            log_status(f"Batch: {batch_dir} (commit {short})", indent=False)
        for run_index, run_id in enumerate(run_ids, start=1):
            _run_full_pipeline(
                project_id, run_id, mode, prob_path,
                interpreter_report=None, total_runs=total_runs,
                simulated_n_participants=args.n_participants,
                max_validation_retries=args.max_retries,
                run_index=run_index,
                batch_dir=batch_dir,
            )
        print("All runs completed.", file=sys.stderr, flush=True)
        return

    # Full pipeline (single run, optionally in latest batch with --append)
    run_id = args.run
    if run_id is None:
        print("Error: specify --run N or --runs N", file=sys.stderr)
        sys.exit(1)
    batch_dir = None
    if args.append:
        batch_dir = _latest_batch_dir(project_id)
        if batch_dir is None:
            print("Error: --append used but no existing batch found.", file=sys.stderr)
            sys.exit(1)
        log_status(f"Appending to batch: {batch_dir}", indent=False)
    _run_full_pipeline(
        project_id, run_id, mode, prob_path,
        args.interpreter_report, total_runs=None,
        simulated_n_participants=args.n_participants,
        max_validation_retries=args.max_retries,
        batch_dir=batch_dir,
    )


if __name__ == "__main__":
    main()
