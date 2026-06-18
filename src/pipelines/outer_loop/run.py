#!/usr/bin/env python3
"""
Claude Code agent pipeline for auto-psych.

Each agent runs as a full Claude Code instance (read/write files, run bash, multi-step reasoning).
Outputs go to data/outer_loop/<project>/experimentN/ directories.

Usage:
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiments 3
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiments 4-6
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --agent 1_theory
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.outer_loop.deployment import write_smoke_experiment
from src.pipelines.outer_loop.orchestrator import (
    ensure_experiment_dirs,
    experiment_dir,
    get_ground_truth_models,
    init_registry,
    outer_data_dir,
    outer_project_dir,
    run_collect_programmatic,
    run_deployment_programmatic,
    run_inner_model_loop_programmatic,
    seed_experiment_models_from_project,
    spawn_cc_agent,
    update_registry_from_interpretation,
    validate_cc_output,
    write_context,
)
from src.pipelines.outer_loop.participants import DEFAULT_OPEN_MODEL
from src.runtime.coding_agent import select_backend

AGENT_KEYS = ["1_theory", "2_design", "3_implement", "4_collect", "5_model_loop"]

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
    inner_loop_iterations: int = 2,
    inner_loop_candidates: int = 3,
    fit_kwargs: Optional[dict] = None,
    backend: Optional[str] = None,
    participant_backend: str = "closed",
    participant_model: Optional[str] = None,
    prolific_mode: str = "none",
    enable_critique: bool = True,
    n_critique_proposals: Optional[int] = None,
) -> None:
    """Run one agent. Raises SystemExit if --validate and output is invalid."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Experiment {exp_num} / Agent {agent_key}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if agent_key == "4_collect":
        run_collect_programmatic(
            exp_dir,
            mode,
            n_participants,
            project_id=project_id,
            ground_truth_model=ground_truth_model,
            participant_backend=participant_backend,
            participant_model=participant_model,
            prolific_mode=prolific_mode,
        )
    elif agent_key == "5_model_loop":
        run_inner_model_loop_programmatic(
            exp_dir,
            max_iterations=inner_loop_iterations,
            candidate_count=inner_loop_candidates,
            fit_kwargs=fit_kwargs,
            enable_critique=enable_critique,
            n_critique_proposals=n_critique_proposals,
        )
    else:
        write_context(
            exp_dir=exp_dir,
            agent_key=agent_key,
            project_id=project_id,
            exp_num=exp_num,
            prev_exp_dir=prev_exp_dir,
        )
        allowed_dirs = [exp_dir, outer_project_dir(project_id)]
        if prev_exp_dir:
            allowed_dirs.append(prev_exp_dir)
        ok_spawn, output = spawn_cc_agent(
            agent_key=agent_key,
            exp_dir=exp_dir,
            allowed_dirs=allowed_dirs,
            backend=backend,
        )
        if not ok_spawn:
            print(f"  [warn] Agent {agent_key} exited with non-zero status", flush=True)

    if validate:
        ok, msg = validate_cc_output(agent_key, exp_dir)
        if ok:
            print(f"  [ok] {agent_key}: {msg}", flush=True)
        else:
            print(
                f"  [error] Validation failed for {agent_key}: {msg}", file=sys.stderr
            )
            sys.exit(1)


def _run_experiment(
    project_id: str,
    exp_num: int,
    mode: str,
    n_participants: int,
    validate: bool,
    resume: bool = False,
    ground_truth_model: Optional[str] = None,
    agent_filter: Optional[str] = None,
    inner_loop_iterations: int = 2,
    inner_loop_candidates: int = 3,
    fit_kwargs: Optional[dict] = None,
    backend: Optional[str] = None,
    participant_backend: str = "closed",
    participant_model: Optional[str] = None,
    deploy_target: str = "none",
    collection_owner: str = "unknown",
    firebase_project: Optional[str] = None,
    firebase_region: str = "us-central1",
    prolific_mode: str = "none",
    deploy_only: bool = False,
    prepare_smoke_experiment: bool = False,
    enable_critique: bool = True,
    n_critique_proposals: Optional[int] = None,
) -> None:
    """Run all (or one) agents for a single experiment."""
    exp_dir_path = experiment_dir(project_id, exp_num)
    if exp_dir_path.exists() and not resume:
        print(
            f"Error: experiment directory already exists: {exp_dir_path}",
            file=sys.stderr,
        )
        print("Use --resume to run into an existing directory.", file=sys.stderr)
        sys.exit(1)
    ensure_experiment_dirs(exp_dir_path)
    init_registry(exp_dir_path)

    if prepare_smoke_experiment:
        smoke_dir = write_smoke_experiment(exp_dir_path)
        print(f"  [smoke] Wrote deployment smoke experiment: {smoke_dir}", flush=True)

    if deploy_only:
        if deploy_target == "none":
            print(
                "Error: --deploy-only requires --deploy-target dry-run or firebase",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"\n{'=' * 60}", flush=True)
        print(
            f"  Experiment {exp_num} / Deployment ({deploy_target}, deploy-only)",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)
        run_deployment_programmatic(
            exp_dir=exp_dir_path,
            project_id=project_id,
            run_id=exp_num,
            deploy_target=deploy_target,
            prolific_mode=prolific_mode,
            n_participants=n_participants,
            collection_owner=collection_owner,
            firebase_project=firebase_project,
            firebase_region=firebase_region,
            backend=backend,
        )
        print(
            f"\nExperiment {exp_num} deployment complete. Outputs: {exp_dir_path}",
            flush=True,
        )
        return

    seeded_models = False
    if exp_num == 1:
        seeded_models = seed_experiment_models_from_project(exp_dir_path, project_id)
        if seeded_models:
            print(
                f"  [seed] Copied project seed models into {exp_dir_path / 'cognitive_models'}",
                flush=True,
            )
            if validate:
                ok, msg = validate_cc_output("1_theory", exp_dir_path)
                if ok:
                    print(f"  [ok] seeded theory: {msg}", flush=True)
                else:
                    print(f"  [error] Seed validation failed: {msg}", file=sys.stderr)
                    sys.exit(1)

    prev_exp_dir = experiment_dir(project_id, exp_num - 1) if exp_num > 1 else None
    if prev_exp_dir and not prev_exp_dir.exists():
        prev_exp_dir = None

    keys_to_run = [agent_filter] if agent_filter else AGENT_KEYS
    if seeded_models and agent_filter is None:
        keys_to_run = [key for key in keys_to_run if key != "1_theory"]

    for agent_key in keys_to_run:
        # No-browser mode never uses the jsPsych experiment, so skip building it
        # in a full run (still runnable explicitly via --agent 3_implement).
        if (
            agent_key == "3_implement"
            and mode == "simulated_participants_nobrowser"
            and not agent_filter
        ):
            print("  [skip] 3_implement not needed in no-browser mode", flush=True)
            continue
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
            inner_loop_iterations=inner_loop_iterations,
            inner_loop_candidates=inner_loop_candidates,
            fit_kwargs=fit_kwargs,
            backend=backend,
            participant_backend=participant_backend,
            participant_model=participant_model,
            prolific_mode=prolific_mode,
            enable_critique=enable_critique,
            n_critique_proposals=n_critique_proposals,
        )
        if agent_key == "3_implement" and deploy_target != "none":
            print(f"\n{'=' * 60}", flush=True)
            print(f"  Experiment {exp_num} / Deployment ({deploy_target})", flush=True)
            print(f"{'=' * 60}", flush=True)
            run_deployment_programmatic(
                exp_dir=exp_dir_path,
                project_id=project_id,
                run_id=exp_num,
                deploy_target=deploy_target,
                prolific_mode=prolific_mode,
                n_participants=n_participants,
                collection_owner=collection_owner,
                firebase_project=firebase_project,
                firebase_region=firebase_region,
                backend=backend,
            )

    if "5_model_loop" in keys_to_run:
        update_registry_from_interpretation(exp_dir_path)

    print(f"\nExperiment {exp_num} complete. Outputs: {exp_dir_path}", flush=True)


@dataclass
class Args:
    """Claude Code agent pipeline for auto-psych."""

    project: str
    """Project ID (e.g. subjective_randomness)."""
    experiment: Optional[int] = None
    """Single experiment number."""
    experiments: Optional[str] = None
    """Experiments to run: N (1..N) or A-B (e.g. 4-6). Overrides --experiment."""
    agent: Optional[
        Literal["1_theory", "2_design", "3_implement", "4_collect", "5_model_loop"]
    ] = None
    """Run only this agent. Omit for full pipeline."""
    mode: Literal[
        "simulated_participants", "simulated_participants_nobrowser", "live"
    ] = "simulated_participants"
    """Data-collection mode."""
    n_participants: int = DEFAULT_N_PARTICIPANTS
    """Number of participants to collect or simulate."""
    ground_truth_model: Optional[str] = None
    """Generate synthetic participant data from this ground-truth model (must be in
    src/pipelines/outer_loop/projects/<project>/ground_truth_models.py). If omitted,
    data is sampled from the theorist's models."""
    validate: bool = False
    """Validate each agent's output; error and stop on failure."""
    resume: bool = False
    """Allow running into an existing experiment directory (skip the exists-check)."""
    inner_loop_iterations: int = 2
    """Max inner-loop model-improvement iterations for 5_model_loop."""
    inner_loop_candidates: int = 3
    """Candidate models per inner-loop iteration for 5_model_loop."""
    coding_agent: Optional[Literal["claude", "opencode"]] = None
    """Coding-agent backend for outer and inner loops. Defaults to the CODING_AGENT
    env var, then 'claude'."""
    participant_backend: Literal["closed", "open"] = "closed"
    """Participant model backend for simulated_participants_nobrowser."""
    closed_model: Optional[str] = None
    """Closed/backend model override for simulated_participants_nobrowser."""
    hf_model: Optional[str] = None
    """Hugging Face model id for open simulated_participants_nobrowser."""
    deploy_target: Literal["none", "dry-run", "firebase"] = "none"
    """Deployment phase after 3_implement."""
    collection_owner: str = os.environ.get("AUTO_PSYCH_COLLECTION_OWNER", "unknown")
    """Human or agent identity responsible for collection bookkeeping."""
    firebase_project: Optional[str] = None
    """Firebase project id. Optional when .firebaserc has projects.default."""
    firebase_region: str = "us-central1"
    """Firebase Functions region for generated rewrites."""
    prolific_mode: Literal["none", "test", "live"] = "none"
    """Create/poll a Prolific study for the deployed experiment."""
    deploy_only: bool = False
    """Run only deployment for an existing experiment; do not spawn a coding agent."""
    prepare_smoke_experiment: bool = False
    """Write a tiny implemented experiment before deploying, useful for smoke tests."""
    draws: int = 2000
    """MCMC posterior draws per chain for inner-loop model fits."""
    tune: int = 2000
    """MCMC tuning (warmup) steps per chain for inner-loop model fits."""
    chains: int = 4
    """MCMC chains for inner-loop model fits."""
    critique: bool = True
    """Run a CriticAL posterior-predictive critique of the incumbent before each
    inner-loop candidate round (the critique feeds the candidate agents)."""
    n_critique_proposals: Optional[int] = None
    """Test statistics the critique agent proposes per round (None ⇒ inner-loop
    default)."""


def main(args: Args) -> None:
    project_id = args.project
    prob_path = outer_project_dir(project_id) / "problem_definition.md"
    if not prob_path.exists():
        print(f"Error: problem definition not found at {prob_path}", file=sys.stderr)
        sys.exit(1)

    if args.ground_truth_model is not None:
        allowed = list(get_ground_truth_models(project_id).keys())
        if args.ground_truth_model not in allowed:
            print(
                f"Error: --ground-truth-model must be one of {allowed}; got {args.ground_truth_model!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Resolve experiment IDs.
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

    participant_model = (
        (args.hf_model or DEFAULT_OPEN_MODEL)
        if args.participant_backend == "open"
        else args.closed_model
    )

    # Resolve the backend once and export it so the programmatic inner loop
    # (which spawns its own agents) inherits the same choice.
    backend = select_backend(args.coding_agent)
    os.environ["CODING_AGENT"] = backend

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}

    print(
        f"Pipeline: project={project_id} experiments={exp_ids} mode={args.mode} "
        f"agent={backend} deploy={args.deploy_target} prolific={args.prolific_mode} "
        f"validate={args.validate}",
        flush=True,
    )
    print(
        f"Inner-loop MCMC: draws={args.draws} tune={args.tune} chains={args.chains}",
        flush=True,
    )
    print(f"Outputs: {outer_data_dir() / project_id}", flush=True)

    for exp_num in exp_ids:
        _run_experiment(
            project_id=project_id,
            exp_num=exp_num,
            mode=args.mode,
            n_participants=args.n_participants,
            validate=args.validate,
            resume=args.resume,
            ground_truth_model=args.ground_truth_model,
            agent_filter=args.agent,
            inner_loop_iterations=args.inner_loop_iterations,
            inner_loop_candidates=args.inner_loop_candidates,
            fit_kwargs=fit_kwargs,
            backend=backend,
            participant_backend=args.participant_backend,
            participant_model=participant_model,
            deploy_target=args.deploy_target,
            collection_owner=args.collection_owner,
            firebase_project=args.firebase_project,
            firebase_region=args.firebase_region,
            prolific_mode=args.prolific_mode,
            deploy_only=args.deploy_only,
            prepare_smoke_experiment=args.prepare_smoke_experiment,
            enable_critique=args.critique,
            n_critique_proposals=args.n_critique_proposals,
        )

    print("\nAll experiments complete.", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
