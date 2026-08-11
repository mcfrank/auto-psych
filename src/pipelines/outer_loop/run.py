#!/usr/bin/env python3
"""
Claude Code agent pipeline for auto-psych.

Each agent runs as a full Claude Code instance (read/write files, run bash, multi-step reasoning).
Outputs go to data/outer_loop/<project>/experimentN/ directories.

Usage:
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiments 3
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiments 4-6
  python3 -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --agent 2_design
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro
from pyprojroot import here

# Ensure repo root on path, so "import src..." works when this file is run
# directly and not only as `python -m src.pipelines.outer_loop.run`. This has to
# happen before the src imports below, hence here() rather than the canonical
# src.runtime.config.REPO_ROOT (which resolves the root the same way).
sys.path.insert(0, str(here()))

from src.models.mcmc_defaults import (
    PRODUCTION_CHAINS,
    PRODUCTION_DRAWS,
    PRODUCTION_TUNE,
)
from src.pipelines.inner_loop.run import load_hints_file
from src.pipelines.outer_loop.deployment import write_smoke_experiment
from src.pipelines.outer_loop.orchestrator import (
    carry_forward_cognitive_models,
    ensure_experiment_dirs,
    experiment_dir,
    get_ground_truth_models,
    init_registry,
    outer_data_dir,
    outer_project_dir,
    run_collect_programmatic,
    run_deployment_programmatic,
    run_design_programmatic,
    run_inner_model_loop_programmatic,
    seed_experiment_models_from_project,
    spawn_cc_agent,
    update_registry_from_interpretation,
    validate_cc_output,
    write_context,
)
from src.pipelines.outer_loop.participants import DEFAULT_OPEN_MODEL
from src.runtime.coding_agent import select_backend
from src.runtime.token_usage import (
    format_summary,
    records_marker,
    records_since,
    start_usage_log,
    summarize,
    write_usage_report,
)

# The pipeline stages. There is no theorist agent: experiment 1's model set is
# seeded from the project's seed_models (required), experiments >= 2 carry the
# previous experiment's cognitive_models forward, and new hypotheses enter only
# via the inner loop (5_model_loop). There is no design agent either: 2_design
# is the programmatic exhaustive EIG selection (run_design_programmatic).
AGENT_KEYS = ["2_design", "3_implement", "4_collect", "5_model_loop"]

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
    critique_alpha: Optional[float] = None,
    max_validation_repairs: int = 2,
    candidate_hints: Optional[list] = None,
    novelty_rmse_threshold: Optional[float] = None,
    prune_dse_multiplier: Optional[float] = None,
    prune_weight_floor: Optional[float] = None,
    candidate_parallelism: Optional[int] = None,
) -> None:
    """Run one agent. Raises SystemExit if --validate and output stays invalid.

    Coding-agent stages get a repair loop: a validation failure is fed back to the
    agent so it can fix its output in place, up to ``max_validation_repairs`` extra
    attempts. Only an exhausted budget aborts the run — one fixable mistake no
    longer throws away the whole pipeline. Programmatic stages (collect, model
    loop) validate once and abort on failure (there is no agent to re-prompt).
    """
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
        _validate_or_exit(agent_key, exp_dir, validate)
    elif agent_key == "5_model_loop":
        run_inner_model_loop_programmatic(
            exp_dir,
            max_iterations=inner_loop_iterations,
            candidate_count=inner_loop_candidates,
            fit_kwargs=fit_kwargs,
            enable_critique=enable_critique,
            n_critique_proposals=n_critique_proposals,
            critique_alpha=critique_alpha,
            candidate_hints=candidate_hints,
            novelty_rmse_threshold=novelty_rmse_threshold,
            prune_dse_multiplier=prune_dse_multiplier,
            prune_weight_floor=prune_weight_floor,
            candidate_parallelism=candidate_parallelism,
        )
        _validate_or_exit(agent_key, exp_dir, validate)
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

        repair_feedback: Optional[str] = None
        for attempt in range(max_validation_repairs + 1):
            ok_spawn, _output = spawn_cc_agent(
                agent_key=agent_key,
                exp_dir=exp_dir,
                allowed_dirs=allowed_dirs,
                backend=backend,
                repair_feedback=repair_feedback,
            )
            if not ok_spawn:
                print(
                    f"  [warn] Agent {agent_key} exited with non-zero status",
                    flush=True,
                )
            if not validate:
                return
            ok, msg = validate_cc_output(agent_key, exp_dir)
            if ok:
                print(f"  [ok] {agent_key}: {msg}", flush=True)
                return
            if attempt < max_validation_repairs:
                print(
                    f"  [repair] {agent_key} failed validation "
                    f"(attempt {attempt + 1}/{max_validation_repairs + 1}): {msg}\n"
                    f"  [repair] feeding the error back to the agent to fix in place",
                    flush=True,
                )
                repair_feedback = msg
            else:
                print(
                    f"  [error] {agent_key} still invalid after "
                    f"{max_validation_repairs + 1} attempt(s): {msg}",
                    file=sys.stderr,
                )
                sys.exit(1)


def _validate_or_exit(agent_key: str, exp_dir: Path, validate: bool) -> None:
    """Validate a programmatic stage's output once; abort the run on failure.

    Programmatic stages (collect, model loop) have no agent to re-prompt, so a
    failure is terminal — unlike coding stages, which get a repair loop.
    """
    if not validate:
        return
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
    critique_alpha: Optional[float] = None,
    run_label: Optional[str] = None,
    max_validation_repairs: int = 2,
    candidate_hints: Optional[list] = None,
    novelty_rmse_threshold: Optional[float] = None,
    prune_dse_multiplier: Optional[float] = None,
    prune_weight_floor: Optional[float] = None,
    candidate_parallelism: Optional[int] = None,
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

    # Track every LLM spend (coding agents, participant/steering calls) for this
    # experiment. The report is written in a finally so a failed or aborted run
    # still accounts for the tokens it already used.
    usage_marker = start_usage_log(exp_dir_path / "token_usage.jsonl")
    try:
        _run_experiment_stages(
            project_id=project_id,
            exp_num=exp_num,
            exp_dir_path=exp_dir_path,
            mode=mode,
            n_participants=n_participants,
            validate=validate,
            ground_truth_model=ground_truth_model,
            agent_filter=agent_filter,
            inner_loop_iterations=inner_loop_iterations,
            inner_loop_candidates=inner_loop_candidates,
            fit_kwargs=fit_kwargs,
            backend=backend,
            participant_backend=participant_backend,
            participant_model=participant_model,
            deploy_target=deploy_target,
            collection_owner=collection_owner,
            firebase_project=firebase_project,
            firebase_region=firebase_region,
            prolific_mode=prolific_mode,
            deploy_only=deploy_only,
            prepare_smoke_experiment=prepare_smoke_experiment,
            enable_critique=enable_critique,
            n_critique_proposals=n_critique_proposals,
            critique_alpha=critique_alpha,
            run_label=run_label,
            max_validation_repairs=max_validation_repairs,
            candidate_hints=candidate_hints,
            novelty_rmse_threshold=novelty_rmse_threshold,
            prune_dse_multiplier=prune_dse_multiplier,
            prune_weight_floor=prune_weight_floor,
            candidate_parallelism=candidate_parallelism,
        )
    finally:
        write_usage_report(
            exp_dir_path, usage_marker, heading=f"experiment {exp_num}"
        )


def _run_experiment_stages(
    *,
    project_id: str,
    exp_num: int,
    exp_dir_path: Path,
    mode: str,
    n_participants: int,
    validate: bool,
    ground_truth_model: Optional[str],
    agent_filter: Optional[str],
    inner_loop_iterations: int,
    inner_loop_candidates: int,
    fit_kwargs: Optional[dict],
    backend: Optional[str],
    participant_backend: str,
    participant_model: Optional[str],
    deploy_target: str,
    collection_owner: str,
    firebase_project: Optional[str],
    firebase_region: str,
    prolific_mode: str,
    deploy_only: bool,
    prepare_smoke_experiment: bool,
    enable_critique: bool,
    n_critique_proposals: Optional[int],
    critique_alpha: Optional[float],
    run_label: Optional[str],
    max_validation_repairs: int,
    candidate_hints: Optional[list],
    novelty_rmse_threshold: Optional[float],
    prune_dse_multiplier: Optional[float],
    prune_weight_floor: Optional[float],
    candidate_parallelism: Optional[int],
) -> None:
    """The body of one experiment, from smoke prep through the agent stages."""
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
            run_label=run_label,
        )
        print(
            f"\nExperiment {exp_num} deployment complete. Outputs: {exp_dir_path}",
            flush=True,
        )
        return

    # Establish this experiment's model set (idempotent: an existing valid
    # cognitive_models/ is left alone, so --resume and --agent reruns are safe).
    # There is no theorist agent — experiment 1 REQUIRES project seed models,
    # and experiments >= 2 carry the previous experiment's set forward.
    if exp_num == 1:
        if seed_experiment_models_from_project(exp_dir_path, project_id):
            print(
                f"  [seed] Copied project seed models into {exp_dir_path / 'cognitive_models'}",
                flush=True,
            )
    else:
        # carry_forward raises loudly if the previous experiment never
        # completed (no manifest to carry).
        if carry_forward_cognitive_models(
            experiment_dir(project_id, exp_num - 1), exp_dir_path
        ):
            print(
                f"  [carry-forward] Copied experiment {exp_num - 1}'s cognitive_models "
                f"into {exp_dir_path / 'cognitive_models'}",
                flush=True,
            )
    if validate:
        ok, msg = validate_cc_output("models", exp_dir_path)
        if ok:
            print(f"  [ok] model set: {msg}", flush=True)
        else:
            hint = (
                " (experiment 1 requires project seed models in "
                f"{outer_project_dir(project_id) / 'seed_models'} — there is no "
                "theorist agent to write them)"
                if exp_num == 1
                else ""
            )
            print(
                f"  [error] Model-set validation failed: {msg}{hint}",
                file=sys.stderr,
            )
            sys.exit(1)

    prev_exp_dir = experiment_dir(project_id, exp_num - 1) if exp_num > 1 else None
    if prev_exp_dir and not prev_exp_dir.exists():
        prev_exp_dir = None

    keys_to_run = [agent_filter] if agent_filter else AGENT_KEYS

    for agent_key in keys_to_run:
        # The design stage is programmatic (there is no design agent). It runs
        # after the model set is seeded / carried forward so experiments >= 2
        # see the current models.
        if agent_key == "2_design":
            run_design_programmatic(
                exp_dir_path, project_id, exp_num=exp_num, prev_exp_dir=prev_exp_dir
            )
            if validate:
                ok, msg = validate_cc_output("2_design", exp_dir_path)
                if not ok:
                    print(f"  [error] Exhaustive design invalid: {msg}", file=sys.stderr)
                    sys.exit(1)
            continue
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
            critique_alpha=critique_alpha,
            max_validation_repairs=max_validation_repairs,
            candidate_hints=candidate_hints,
            novelty_rmse_threshold=novelty_rmse_threshold,
            prune_dse_multiplier=prune_dse_multiplier,
            prune_weight_floor=prune_weight_floor,
            candidate_parallelism=candidate_parallelism,
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
                run_label=run_label,
            )
            if prolific_mode == "test":
                # Test mode creates a DRAFT study (not published) and deploys the
                # experiment, then STOPS — you preview it yourself (open the study
                # in Prolific, or the experiment URL with a made-up PROLIFIC_PID).
                # There is nothing to collect/model until you do.
                print(
                    "\n  [test] Draft Prolific study created (NOT published) and experiment "
                    "deployed. Preview it from your Prolific dashboard, or open the experiment "
                    "URL with a made-up PROLIFIC_PID. Stopping before collection/modeling.",
                    flush=True,
                )
                print(f"\nExperiment {exp_num} (test) complete. Outputs: {exp_dir_path}", flush=True)
                return

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
        Literal["2_design", "3_implement", "4_collect", "5_model_loop"]
    ] = None
    """Run only this stage. Omit for full pipeline. 2_design is the programmatic
    exhaustive EIG selection (no coding agent)."""
    run_label: Optional[str] = None
    """Label that isolates this run's Firebase hosting paths (/e{N}-{label}/) so
    parallel runs don't deploy to the same URLs. Defaults to a unique auto token."""
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
    """Validate each agent's output. A failed coding stage is fed its error and
    re-run to fix it in place (up to --max-validation-repairs times); the run
    aborts only if the output is still invalid after that."""
    max_validation_repairs: int = 2
    """How many extra times a coding stage may be re-run with its validation error
    as feedback before the run aborts. 0 = fail on the first invalid output."""
    resume: bool = False
    """Allow running into an existing experiment directory (skip the exists-check)."""
    inner_loop_iterations: int = 2
    """Max inner-loop model-improvement iterations for 5_model_loop."""
    inner_loop_candidates: int = 3
    """Candidate models per inner-loop iteration for 5_model_loop."""
    coding_agent: Optional[Literal["claude", "opencode"]] = None
    """Coding-agent backend for outer and inner loops. Defaults to the CODING_AGENT
    env var, then 'opencode'. Pass 'claude' for Claude Code."""
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
    draws: int = PRODUCTION_DRAWS
    """MCMC posterior draws per chain for inner-loop model fits
    (src.models.mcmc_defaults)."""
    tune: int = PRODUCTION_TUNE
    """MCMC tuning (warmup) steps per chain for inner-loop model fits."""
    chains: int = PRODUCTION_CHAINS
    """MCMC chains for inner-loop model fits."""
    critique: bool = True
    """Run a CriticAL posterior-predictive critique of the incumbent before each
    inner-loop candidate round (the critique feeds the candidate agents)."""
    n_critique_proposals: Optional[int] = None
    """Test statistics the critique agent proposes per round (None ⇒ inner-loop
    default)."""
    critique_alpha: Optional[float] = None
    """Raw p-value threshold for flagging a critique discrepancy (a Benjamini-
    Hochberg FDR-adjusted q is reported alongside it). None ⇒ inner-loop default
    of 0.05; lower = stricter."""
    hints_file: Optional[Path] = None
    """YAML list of exploration hints cycled across a round's candidates
    (None ⇒ the inner loop's built-in lens battery)."""
    novelty_rmse_threshold: Optional[float] = None
    """Reject a candidate whose p_left is within this RMSE of an admitted
    model's (None ⇒ inner-loop default 0.02; 0 disables the gate)."""
    prune_dse_multiplier: Optional[float] = None
    """Prune agent models with elpd_diff > multiplier*dse AND negligible
    stacking weight after each scoring pass (None ⇒ inner-loop default 2.0;
    0 disables pruning)."""
    prune_weight_floor: Optional[float] = None
    """Stacking-weight floor for pruning (None ⇒ inner-loop default 0.01)."""
    candidate_parallelism: Optional[int] = None
    """Concurrent candidate agents per inner-loop round (None ⇒ all of a
    round's candidates at once; 1 = sequential)."""
    confirm_live_recruitment: bool = False
    """Required alongside --prolific-mode live: going live recruits and PAYS
    real participants, so it must be a second, explicit act (mirrors
    smoke_firebase_deploy's --confirm-production)."""


def main(args: Args) -> None:
    # Money gate first — before any filesystem or network work. A YAML flag
    # alone must never be able to start real recruitment.
    if args.prolific_mode == "live" and not args.confirm_live_recruitment:
        print(
            "Error: --prolific-mode live recruits and PAYS real participants. "
            "Pass --confirm-live-recruitment to confirm this is intentional "
            "(config launchers: set `confirm_live_recruitment: true`).",
            file=sys.stderr,
        )
        sys.exit(1)

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

    # Load exploration hints once; a broken hints file must fail before any
    # experiment work starts, not mid-run at the first candidate round.
    candidate_hints = load_hints_file(args.hints_file) if args.hints_file else None

    # One label per invocation, shared across this run's experiments, so parallel
    # runs deploy to distinct /e{N}-{label}/ hosting paths instead of colliding.
    run_label = args.run_label or uuid.uuid4().hex[:8]

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

    # Test mode produces a draft to preview and stops before collection, so a
    # later experiment would have no prior-experiment data to build on. Run only
    # the first.
    if args.prolific_mode == "test" and len(exp_ids) > 1:
        print(
            f"  [test] prolific_mode=test runs only experiment {exp_ids[0]} (a draft to "
            f"preview); skipping {exp_ids[1:]}.",
            file=sys.stderr,
            flush=True,
        )
        exp_ids = exp_ids[:1]

    run_usage_marker = records_marker()
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
            critique_alpha=args.critique_alpha,
            run_label=run_label,
            max_validation_repairs=args.max_validation_repairs,
            candidate_hints=candidate_hints,
            novelty_rmse_threshold=args.novelty_rmse_threshold,
            prune_dse_multiplier=args.prune_dse_multiplier,
            prune_weight_floor=args.prune_weight_floor,
            candidate_parallelism=args.candidate_parallelism,
        )

    print("\nAll experiments complete.", flush=True)
    run_summary = summarize(records_since(run_usage_marker))
    print(
        format_summary(run_summary, f"run total ({len(exp_ids)} experiment(s))"),
        flush=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
