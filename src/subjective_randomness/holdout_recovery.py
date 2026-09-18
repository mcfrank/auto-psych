"""Ground-truth holdout recovery through the full agentic loop.

The analysis holds one seed model out as the "ground truth": its fixed-param
PyMC model generates every synthetic response, while the outer+inner loop —
the programmatic exhaustive EIG design, real candidate-conjecturing agents,
MCMC fits compared by ELPD-LOO — starts from the *remaining* seed models and
tries to recover the held-out process. After every inner-loop scoring step we
ask: how well does the then-best model predict the ground truth's ``p_left``
on a large held-out stimulus set (Pearson r and RMSE)?

Layout under ``results_root`` (one run per held-out model)::

    <gt_model>/
        experiment1..N/        # full agentic pipeline output trees
        eval_stimuli.json      # held-out stimulus set (post-run exclusion)
        trajectory.json        # per-step correlation trajectory + leakage audit

The expensive seams (`run_design_programmatic`, `generate_responses`,
`fit_model`, ...) are imported at module level so tests can monkeypatch them
here.

Implementation is split across submodules:

* ``holdout_data`` — response generation and data-prep helpers
* ``holdout_eval`` — eval-pool construction and trajectory evaluation
* ``leakage_audit`` — ground-truth leakage audit

This module contains experiment orchestration and config-driven entry points,
and re-exports every public name for backward compatibility.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.models.model_manifest import read_manifest_names
from src.pipelines.outer_loop.columns import write_responses_csv
from src.pipelines.outer_loop.orchestrator import (
    carry_forward_cognitive_models,
    ensure_experiment_dirs,
    init_registry,
    project_seed_models_dir,
    run_design_programmatic,
    run_inner_model_loop_programmatic,
    seed_experiment_models_from_project,
    update_registry_from_interpretation,
    validate_cc_output,
)
from src.runtime.config import REPO_ROOT
from src.runtime.token_usage import start_usage_log, write_usage_report
from src.subjective_randomness.config import resolve_path
from src.subjective_randomness.simulate import load_stimuli

# ─────────────────────────────────────────────
# Re-exports from holdout_data (Seam A)
# ─────────────────────────────────────────────
from src.subjective_randomness.holdout_data import (  # noqa: F401
    GENERATING_MODEL_COLUMN,
    PROJECT_ID,
    RAW_RESPONSE_COLUMNS,
    _default_params_from_file,
    _family_default_params,
    _raw_eval_rows,
    _require_exact_params,
    _require_no_generating_model_column,
    generate_responses,
    p_left_fixed_params,
    resolve_generating_params,
    seed_exclusion,
    seed_model_names,
    strip_generating_model,
    strip_to_raw_columns,
    validate_raw_pool_models,
)

# ─────────────────────────────────────────────
# Re-exports from holdout_eval (Seams C + D)
# ─────────────────────────────────────────────
from src.subjective_randomness.holdout_eval import (  # noqa: F401
    TRAJECTORY_COLUMNS,
    _bma_prediction,
    _eval_prediction,
    _fitted_seed_baseline,
    _participant_ids_in,
    _pool_experiment_responses,
    _resolve_model_dir,
    _unordered_pair,
    build_eval_stimuli,
    collect_trained_pairs,
    evaluate_trajectory,
    fitted_seed_baseline_correlation,
    reevaluate_trajectories,
    seed_baseline_correlation,
)

# ─────────────────────────────────────────────
# Re-exports from leakage_audit (Seam E)
# ─────────────────────────────────────────────
from src.subjective_randomness.leakage_audit import (  # noqa: F401
    _PM_DATA_COLUMN,
    _RESULTS_DIR_NAME,
    _csv_header_columns,
    _csvs_naming_generating_model,
    _distinctive_param_names,
    _manifests_naming_gt,
    leakage_check,
)

# ─────────────────────────────────────────────
# Patchable seams — imported at module level so tests can monkeypatch them
# on this module. Functions that stay in this file (run_holdout_experiments,
# _run_holdout_recovery_resolved) look them up from this module's globals.
# ─────────────────────────────────────────────
from src.models.pymc_inference import (  # noqa: F401
    fit_model,
    load_pymc_model,
    make_stim_data,
    pm_data_inputs,
)


# ─────────────────────────────────────────────
# Stage validation helpers (used only by run_holdout_experiments)
# ─────────────────────────────────────────────


def _require_valid(agent_key: str, exp_dir: Path) -> None:
    """Fail loudly if a pipeline stage's output does not validate."""
    ok, msg = validate_cc_output(agent_key, exp_dir)
    if not ok:
        raise RuntimeError(f"{agent_key} output invalid in {exp_dir}: {msg}")


def _stage_done(agent_key: str, exp_dir: Path) -> bool:
    """True when a stage's existing output already passes its validator.

    Stages run in strict order and the harness stops at the first invalid one,
    so a stopped run leaves a prefix of valid stages — resume skips exactly
    that prefix and reruns everything from the first invalid stage on.
    """
    ok, _ = validate_cc_output(agent_key, exp_dir)
    if ok:
        print(
            f"  [holdout] Resume: {agent_key} already valid in {exp_dir.name} — skipping",
            flush=True,
        )
    return ok


# ─────────────────────────────────────────────
# Agentic experiment sequence (one held-out model)
# ─────────────────────────────────────────────


def run_holdout_experiments(
    gt_model: str,
    gt_params: Mapping[str, float],
    run_root: Path,
    *,
    seed_models_dir: Path,
    n_experiments: int,
    n_participants: int,
    inner_loop_iterations: int,
    candidate_count: int,
    fit_kwargs: Mapping[str, Any],
    project_id: str = PROJECT_ID,
    cache_dir: Optional[Path] = None,
    seed: int = 0,
    agent_timeout_sec: int = 900,
    backend: Optional[str] = None,
    agent_model: Optional[str] = None,
    resume: bool = False,
    gt_models_dir: Optional[Path] = None,
    design_n_eig: int = 32,
    design_n_random: int = 0,
    pool_models_dir: Optional[Path] = None,
    agent_root: Optional[Path] = None,
) -> List[Path]:
    """Run the full agentic pipeline for ``n_experiments`` with a held-out GT.

    Experiment 1 is seeded with every live-pool seed model *except* ``gt_model``
    (when the GT is in the pool at all — an old-registry or impossible GT simply
    is not); experiments >= 2 carry the previous experiment's cognitive_models
    forward, exactly as the live pipeline does (there is no theorist agent).
    Each experiment runs the programmatic exhaustive EIG design, collects
    responses programmatically from the held-out model (fixed params,
    ``pm.do``), and runs the inner model loop (which records the per-step
    ``history.json`` this analysis consumes). Every stage's output is validated
    and any failure raises — a half-run experiment is never silently carried
    forward.

    The responses are written without the generator's name
    (``strip_generating_model``): ``data/responses.csv`` and everything derived
    from it are read by the candidate and critique agents, and the held-out
    model's identity must not reach them.

    With ``resume=True`` a stopped run continues: stages whose output already
    validates are skipped, and everything from the first invalid stage on is
    rerun. A partial ``model_loop/`` is wiped before rerunning (it is fully
    regenerable — MCMC fits live in the shared cache — and rerunning over it
    would orphan its admitted candidates from the reseeded manifest).
    """
    run_root = Path(run_root)
    seed_models_dir = Path(seed_models_dir)
    # seed_models_dir is the GT/baseline registry; the agent seed pool always
    # comes from the live project assets (project_seed_models_dir). The
    # ground-truth generator may live elsewhere (e.g. the impossible-models dir).
    gt_models_dir = (
        Path(gt_models_dir) if gt_models_dir is not None else seed_models_dir
    )
    # Hold the GT out of experiment 1's seed pool whenever it IS in the pool's
    # own manifest. An active seed used as GT is excluded; a GT absent from the
    # pool (e.g. an impossible model) needs no exclusion. Membership by name.
    # Must read the SAME directory that seeding below reads, or the membership
    # test and the seeding disagree (the array scrubs the held-out entry from
    # the pool, so reading the default pool would mis-detect).
    seed_exclude = seed_exclusion(
        gt_model, pool_models_dir or project_seed_models_dir(project_id)
    )
    exp_dirs: List[Path] = []

    for exp_num in range(1, n_experiments + 1):
        exp_dir = run_root / f"experiment{exp_num}"
        if exp_dir.exists() and not resume:
            raise FileExistsError(
                f"Experiment directory already exists: {exp_dir} — pass "
                f"resume=True (CLI: --resume) to continue a stopped run."
            )
        ensure_experiment_dirs(exp_dir)
        init_registry(exp_dir)
        prev_exp_dir = run_root / f"experiment{exp_num - 1}" if exp_num > 1 else None

        # Model set: seeded pool in experiment 1; experiments >= 2 carry the
        # previous experiment's cognitive_models forward. There is no theorist
        # agent — new hypotheses enter only via the inner loop, exactly as in
        # the live pipeline this harness validates.
        if not (resume and _stage_done("models", exp_dir)):
            if exp_num == 1:
                seeded = seed_experiment_models_from_project(
                    exp_dir, project_id, exclude=seed_exclude,
                    seed_dir=pool_models_dir,
                )
                if not seeded:
                    raise RuntimeError(
                        f"Could not seed experiment 1 from "
                        f"{pool_models_dir or project_seed_models_dir(project_id)}"
                        + (
                            f" — {exp_dir / 'cognitive_models'} already exists but "
                            f"fails validation; delete it to re-seed."
                            if resume
                            else ""
                        )
                    )
            else:
                carry_forward_cognitive_models(prev_exp_dir, exp_dir)
            _require_valid("models", exp_dir)

        # Design: the SAME programmatic exhaustive EIG selection as the live
        # pipeline (no design agent). Experiments >= 2 design from the previous
        # experiment's posterior (models fit on its responses, registry weights).
        if not (resume and _stage_done("2_design", exp_dir)):
            run_design_programmatic(
                exp_dir, project_id, exp_num=exp_num, prev_exp_dir=prev_exp_dir,
                k=design_n_eig, n_random=design_n_random,
            )
            _require_valid("2_design", exp_dir)

        # Collect: every response comes from the held-out ground truth. The
        # per-experiment seed offset gives repeated stimuli fresh Bernoulli draws.
        if not (resume and _stage_done("4_collect", exp_dir)):
            stimuli = load_stimuli(exp_dir / "design" / "stimuli.json")
            rows = generate_responses(
                gt_model,
                gt_models_dir,
                stimuli,
                gt_params,
                n_participants=n_participants,
                seed=seed + exp_num,
            )
            agent_rows = strip_generating_model(rows)
            write_responses_csv(agent_rows, exp_dir / "data" / "responses.csv")
            _require_valid("4_collect", exp_dir)
        # Checked on every path (fresh or resumed): a responses file that names
        # its generator must never feed the inner loop.
        _require_no_generating_model_column(exp_dir / "data" / "responses.csv")

        # Inner loop: fits + agent-conjectured candidates over pooled responses.
        history_path = exp_dir / "model_loop" / "history.json"
        if not (
            resume and _stage_done("5_model_loop", exp_dir) and history_path.exists()
        ):
            if resume and (exp_dir / "model_loop").exists():
                shutil.rmtree(exp_dir / "model_loop")
            run_inner_model_loop_programmatic(
                exp_dir,
                max_iterations=inner_loop_iterations,
                candidate_count=candidate_count,
                fit_kwargs=dict(fit_kwargs),
                backend=backend,
                agent_model=agent_model,
                cache_dir=cache_dir,
                project_id=project_id,
                agent_timeout_sec=agent_timeout_sec,
                agent_root=agent_root,
            )
            update_registry_from_interpretation(exp_dir)
            _require_valid("5_model_loop", exp_dir)
            if not history_path.exists():
                raise RuntimeError(
                    f"Inner loop wrote no history.json in {exp_dir / 'model_loop'} "
                    f"— the trajectory cannot be evaluated."
                )
        exp_dirs.append(exp_dir)

    return exp_dirs


# ─────────────────────────────────────────────
# Config-driven entry point
# ─────────────────────────────────────────────


def _manifest_model_names(exp_dir: Path) -> List[str]:
    return read_manifest_names(exp_dir / "cognitive_models")


def run_holdout_recovery_from_config(
    config: Mapping[str, Any],
    config_path: Path,
    results_root: Path,
    *,
    gt_model_override: Optional[str] = None,
    n_experiments_override: Optional[int] = None,
    n_participants_override: Optional[int] = None,
    inner_loop_overrides: Optional[Mapping[str, int]] = None,
    fit_overrides: Optional[Mapping[str, Any]] = None,
    design_overrides: Optional[Mapping[str, int]] = None,
    seed_override: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    backend_override: Optional[str] = None,
    agent_model_override: Optional[str] = None,
    agent_timeout_override: Optional[int] = None,
    resume: bool = False,
    gt_models_dir: Optional[Path] = None,
    gt_family_dir: Optional[Path] = None,
    summary_root: Optional[Path] = None,
    gt_params_by_model_override: Optional[Mapping[str, Mapping[str, float]]] = None,
    agent_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run holdout recovery for every configured ground-truth model.

    With ``resume=True``, a ground truth whose ``trajectory.json`` already
    exists is loaded from disk and skipped entirely (after a loud consistency
    check against the config's ``n_experiments``), and incomplete runs continue
    from their first invalid stage instead of refusing the existing directory.

    Config keys:
        project_id        project whose pipeline assets drive the agents
        seed_models_dir   the GT/baseline registry: models with pure-Python
                          family twins used to generate ground-truth data and
                          fixed-param baselines. Experiment 1's agent seed pool
                          always comes from ``project_seed_models_dir(project_id)``,
                          which mirrors the registry manifest but also keeps
                          superseded models on disk (so a superseded GT can be
                          generated without ever entering the pool).
        gt_models         null | [names] | {name: params|null}; null params ->
                          the family's DEFAULT_PARAMS
        n_experiments, n_participants, seed
        inner_loop        {max_iterations, candidate_count}
        agent             {timeout_sec, backend}
        eval_pool         {n_pairs, lengths, seed, min_remaining}
        fit               MCMC kwargs (draws/tune/chains/...)

    ``gt_models_dir`` (default: ``seed_models_dir``) is where the ground-truth
    models live; the impossible-theory variant points it at a separate directory
    so the agent seed pool stays the normal seed models. When
    ``gt_params_by_model_override`` is given, the per-model fixed params are
    taken from it verbatim instead of resolving the config's ``gt_models`` spec
    (impossible models have no ``model_families`` defaults to fall back on).
    """
    project_id = config.get("project_id", PROJECT_ID)
    # seed_models_dir is the GT/baseline registry and is allowed to differ from
    # the live project seed pool (see docstring). Its adequacy is validated
    # eagerly just below: resolving the configured gt_models imports each GT's
    # pure-Python family twin and raises loudly if one is missing.
    seed_models_dir = resolve_path(config["seed_models_dir"], config_path)
    gt_models_dir = (
        Path(gt_models_dir) if gt_models_dir is not None else seed_models_dir
    )

    if gt_params_by_model_override is not None:
        gt_params_by_model = {
            name: dict(params) for name, params in gt_params_by_model_override.items()
        }
    else:
        gt_params_by_model = resolve_generating_params(
            config.get("gt_models"), seed_models_dir, gt_family_dir
        )
    if gt_model_override is not None:
        if gt_model_override not in gt_params_by_model:
            raise ValueError(
                f"--gt-model {gt_model_override!r} is not among the configured "
                f"gt_models {sorted(gt_params_by_model)}"
            )
        gt_params_by_model = {gt_model_override: gt_params_by_model[gt_model_override]}

    n_experiments = (
        n_experiments_override
        if n_experiments_override is not None
        else int(config.get("n_experiments", 3))
    )
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}.")
    n_participants = (
        n_participants_override
        if n_participants_override is not None
        else int(config.get("n_participants", 30))
    )
    if n_participants < 1:
        raise ValueError(f"n_participants must be >= 1, got {n_participants}.")
    seed = seed_override if seed_override is not None else int(config.get("seed", 0))

    inner_cfg = {**dict(config.get("inner_loop", {})), **dict(inner_loop_overrides or {})}
    inner_loop_iterations = int(inner_cfg.get("max_iterations", 2))
    candidate_count = int(inner_cfg.get("candidate_count", 3))
    # Design split: n_eig stimuli chosen by EIG + n_random for coverage. The
    # random half is a single fixed sample per experiment (shown to every
    # participant); ablations set n_eig=0 (all random) or n_random=0 (all EIG).
    design_cfg = {**dict(config.get("design", {})), **dict(design_overrides or {})}
    pool_models_dir = (
        resolve_path(config["pool_models_dir"]) if config.get("pool_models_dir") else None
    )
    design_n_eig = int(design_cfg.get("n_eig", 32))
    design_n_random = int(design_cfg.get("n_random", 0))

    agent_cfg = dict(config.get("agent", {}))
    agent_timeout_sec = agent_timeout_override or int(agent_cfg.get("timeout_sec", 900))
    backend = backend_override or agent_cfg.get("backend")
    agent_model = agent_model_override or agent_cfg.get("model")

    fit_kwargs = {**dict(config.get("fit", {})), **dict(fit_overrides or {})}

    pool_cfg = dict(config.get("eval_pool", {}))
    predict_max_draws_cfg = pool_cfg.get("predict_max_draws")
    eval_pool = {
        "n_pairs": int(pool_cfg.get("n_pairs", 500)),
        "lengths": [int(x) for x in pool_cfg.get("lengths", (6, 8))],
        "seed": int(pool_cfg.get("seed", 11)),
        "min_remaining": int(pool_cfg.get("min_remaining", 100)),
        # Exhaustive: use every distinct same-length pair, not an n_pairs sample.
        # Defaults to TRUE so seed- and impossible-holdout results use the same
        # pool. predict_max_draws thins the posterior to bound the array size.
        "exhaustive": bool(pool_cfg.get("exhaustive", True)),
        "predict_max_draws": (
            int(predict_max_draws_cfg) if predict_max_draws_cfg is not None else None
        ),
    }

    results_root = Path(results_root)
    # Track every LLM spend (the inner-loop agents) across the whole recovery
    # run. The report is written in a finally so an aborted run still
    # accounts for the tokens it already used.
    usage_marker = start_usage_log(results_root / "token_usage.jsonl")
    try:
        return _run_holdout_recovery_resolved(
            gt_params_by_model,
            results_root,
            seed_models_dir=seed_models_dir,
            gt_models_dir=gt_models_dir,
            gt_family_dir=gt_family_dir,
            summary_root=summary_root,
            project_id=project_id,
            n_experiments=n_experiments,
            n_participants=n_participants,
            inner_loop_iterations=inner_loop_iterations,
            candidate_count=candidate_count,
            fit_kwargs=fit_kwargs,
            eval_pool=eval_pool,
            seed=seed,
            agent_timeout_sec=agent_timeout_sec,
            backend=backend,
            agent_model=agent_model,
            cache_dir=cache_dir,
            resume=resume,
            design_n_eig=design_n_eig,
            design_n_random=design_n_random,
            pool_models_dir=pool_models_dir,
            agent_root=agent_root,
        )
    finally:
        write_usage_report(results_root, usage_marker, heading="holdout recovery")


def _run_holdout_recovery_resolved(
    gt_params_by_model: Dict[str, Dict[str, float]],
    results_root: Path,
    *,
    seed_models_dir: Path,
    gt_models_dir: Path,
    gt_family_dir: Optional[Path] = None,
    summary_root: Optional[Path] = None,
    project_id: str,
    n_experiments: int,
    n_participants: int,
    inner_loop_iterations: int,
    candidate_count: int,
    fit_kwargs: Dict[str, Any],
    eval_pool: Dict[str, Any],
    seed: int,
    agent_timeout_sec: int,
    backend: Optional[str],
    agent_model: Optional[str],
    cache_dir: Optional[Path],
    resume: bool,
    design_n_eig: int = 32,
    design_n_random: int = 0,
    pool_models_dir: Optional[Path] = None,
    agent_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """The recovery loop proper, after all config resolution and validation."""
    # Every project seed model — the fitted-seed baseline for each ground truth
    # fits the *other* seed models (all of these except that GT).
    # Names only — the fitted-seed baseline fits these by MCMC, so no
    # pure-Python family twin (and no default params) is required here.
    all_seed_models = set(seed_model_names(seed_models_dir))
    gt_runs: List[Dict[str, Any]] = []
    for gt_model, gt_params in gt_params_by_model.items():
        run_root = results_root / gt_model
        # The trajectory summary embeds the GT's true params, so it is written
        # OUTSIDE the agent's run tree (summary_root) rather than into
        # results_root/<gt>/, which lives in the agent's cwd. Defaults to
        # run_root when no summary_root is given (non-holdout callers / tests).
        summary_dir = (summary_root / gt_model) if summary_root else run_root
        trajectory_path = summary_dir / "trajectory.json"
        if resume and trajectory_path.exists():
            gt_run = json.loads(trajectory_path.read_text(encoding="utf-8"))
            n_recorded = len(gt_run.get("experiments", []))
            if n_recorded != n_experiments:
                raise ValueError(
                    f"{trajectory_path} records {n_recorded} experiment(s) but the "
                    f"config expects {n_experiments}; delete {run_root} to re-run "
                    f"this ground truth, or align n_experiments."
                )
            print(
                f"[holdout] {gt_model}: already complete ({trajectory_path}) — "
                f"skipping",
                flush=True,
            )
            gt_runs.append(gt_run)
            continue
        print(f"[holdout] Ground truth: {gt_model} -> {run_root}", flush=True)
        exp_dirs = run_holdout_experiments(
            gt_model,
            gt_params,
            run_root,
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            n_participants=n_participants,
            inner_loop_iterations=inner_loop_iterations,
            candidate_count=candidate_count,
            fit_kwargs=fit_kwargs,
            project_id=project_id,
            cache_dir=cache_dir,
            seed=seed,
            agent_timeout_sec=agent_timeout_sec,
            backend=backend,
            agent_model=agent_model,
            resume=resume,
            gt_models_dir=gt_models_dir,
            design_n_eig=design_n_eig,
            design_n_random=design_n_random,
            pool_models_dir=pool_models_dir,
            agent_root=agent_root,
        )

        eval_info = build_eval_stimuli(
            run_root,
            n_experiments=n_experiments,
            n_pairs=eval_pool["n_pairs"],
            lengths=eval_pool["lengths"],
            seed=eval_pool["seed"],
            min_remaining=eval_pool["min_remaining"],
            exhaustive=eval_pool["exhaustive"],
        )
        (run_root / "eval_stimuli.json").write_text(
            json.dumps(eval_info["stimuli"], indent=2), encoding="utf-8"
        )

        other_seeds = sorted(all_seed_models - {gt_model})
        trajectory = evaluate_trajectory(
            run_root,
            gt_model,
            gt_params,
            eval_info["stimuli"],
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
            gt_models_dir=gt_models_dir,
            predict_max_draws=eval_pool["predict_max_draws"],
        )
        leakage = leakage_check(
            run_root,
            gt_model,
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            gt_models_dir=gt_models_dir,
            gt_family_dir=gt_family_dir,
            # The agents' checkout is the scrubbed agent tree (when agent_root
            # is set), or the tree this process runs from (when harness and
            # agents share a copy). This is where the scrubbed manifests live.
            checkout_root=agent_root if agent_root is not None else REPO_ROOT,
        )
        baseline = seed_baseline_correlation(
            gt_model,
            gt_params,
            eval_info["stimuli"],
            seed_models_dir=seed_models_dir,
            gt_models_dir=gt_models_dir,
            gt_family_dir=gt_family_dir,
        )
        fitted_baseline = fitted_seed_baseline_correlation(
            run_root,
            gt_model,
            gt_params,
            eval_info["stimuli"],
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            other_seed_models=other_seeds,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
            gt_models_dir=gt_models_dir,
            predict_max_draws=eval_pool["predict_max_draws"],
        )

        gt_run = {
            "gt_model": gt_model,
            "params": dict(gt_params),
            "run_root": str(run_root),
            "n_eval_stimuli": len(eval_info["stimuli"]),
            "n_eval_dropped": eval_info["n_dropped"],
            "trajectory": trajectory,
            "baseline": baseline,
            "fitted_baseline": fitted_baseline,
            "leakage": leakage,
            "experiments": [
                {
                    "experiment": exp_num,
                    "manifest_models": _manifest_model_names(exp_dir),
                }
                for exp_num, exp_dir in enumerate(exp_dirs, start=1)
            ],
        }
        summary_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(gt_run, indent=2), encoding="utf-8")
        gt_runs.append(gt_run)

    return {
        "project_id": project_id,
        "seed_models_dir": str(seed_models_dir),
        "n_experiments": n_experiments,
        "n_participants": n_participants,
        "inner_loop": {
            "max_iterations": inner_loop_iterations,
            "candidate_count": candidate_count,
        },
        "fit_kwargs": fit_kwargs,
        "seed": seed,
        "eval_pool": eval_pool,
        "metrics_version": 2,
        "gt_runs": gt_runs,
    }


def run_impossible_holdout_recovery_from_config(
    config: Mapping[str, Any],
    config_path: Path,
    results_root: Path,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run holdout recovery with impossible-theory ground-truth generators.

    Identical to ``run_holdout_recovery_from_config`` except the ground truth is
    a deliberately weird model (e.g. "more heads => more random") that lives in a
    separate ``gt_models_dir`` outside the project seed pool. The agentic loop is
    still seeded with the normal seed models, so it is *expected* to fail to
    recover the ground truth (the held-out correlation should stay low). The
    config must provide:

      * ``gt_models_dir`` — directory of impossible PyMC models (it has no
        ``models_manifest.yaml``, so it never enters the agent seed pool).
      * ``gt_models`` — a ``{name: {beta, side_bias}}`` mapping; explicit params
        are required because impossible models have no ``model_families``
        defaults to fall back on.

    All keyword overrides are forwarded to ``run_holdout_recovery_from_config``.
    """
    if "gt_models_dir" not in config:
        raise KeyError(
            "Impossible-theory holdout config must set 'gt_models_dir' (the "
            "directory of impossible ground-truth models, separate from the "
            "project seed pool)."
        )
    gt_models_dir = resolve_path(config["gt_models_dir"], config_path)

    spec = config.get("gt_models")
    if not isinstance(spec, Mapping) or not spec:
        raise ValueError(
            "Impossible-theory holdout config must give 'gt_models' as a "
            "non-empty mapping of {name: {beta, side_bias}}; got "
            f"{type(spec).__name__}."
        )
    gt_params_by_model: Dict[str, Dict[str, float]] = {}
    for name, params in spec.items():
        if not params:
            raise ValueError(
                f"Impossible ground-truth model {name!r} needs explicit params "
                f"{{beta, side_bias}}: there is no model_families default to "
                f"fall back on for an impossible model."
            )
        gt_params_by_model[name] = dict(params)

    return run_holdout_recovery_from_config(
        config,
        config_path,
        results_root,
        gt_models_dir=gt_models_dir,
        gt_params_by_model_override=gt_params_by_model,
        **overrides,
    )


def trajectory_tidy_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per (held-out model, trajectory step), ready for a tidy CSV."""
    rows: List[Dict[str, Any]] = []
    for gt_run in result["gt_runs"]:
        for entry in gt_run["trajectory"]:
            rows.append({"gt_model": gt_run["gt_model"], **entry})
    return rows
