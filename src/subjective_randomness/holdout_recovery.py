"""Ground-truth holdout recovery through the full agentic loop.

The analysis holds one seed model out as the "ground truth": its fixed-param
PyMC model generates every synthetic response, while the outer+inner loop —
real theory and design agents, real candidate-conjecturing agents, MCMC fits
compared by ELPD-LOO — starts from the *remaining* seed models and tries to
recover the held-out process. After every inner-loop scoring step we ask: how
well does the then-best model predict the ground truth's ``p_left`` on a large
held-out stimulus set (Pearson r and RMSE)?

Layout under ``results_root`` (one run per held-out model)::

    <gt_model>/
        experiment1..N/        # full agentic pipeline output trees
        eval_stimuli.json      # held-out stimulus set (post-run exclusion)
        trajectory.json        # per-step correlation trajectory + leakage audit

The expensive seams (`spawn_cc_agent`, `generate_responses`, `fit_model`, ...)
are imported at module level so tests can monkeypatch them here.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from src.models.pymc_inference import fit_model, make_stim_data, pm_data_inputs
from src.pipelines.outer_loop.eig import annotate as annotate_eig
from src.pipelines.outer_loop.orchestrator import (
    ensure_experiment_dirs,
    init_registry,
    outer_project_dir,
    project_seed_models_dir,
    run_inner_model_loop_programmatic,
    seed_experiment_models_from_project,
    spawn_cc_agent,
    update_registry_from_interpretation,
    validate_cc_output,
    write_context,
)
from src.subjective_randomness.config import resolve_path
from src.subjective_randomness.model_recovery import (
    feature_rows,
    generate_responses,
    p_left_fixed_params,
    resolve_generating_params,
    write_responses_csv,
)
from src.subjective_randomness.recover import pearson_r
from src.subjective_randomness.simulate import load_stimuli
from src.subjective_randomness.stimulus_design import generate_candidate_pool

PROJECT_ID = "subjective_randomness"

# Design EIG defaults — match the 2_design prompt's documented CLI invocation,
# so harness-scored stimuli are equivalent to what the agent would have written.
DESIGN_TOP_N = 20
DESIGN_EIG_SAMPLES = 200
DESIGN_EIG_SEED = 42

TRAJECTORY_COLUMNS = [
    "gt_model",
    "experiment",
    "step",
    "iteration",
    "global_step",
    "best_model",
    "pearson_r",
    "rmse",
    "pearson_r_bma",
    "rmse_bma",
]


def _require_valid(agent_key: str, exp_dir: Path) -> None:
    """Fail loudly if a pipeline stage's output does not validate."""
    ok, msg = validate_cc_output(agent_key, exp_dir)
    if not ok:
        raise RuntimeError(f"{agent_key} output invalid in {exp_dir}: {msg}")


def _has_candidate_pool(exp_dir: Path) -> bool:
    """True when ``design/candidates.json`` exists and parses to a non-empty list."""
    path = exp_dir / "design" / "candidates.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(data, list) and len(data) > 0


def _ensure_design_stimuli(
    exp_dir: Path,
    project_id: str,
    *,
    top_n: int = DESIGN_TOP_N,
    n_samples: int = DESIGN_EIG_SAMPLES,
    seed: int = DESIGN_EIG_SEED,
) -> None:
    """Guarantee ``design/stimuli.json`` exists after the design step.

    The design agent's creative job is the candidate pool
    (``design/candidates.json``); scoring it by EIG into the top-N stimuli is
    deterministic. Agents sometimes background that slow scoring and end their
    turn before ``stimuli.json`` is written, leaving only ``candidates.json`` —
    so when ``stimuli.json`` is missing the harness runs the EIG annotation
    itself rather than stalling on a step it can finish deterministically.
    """
    design_dir = exp_dir / "design"
    stimuli_path = design_dir / "stimuli.json"
    if stimuli_path.exists():
        return
    candidates_path = design_dir / "candidates.json"
    if not candidates_path.exists():
        return  # nothing to score — _require_valid raises the clear missing-file error
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            f"design/candidates.json is not a non-empty list: {candidates_path}"
        )
    featurize_path = outer_project_dir(project_id) / "preprocess.py"
    annotated = annotate_eig(
        candidates,
        exp_dir / "cognitive_models",
        exp_dir / "model_registry.yaml",
        featurize_path=featurize_path if featurize_path.exists() else None,
        n_samples=n_samples,
        seed=seed,
    )
    stimuli_path.write_text(json.dumps(annotated[:top_n], indent=2), encoding="utf-8")
    print(
        f"  [holdout] Design agent left no stimuli.json in {exp_dir.name}; scored "
        f"{len(candidates)} candidate(s) by EIG -> top {min(top_n, len(annotated))}",
        flush=True,
    )


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
    resume: bool = False,
) -> List[Path]:
    """Run the full agentic pipeline for ``n_experiments`` with a held-out GT.

    Experiment 1 is seeded with every project seed model *except* ``gt_model``;
    experiments >= 2 run the real theory agent. Each experiment runs the real
    design agent, collects responses programmatically from the held-out model
    (fixed params, ``pm.do``), and runs the inner model loop (which records the
    per-step ``history.json`` this analysis consumes). Every stage's output is
    validated and any failure raises — a half-run experiment is never silently
    carried forward.

    With ``resume=True`` a stopped run continues: stages whose output already
    validates are skipped, and everything from the first invalid stage on is
    rerun. A partial ``model_loop/`` is wiped before rerunning (it is fully
    regenerable — MCMC fits live in the shared cache — and rerunning over it
    would orphan its admitted candidates from the reseeded manifest).
    """
    run_root = Path(run_root)
    seed_models_dir = Path(seed_models_dir)
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
        allowed_dirs = [exp_dir, outer_project_dir(project_id)]
        if prev_exp_dir is not None:
            allowed_dirs.append(prev_exp_dir)

        # Theory: seeded holdout set in experiment 1, real agent afterwards.
        if not (resume and _stage_done("1_theory", exp_dir)):
            if exp_num == 1:
                seeded = seed_experiment_models_from_project(
                    exp_dir, project_id, exclude=(gt_model,)
                )
                if not seeded:
                    raise RuntimeError(
                        f"Could not seed experiment 1 from "
                        f"{project_seed_models_dir(project_id)}"
                        + (
                            f" — {exp_dir / 'cognitive_models'} already exists but "
                            f"fails validation; delete it to re-seed."
                            if resume
                            else ""
                        )
                    )
            else:
                write_context(exp_dir, "1_theory", project_id, exp_num, prev_exp_dir)
                ok, _ = spawn_cc_agent(
                    "1_theory",
                    exp_dir,
                    allowed_dirs=allowed_dirs,
                    timeout_secs=agent_timeout_sec,
                    backend=backend,
                )
                if not ok:
                    print(
                        f"  [holdout] Warning: 1_theory agent exited without "
                        f"success in {exp_dir}",
                        flush=True,
                    )
            _require_valid("1_theory", exp_dir)

        # Design: the agent proposes a candidate pool (design/candidates.json);
        # turning it into stimuli.json by EIG is deterministic, so the harness
        # finishes that step if the agent backgrounded it. On resume an existing
        # valid candidate pool is reused instead of re-running the costly agent.
        if not (resume and _stage_done("2_design", exp_dir)):
            if not (resume and _has_candidate_pool(exp_dir)):
                write_context(exp_dir, "2_design", project_id, exp_num, prev_exp_dir)
                ok, _ = spawn_cc_agent(
                    "2_design",
                    exp_dir,
                    allowed_dirs=allowed_dirs,
                    timeout_secs=agent_timeout_sec,
                    backend=backend,
                )
                if not ok:
                    print(
                        f"  [holdout] Warning: 2_design agent exited without "
                        f"success in {exp_dir}",
                        flush=True,
                    )
            _ensure_design_stimuli(exp_dir, project_id)
            _require_valid("2_design", exp_dir)

        # Collect: every response comes from the held-out ground truth. The
        # per-experiment seed offset gives repeated stimuli fresh Bernoulli draws.
        if not (resume and _stage_done("4_collect", exp_dir)):
            stimuli = load_stimuli(exp_dir / "design" / "stimuli.json")
            rows = generate_responses(
                gt_model,
                seed_models_dir,
                stimuli,
                gt_params,
                n_participants=n_participants,
                seed=seed + exp_num,
                generator="pymc",
            )
            write_responses_csv(rows, exp_dir / "data" / "responses.csv")
            _require_valid("4_collect", exp_dir)

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
                cache_dir=cache_dir,
                project_id=project_id,
                agent_timeout_sec=agent_timeout_sec,
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
# Held-out eval set (post-run exclusion)
# ─────────────────────────────────────────────


def _unordered_pair(sequence_a: str, sequence_b: str) -> Tuple[str, str]:
    return tuple(sorted((sequence_a, sequence_b)))  # type: ignore[return-value]


def collect_trained_pairs(run_root: Path, n_experiments: int) -> Set[Tuple[str, str]]:
    """Every unordered stimulus pair that appeared in the run's training data."""
    pairs: Set[Tuple[str, str]] = set()
    for exp_num in range(1, n_experiments + 1):
        path = Path(run_root) / f"experiment{exp_num}" / "data" / "responses.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing training responses for experiment {exp_num}: {path}"
            )
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pairs.add(_unordered_pair(row["sequence_a"], row["sequence_b"]))
    return pairs


def build_eval_stimuli(
    run_root: Path,
    *,
    n_experiments: int,
    n_pairs: int,
    lengths: Sequence[int],
    seed: int,
    min_remaining: int = 1,
) -> Dict[str, Any]:
    """Generate the held-out eval pool, excluding every pair used in training.

    The design agents choose training stimuli freely, so holdout is guaranteed
    *after* the run: any pool pair that appeared (in either order) in any of
    the run's ``responses.csv`` files is dropped. The surviving set is fixed and
    shared across every trajectory step.
    """
    pool = generate_candidate_pool(n_pairs, lengths=tuple(lengths), seed=seed)
    trained = collect_trained_pairs(run_root, n_experiments)
    kept = [
        stim
        for stim in pool
        if _unordered_pair(stim["sequence_a"], stim["sequence_b"]) not in trained
    ]
    if len(kept) < min_remaining:
        raise ValueError(
            f"Only {len(kept)} of {n_pairs} eval stimuli remain after excluding "
            f"trained pairs (min_remaining={min_remaining}); enlarge the pool or "
            f"its lengths."
        )
    return {"stimuli": kept, "n_pool": n_pairs, "n_dropped": n_pairs - len(kept)}


# ─────────────────────────────────────────────
# Per-step correlation trajectory
# ─────────────────────────────────────────────


def _participant_ids_in(responses_path: Path) -> Optional[List[int]]:
    """Distinct participant ids in a responses CSV, sorted.

    Returns ``None`` when the responses carry no ``participant_id`` column —
    only models with a participant random effect need it, so absence is fine
    until such a model actually asks for it (then prediction raises loudly).
    """
    with Path(responses_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "participant_id" not in (reader.fieldnames or []):
            return None
        ids = sorted({int(row["participant_id"]) for row in reader})
    return ids or None


def _eval_prediction(
    fitted: Any,
    base_rows: Sequence[Mapping[str, Any]],
    *,
    participant_ids: Optional[Sequence[int]],
) -> np.ndarray:
    """Population-level held-out ``p_left`` for one fitted model.

    Models without a participant random effect predict directly. A model that
    indexes a ``participant_id`` container has no population-level ``p_left`` of
    its own (its ``p_left`` is per participant), so we marginalize the random
    effect: replicate each stimulus across the participants the model was fit on
    and average the per-participant ``p_left`` (over participants *and* posterior
    draws). The result is one population-mean probability per stimulus, directly
    comparable to the non-hierarchical ground truth.
    """
    n_stim = len(base_rows)
    if "participant_id" not in pm_data_inputs(fitted.model):
        stim_data = make_stim_data(fitted.model, list(base_rows))
        return np.asarray(fitted.predict_p_left(stim_data), dtype="float64")
    if not participant_ids:
        raise ValueError(
            "Model indexes a participant_id random effect but the training "
            "responses carry no participant_id to marginalize over."
        )
    rows = [
        {**row, "participant_id": pid}
        for pid in participant_ids
        for row in base_rows
    ]
    stim_data = make_stim_data(fitted.model, rows)
    preds = np.asarray(fitted.predict_p_left(stim_data), dtype="float64")
    return preds.reshape(len(participant_ids), n_stim).mean(axis=0)


def _fitted_seed_baseline(
    seed_models: Sequence[str],
    seed_models_dir: Path,
    responses_path: Path,
    eval_rows: Sequence[Mapping[str, Any]],
    gt_p: np.ndarray,
    *,
    participant_ids: Optional[Sequence[int]],
    cache_dir: Optional[Path],
    fit_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fit each canonical seed model on ``responses_path`` and correlate with GT.

    Predicts held-out ``p_left`` for each seed model and correlates with the
    ground truth. Returns the mean Pearson r / RMSE over the seed models and the
    per-model breakdown — the recovery from *fitting the existing starting
    models*, with no agent-discovered structure.
    """
    per_model: Dict[str, Dict[str, Optional[float]]] = {}
    for name in seed_models:
        fitted = fit_model(
            name,
            seed_models_dir,
            responses_path,
            cache_dir=cache_dir,
            **dict(fit_kwargs),
        )
        pred = _eval_prediction(fitted, eval_rows, participant_ids=participant_ids)
        per_model[name] = {
            "pearson_r": pearson_r(gt_p.tolist(), pred.tolist()),
            "rmse": float(np.sqrt(np.mean((gt_p - pred) ** 2))),
        }
    rs = [v["pearson_r"] for v in per_model.values() if v["pearson_r"] is not None]
    rmses = [v["rmse"] for v in per_model.values()]
    return {
        "pearson_r": float(np.mean(rs)) if rs else None,
        "rmse": float(np.mean(rmses)) if rmses else None,
        "per_model": per_model,
    }


def _bma_prediction(
    weights: Mapping[str, float], predictions: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Posterior-weighted average of per-model ``p_left`` predictions.

    ``weights`` are the model posterior probabilities; ``predictions`` holds one
    ``p_left`` vector per model. The average is over the supplied (nonzero-weight)
    models, renormalized by their total weight so it is an exact convex
    combination even if those weights do not sum to exactly 1.
    """
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError(
            f"Bayesian model average needs positive posterior mass; got weights "
            f"summing to {total} over {sorted(weights)}."
        )
    stacked = np.zeros_like(next(iter(predictions.values())), dtype="float64")
    for name, weight in weights.items():
        stacked += (weight / total) * predictions[name]
    return stacked


def evaluate_trajectory(
    run_root: Path,
    gt_model: str,
    gt_params: Mapping[str, float],
    eval_stimuli: Sequence[Mapping[str, str]],
    *,
    seed_models_dir: Path,
    n_experiments: int,
    cache_dir: Optional[Path],
    fit_kwargs: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Correlate every inner-loop step's models with the ground truth.

    For each experiment's ``history.json`` step we compute predictions of the
    ground truth's held-out ``p_left`` and report the Pearson r / RMSE of each
    against the fixed-param ground truth:

    * ``pearson_r`` / ``rmse`` — the single then-best model.
    * ``pearson_r_bma`` / ``rmse_bma`` — the Bayesian model average, i.e. the
      posterior-weighted mean of every model with nonzero posterior mass.

    Every needed model is refit on that experiment's pooled responses (a cache
    hit when the run shared ``cache_dir``); zero-weight models are skipped.
    """
    run_root = Path(run_root)
    gt_p = p_left_fixed_params(gt_model, seed_models_dir, eval_stimuli, gt_params)
    eval_rows = feature_rows(eval_stimuli)

    rows: List[Dict[str, Any]] = []
    global_step = 0
    for exp_num in range(1, n_experiments + 1):
        loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
        history_path = loop_dir / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"No history.json for experiment {exp_num}: {history_path}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            raise ValueError(f"Empty inner-loop history: {history_path}")
        participant_ids = _participant_ids_in(loop_dir / "responses.csv")

        for entry in history:
            best = entry["best_model"]
            posteriors = entry["posteriors"]
            # The best line needs `best`; the BMA needs every nonzero-weight
            # model. Fit each needed model once and reuse its prediction.
            weights = {m: w for m, w in posteriors.items() if w > 0.0}
            needed = sorted(set(weights) | {best})
            predictions: Dict[str, np.ndarray] = {}
            for name in needed:
                fitted = fit_model(
                    name,
                    loop_dir / "models",
                    loop_dir / "responses.csv",
                    cache_dir=cache_dir,
                    **dict(fit_kwargs),
                )
                try:
                    predictions[name] = _eval_prediction(
                        fitted, eval_rows, participant_ids=participant_ids
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not predict held-out p_left with model {name!r} "
                        f"(experiment {exp_num}, step {entry['step']}): {exc}"
                    ) from exc

            best_pred = predictions[best]
            bma_pred = _bma_prediction(weights, predictions)
            rows.append(
                {
                    "experiment": exp_num,
                    "step": entry["step"],
                    "iteration": entry["iteration"],
                    "global_step": global_step,
                    "best_model": best,
                    "pearson_r": pearson_r(gt_p.tolist(), best_pred.tolist()),
                    "rmse": float(np.sqrt(np.mean((gt_p - best_pred) ** 2))),
                    "pearson_r_bma": pearson_r(gt_p.tolist(), bma_pred.tolist()),
                    "rmse_bma": float(np.sqrt(np.mean((gt_p - bma_pred) ** 2))),
                }
            )
            global_step += 1
    return rows


def seed_baseline_correlation(
    gt_model: str,
    gt_params: Mapping[str, float],
    eval_stimuli: Sequence[Mapping[str, str]],
    *,
    seed_models_dir: Path,
) -> Dict[str, Any]:
    """No-learning baseline: how well the *other* seed models predict the GT.

    For every project seed model except ``gt_model``, compute its fixed
    default-parameter ``p_left`` on the held-out stimuli and correlate it with
    the ground truth's ``p_left`` (the same fixed-param forward pass that
    generated the responses). Returns the per-model correlations and their mean
    — the off-the-shelf alternatives the loop starts from in experiment 1,
    before any fitting or agent-written models. Fails loudly only if *no* other
    seed model yields a defined correlation.
    """
    seed_models_dir = Path(seed_models_dir)
    gt_p = p_left_fixed_params(gt_model, seed_models_dir, eval_stimuli, gt_params)
    defaults = resolve_generating_params(None, seed_models_dir)

    per_model: Dict[str, Optional[float]] = {}
    for name, params in defaults.items():
        if name == gt_model:
            continue
        pred = p_left_fixed_params(name, seed_models_dir, eval_stimuli, params)
        per_model[name] = pearson_r(gt_p.tolist(), pred.tolist())

    defined = [r for r in per_model.values() if r is not None]
    if not defined:
        raise ValueError(
            f"No defined baseline correlation for held-out {gt_model!r}: every "
            f"other seed model gave a constant prediction on the eval stimuli."
        )
    return {"mean_r": float(np.mean(defined)), "per_model": per_model}


def _pool_experiment_responses(run_root: Path, n_experiments: int) -> Path:
    """Concatenate every experiment's inner-loop responses into one CSV.

    All experiments draw from the same ground-truth process (differing only in
    stimuli), so their featurized responses share a schema and pool cleanly.
    Written deterministically to ``run_root/pooled_responses.csv`` so the fit it
    feeds is cache-stable across re-runs. Fails loudly on a missing file or a
    column-schema mismatch.
    """
    run_root = Path(run_root)
    fieldnames: Optional[Sequence[str]] = None
    pooled: List[Dict[str, str]] = []
    for exp_num in range(1, n_experiments + 1):
        path = run_root / f"experiment{exp_num}" / "model_loop" / "responses.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing inner-loop responses for experiment {exp_num}: {path}"
            )
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(
                    f"Response-column mismatch pooling experiment {exp_num}: "
                    f"{reader.fieldnames} != {fieldnames}"
                )
            pooled.extend(reader)
    out_path = run_root / "pooled_responses.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames or []))
        writer.writeheader()
        writer.writerows(pooled)
    return out_path


def fitted_seed_baseline_correlation(
    run_root: Path,
    gt_model: str,
    gt_params: Mapping[str, float],
    eval_stimuli: Sequence[Mapping[str, str]],
    *,
    seed_models_dir: Path,
    n_experiments: int,
    other_seed_models: Sequence[str],
    cache_dir: Optional[Path],
    fit_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fitted-seed baseline: other seed models fit on *all* collected data.

    Pools every experiment's responses, fits each non-GT seed model once on the
    pool, predicts held-out ``p_left``, and correlates with the ground truth.
    Returns the mean Pearson r / RMSE over the seed models, the per-model
    breakdown, and the pooled response count — one flat number per ground truth.
    It isolates the value of agent-discovered structure: same data, same fitting
    machinery, only the starting model forms.
    """
    run_root = Path(run_root)
    gt_p = p_left_fixed_params(gt_model, seed_models_dir, eval_stimuli, gt_params)
    eval_rows = feature_rows(eval_stimuli)
    pooled_path = _pool_experiment_responses(run_root, n_experiments)
    participant_ids = _participant_ids_in(pooled_path)
    n_responses = sum(1 for _ in pooled_path.open(encoding="utf-8")) - 1

    baseline = _fitted_seed_baseline(
        other_seed_models,
        Path(seed_models_dir),
        pooled_path,
        eval_rows,
        gt_p,
        participant_ids=participant_ids,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
    )
    if baseline["pearson_r"] is None:
        raise ValueError(
            f"No defined fitted-seed baseline for held-out {gt_model!r}: every "
            f"other seed model gave a constant prediction on the eval stimuli."
        )
    baseline["mean_r"] = baseline.pop("pearson_r")
    baseline["mean_rmse"] = baseline.pop("rmse")
    baseline["n_responses"] = n_responses
    return baseline


def reevaluate_trajectories(
    result: Mapping[str, Any],
    *,
    seed_models_dir: Path,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    """Recompute every ground truth's trajectory from its finished run tree.

    Reads each ``gt_run``'s on-disk ``run_root`` (its ``history.json`` per
    experiment) and ``eval_stimuli.json``, then recomputes the best-model and
    Bayesian-model-average trajectories plus the default-param and fitted-seed
    baselines through the shared MCMC cache — so a run whose hours-long agentic
    loop already finished can be re-analyzed (e.g. to add a baseline or
    regenerate the figure) without re-running any agents. Returns a new result;
    the input is not mutated.
    """
    seed_models_dir = Path(seed_models_dir)
    n_experiments = int(result["n_experiments"])
    fit_kwargs = dict(result.get("fit_kwargs", {}))

    all_seed_models = set(resolve_generating_params(None, seed_models_dir))
    new_runs: List[Dict[str, Any]] = []
    for gt_run in result["gt_runs"]:
        run_root = Path(gt_run["run_root"])
        eval_stimuli = json.loads(
            (run_root / "eval_stimuli.json").read_text(encoding="utf-8")
        )
        other_seeds = sorted(all_seed_models - {gt_run["gt_model"]})
        trajectory = evaluate_trajectory(
            run_root,
            gt_run["gt_model"],
            gt_run["params"],
            eval_stimuli,
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
        )
        baseline = seed_baseline_correlation(
            gt_run["gt_model"],
            gt_run["params"],
            eval_stimuli,
            seed_models_dir=seed_models_dir,
        )
        fitted_baseline = fitted_seed_baseline_correlation(
            run_root,
            gt_run["gt_model"],
            gt_run["params"],
            eval_stimuli,
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            other_seed_models=other_seeds,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
        )
        new_runs.append(
            {
                **gt_run,
                "trajectory": trajectory,
                "baseline": baseline,
                "fitted_baseline": fitted_baseline,
            }
        )
    return {**result, "gt_runs": new_runs}


# ─────────────────────────────────────────────
# Leakage audit
# ─────────────────────────────────────────────


def _distinctive_param_names(gt_model: str) -> Set[str]:
    """The GT family's parameter names that other families do not share."""
    module = importlib.import_module(
        f"src.subjective_randomness.model_families.{gt_model}"
    )
    return set(module.DEFAULT_PARAMS) - {"beta", "side_bias"}


def leakage_check(
    run_root: Path,
    gt_model: str,
    *,
    seed_models_dir: Path,
    n_experiments: int,
) -> Dict[str, Any]:
    """Audit a run for ground-truth leakage into agent-written models.

    Agents can read the project assets dir, which contains the held-out seed
    model's source. This flags (without enforcing) byte-identical copies,
    mentions of the GT family's distinctive parameter names, and files named
    after the GT model, across every experiment's ``cognitive_models/`` and
    ``model_loop/models/``. Heuristic: a paraphrased reimplementation can evade
    it, so flags are an audit trail, not proof of a clean run.
    """
    run_root = Path(run_root)
    gt_hash = hashlib.sha256(
        (Path(seed_models_dir) / f"{gt_model}.py").read_bytes()
    ).hexdigest()
    distinctive = _distinctive_param_names(gt_model)

    files: List[Dict[str, Any]] = []
    for exp_num in range(1, n_experiments + 1):
        exp_dir = run_root / f"experiment{exp_num}"
        for sub in ("cognitive_models", Path("model_loop") / "models"):
            model_dir = exp_dir / sub
            if not model_dir.is_dir():
                continue
            for path in sorted(model_dir.glob("*.py")):
                source = path.read_text(encoding="utf-8")
                files.append(
                    {
                        "path": str(path.relative_to(run_root)),
                        "identical": hashlib.sha256(path.read_bytes()).hexdigest()
                        == gt_hash,
                        "mentions_gt_params": any(p in source for p in distinctive),
                        "gt_named": path.name == f"{gt_model}.py",
                    }
                )
    return {
        "files": files,
        "any_identical": any(f["identical"] for f in files),
        "any_mention": any(f["mentions_gt_params"] for f in files),
        "any_gt_named": any(f["gt_named"] for f in files),
    }


# ─────────────────────────────────────────────
# Config-driven entry point
# ─────────────────────────────────────────────


def _manifest_model_names(exp_dir: Path) -> List[str]:
    manifest = yaml.safe_load(
        (exp_dir / "cognitive_models" / "models_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    return [m["name"] if isinstance(m, dict) else m for m in manifest["models"]]


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
    seed_override: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    backend_override: Optional[str] = None,
    agent_timeout_override: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run holdout recovery for every configured ground-truth model.

    With ``resume=True``, a ground truth whose ``trajectory.json`` already
    exists is loaded from disk and skipped entirely (after a loud consistency
    check against the config's ``n_experiments``), and incomplete runs continue
    from their first invalid stage instead of refusing the existing directory.

    Config keys:
        project_id        project whose pipeline assets drive the agents
        seed_models_dir   the project's seed-model directory (must match
                          ``project_seed_models_dir(project_id)`` — exp 1 is
                          seeded from the project assets)
        gt_models         null | [names] | {name: params|null}; null params ->
                          the family's DEFAULT_PARAMS
        n_experiments, n_participants, seed
        inner_loop        {max_iterations, candidate_count}
        agent             {timeout_sec, backend}
        eval_pool         {n_pairs, lengths, seed, min_remaining}
        fit               MCMC kwargs (draws/tune/chains/...)
    """
    project_id = config.get("project_id", PROJECT_ID)
    seed_models_dir = resolve_path(config["seed_models_dir"], config_path)
    project_seed_dir = project_seed_models_dir(project_id)
    if seed_models_dir.resolve() != project_seed_dir.resolve():
        raise ValueError(
            f"seed_models_dir ({seed_models_dir}) must be the project's seed "
            f"directory ({project_seed_dir}): experiment 1 is seeded from the "
            f"project assets, so a different generator set would be incoherent."
        )

    gt_params_by_model = resolve_generating_params(
        config.get("gt_models"), seed_models_dir
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

    agent_cfg = dict(config.get("agent", {}))
    agent_timeout_sec = agent_timeout_override or int(agent_cfg.get("timeout_sec", 900))
    backend = backend_override or agent_cfg.get("backend")

    fit_kwargs = {**dict(config.get("fit", {})), **dict(fit_overrides or {})}

    pool_cfg = dict(config.get("eval_pool", {}))
    eval_pool = {
        "n_pairs": int(pool_cfg.get("n_pairs", 500)),
        "lengths": [int(x) for x in pool_cfg.get("lengths", (6, 8))],
        "seed": int(pool_cfg.get("seed", 11)),
        "min_remaining": int(pool_cfg.get("min_remaining", 100)),
    }

    results_root = Path(results_root)
    # Every project seed model — the fitted-seed baseline for each ground truth
    # fits the *other* seed models (all of these except that GT).
    all_seed_models = set(resolve_generating_params(None, seed_models_dir))
    gt_runs: List[Dict[str, Any]] = []
    for gt_model, gt_params in gt_params_by_model.items():
        run_root = results_root / gt_model
        trajectory_path = run_root / "trajectory.json"
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
            resume=resume,
        )

        eval_info = build_eval_stimuli(
            run_root,
            n_experiments=n_experiments,
            n_pairs=eval_pool["n_pairs"],
            lengths=eval_pool["lengths"],
            seed=eval_pool["seed"],
            min_remaining=eval_pool["min_remaining"],
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
        )
        leakage = leakage_check(
            run_root,
            gt_model,
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
        )
        baseline = seed_baseline_correlation(
            gt_model,
            gt_params,
            eval_info["stimuli"],
            seed_models_dir=seed_models_dir,
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
        (run_root / "trajectory.json").write_text(
            json.dumps(gt_run, indent=2), encoding="utf-8"
        )
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
        "gt_runs": gt_runs,
    }


def trajectory_tidy_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per (held-out model, trajectory step), ready for a tidy CSV."""
    rows: List[Dict[str, Any]] = []
    for gt_run in result["gt_runs"]:
        for entry in gt_run["trajectory"]:
            rows.append({"gt_model": gt_run["gt_model"], **entry})
    return rows
