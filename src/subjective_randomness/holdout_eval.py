"""Eval-pool construction and trajectory evaluation for holdout recovery.

Builds the held-out stimulus set (excluding training pairs), computes
per-inner-loop-step correlations between the then-best model and the
ground truth, and provides seed-baseline and fitted-seed-baseline
comparisons.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.models.pymc_inference import fit_model, make_stim_data, pm_data_inputs
from src.subjective_randomness.holdout_data import (
    _raw_eval_rows,
    p_left_fixed_params,
    resolve_generating_params,
    seed_model_names,
)
from src.subjective_randomness.recover import pearson_r
from src.subjective_randomness.recovery_metrics import (
    bias as _bias,
    calibration as _calibration,
    kl_regret as _kl_regret,
    rmse as _rmse,
)
from src.subjective_randomness.stimulus_design import (
    enumerate_all_pairs,
    generate_candidate_pool,
)

TRAJECTORY_COLUMNS = [
    "gt_model",
    "experiment",
    "step",
    "iteration",
    "global_step",
    "best_model",
    "pearson_r",
    "rmse",
    "kl_regret",
    "bias",
    "calib_slope",
    "calib_intercept",
    "pearson_r_bma",
    "rmse_bma",
    "kl_regret_bma",
    "bias_bma",
    "calib_slope_bma",
    "calib_intercept_bma",
]


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
    exhaustive: bool = False,
    extra_excluded_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate the held-out eval pool, excluding every pair used in training.

    The design stage picks training stimuli by EIG, wherever in the pair space
    they fall, so holdout is guaranteed *after* the run: any pool pair that
    appeared (in either order) in any of the run's ``responses.csv`` files is
    dropped. The surviving set is fixed and shared across every trajectory step.

    With ``exhaustive=True`` the pool is every distinct same-length unordered
    pair at the given ``lengths`` rather than an ``n_pairs`` sample, so the correlation is
    measured over the whole stimulus space at those lengths (``n_pairs``/``seed``
    are then unused).

    ``extra_excluded_pairs``, when given, is an additional set of unordered
    pairs to drop (e.g. the training pairs of a second cell in a matched-cell
    comparison).
    """
    pool = (
        enumerate_all_pairs(lengths, same_length_only=True)
        if exhaustive
        else generate_candidate_pool(n_pairs, lengths=tuple(lengths), seed=seed)
    )
    trained = collect_trained_pairs(run_root, n_experiments)
    if extra_excluded_pairs:
        trained = trained | extra_excluded_pairs
    kept = [
        stim
        for stim in pool
        if _unordered_pair(stim["sequence_a"], stim["sequence_b"]) not in trained
    ]
    if len(kept) < min_remaining:
        raise ValueError(
            f"Only {len(kept)} of {len(pool)} eval stimuli remain after excluding "
            f"trained pairs (min_remaining={min_remaining}); enlarge the pool or "
            f"its lengths."
        )
    return {"stimuli": kept, "n_pool": len(pool), "n_dropped": len(pool) - len(kept)}


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
    max_draws: Optional[int] = None,
) -> np.ndarray:
    """Population-level held-out ``p_left`` for one fitted model.

    Models without a participant random effect predict directly. A model that
    indexes a ``participant_id`` container has no population-level ``p_left`` of
    its own (its ``p_left`` is per participant), so we marginalize the random
    effect: replicate each stimulus across the participants the model was fit on
    and average the per-participant ``p_left`` (over participants *and* posterior
    draws). The result is one population-mean probability per stimulus, directly
    comparable to the non-hierarchical ground truth.

    ``max_draws`` thins the posterior for the prediction (see
    ``FittedModel.predict_p_left``); it is only forwarded when set, so callers
    that pass a predictor without that keyword keep working.
    """
    predict_kwargs = {} if max_draws is None else {"max_draws": max_draws}
    n_stim = len(base_rows)
    if "participant_id" not in pm_data_inputs(fitted.model):
        stim_data = make_stim_data(fitted.model, list(base_rows))
        return np.asarray(
            fitted.predict_p_left(stim_data, **predict_kwargs), dtype="float64"
        )
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
    preds = np.asarray(
        fitted.predict_p_left(stim_data, **predict_kwargs), dtype="float64"
    )
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
    predict_max_draws: Optional[int] = None,
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
        pred = _eval_prediction(
            fitted, eval_rows, participant_ids=participant_ids,
            max_draws=predict_max_draws,
        )
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


def _resolve_model_dir(models_dir: Path, name: str) -> Path:
    """Directory to load ``{name}.py`` from: the models dir, or its ``pruned/``
    subdir if the model was pruned after the history step recorded it.

    A trajectory step names models (its best, and every nonzero-weight rival) as
    they stood at that step. A LATER pruning pass can move one of them to
    ``models/pruned/`` (see ``_prune_losers``); the recorded history is still
    valid, so reloading it for trajectory evaluation has to look where the file
    went. This only *redirects* pruned models — when the model is not in
    ``pruned/`` we return the models dir unchanged and let the loader raise its
    own clear error for a genuinely missing file.
    """
    models_dir = Path(models_dir)
    if (models_dir / f"{name}.py").exists():
        return models_dir
    pruned_dir = models_dir / "pruned"
    if (pruned_dir / f"{name}.py").exists():
        return pruned_dir
    return models_dir


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
    gt_models_dir: Optional[Path] = None,
    predict_max_draws: Optional[int] = None,
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
    gt_models_dir = (
        Path(gt_models_dir) if gt_models_dir is not None else Path(seed_models_dir)
    )
    gt_p = p_left_fixed_params(gt_model, gt_models_dir, eval_stimuli, gt_params)
    eval_rows = _raw_eval_rows(eval_stimuli)

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
                    _resolve_model_dir(loop_dir / "models", name),
                    loop_dir / "responses.csv",
                    cache_dir=cache_dir,
                    **dict(fit_kwargs),
                )
                try:
                    predictions[name] = _eval_prediction(
                        fitted, eval_rows, participant_ids=participant_ids,
                        max_draws=predict_max_draws,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not predict held-out p_left with model {name!r} "
                        f"(experiment {exp_num}, step {entry['step']}): {exc}"
                    ) from exc

            best_pred = predictions[best]
            # With no positive-weight models (an empty/degenerate posterior at
            # this step) the BMA has nothing to average — fall back to the best
            # single model's prediction rather than failing the whole run.
            bma_pred = _bma_prediction(weights, predictions) if weights else best_pred

            gt_list = gt_p.tolist()
            best_list = best_pred.tolist()
            bma_list = bma_pred.tolist()
            best_slope, best_intercept = _calibration(gt_list, best_list)
            bma_slope, bma_intercept = _calibration(gt_list, bma_list)
            rows.append(
                {
                    "experiment": exp_num,
                    "step": entry["step"],
                    "iteration": entry["iteration"],
                    "global_step": global_step,
                    "best_model": best,
                    "pearson_r": pearson_r(gt_list, best_list),
                    "rmse": float(np.sqrt(np.mean((gt_p - best_pred) ** 2))),
                    "kl_regret": _kl_regret(gt_list, best_list),
                    "bias": _bias(gt_list, best_list),
                    "calib_slope": best_slope,
                    "calib_intercept": best_intercept,
                    "pearson_r_bma": pearson_r(gt_list, bma_list),
                    "rmse_bma": float(np.sqrt(np.mean((gt_p - bma_pred) ** 2))),
                    "kl_regret_bma": _kl_regret(gt_list, bma_list),
                    "bias_bma": _bias(gt_list, bma_list),
                    "calib_slope_bma": bma_slope,
                    "calib_intercept_bma": bma_intercept,
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
    gt_models_dir: Optional[Path] = None,
    gt_family_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """No-learning baseline: how well the *other* seed models predict the GT.

    For every project seed model except ``gt_model``, compute its fixed
    default-parameter ``p_left`` on the held-out stimuli and correlate it with
    the ground truth's ``p_left`` (the same fixed-param forward pass that
    generated the responses). Returns the per-model correlations and their mean
    — the off-the-shelf alternatives the loop starts from in experiment 1,
    before any fitting or agent-written models. Fails loudly only if *no* other
    seed model yields a defined correlation.

    ``gt_models_dir`` is where the ground-truth model lives (default:
    ``seed_models_dir``); the *other* seed models always come from
    ``seed_models_dir``. For an impossible ground truth these differ, and since
    the impossible model is not among the project seeds nothing is excluded —
    every seed model is scored against it.
    """
    seed_models_dir = Path(seed_models_dir)
    gt_models_dir = (
        Path(gt_models_dir) if gt_models_dir is not None else seed_models_dir
    )
    gt_p = p_left_fixed_params(gt_model, gt_models_dir, eval_stimuli, gt_params)
    defaults = resolve_generating_params(None, seed_models_dir, gt_family_dir)

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
    gt_models_dir: Optional[Path] = None,
    predict_max_draws: Optional[int] = None,
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
    gt_models_dir = (
        Path(gt_models_dir) if gt_models_dir is not None else Path(seed_models_dir)
    )
    gt_p = p_left_fixed_params(gt_model, gt_models_dir, eval_stimuli, gt_params)
    eval_rows = _raw_eval_rows(eval_stimuli)
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
        predict_max_draws=predict_max_draws,
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
    gt_models_dir: Optional[Path] = None,
    eval_pool_override: Optional[Mapping[str, Any]] = None,
    extra_excluded_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Recompute every ground truth's trajectory from its finished run tree.

    Reads each ``gt_run``'s on-disk ``run_root`` (its ``history.json`` per
    experiment) and ``eval_stimuli.json``, then recomputes the best-model and
    Bayesian-model-average trajectories plus the default-param and fitted-seed
    baselines through the shared MCMC cache — so a run whose hours-long agentic
    loop already finished can be re-analyzed (e.g. to add a baseline or
    regenerate the figure) without re-running any agents. Returns a new result;
    the input is not mutated.

    ``gt_models_dir`` (default: ``seed_models_dir``) is where each ground-truth
    generator lives. Pass the impossible-models directory to re-score a run
    whose ground truth sits outside the seed pool.

    ``eval_pool_override`` re-derives the held-out stimulus set instead of
    reading each run's recorded ``eval_stimuli.json``. The run's *training*
    pairs are still excluded (holdout is preserved), but the pool is rebuilt
    from the override's ``lengths``/``exhaustive``/``n_pairs``/``seed``/
    ``min_remaining`` (and ``predict_max_draws`` thins prediction). This is how
    runs whose original eval pools differed (a sampled set vs. the exhaustive
    space) are re-scored on one common pool. The enriched result then advertises
    the pool it actually scored on (its top-level ``eval_pool`` and each
    ``gt_run``'s ``n_eval_stimuli``/``n_eval_dropped`` are updated).
    """
    seed_models_dir = Path(seed_models_dir)
    gt_models_dir = Path(gt_models_dir) if gt_models_dir is not None else None
    n_experiments = int(result["n_experiments"])
    fit_kwargs = dict(result.get("fit_kwargs", {}))

    eval_pool = (
        dict(eval_pool_override)
        if eval_pool_override is not None
        else dict(result.get("eval_pool", {}))
    )
    predict_max_draws = eval_pool.get("predict_max_draws")

    # Names only — the fitted-seed baseline fits these by MCMC, so no
    # pure-Python family twin (and no default params) is required here.
    all_seed_models = set(seed_model_names(seed_models_dir))
    new_runs: List[Dict[str, Any]] = []
    for gt_run in result["gt_runs"]:
        run_root = Path(gt_run["run_root"])
        rebuilt: Optional[Dict[str, Any]] = None
        if eval_pool_override is not None:
            rebuilt = build_eval_stimuli(
                run_root,
                n_experiments=n_experiments,
                n_pairs=int(eval_pool.get("n_pairs", 0)),
                lengths=tuple(eval_pool["lengths"]),
                seed=int(eval_pool.get("seed", 0)),
                min_remaining=int(eval_pool.get("min_remaining", 1)),
                exhaustive=bool(eval_pool.get("exhaustive", False)),
                extra_excluded_pairs=extra_excluded_pairs,
            )
            eval_stimuli = rebuilt["stimuli"]
        else:
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
            gt_models_dir=gt_models_dir,
            predict_max_draws=predict_max_draws,
        )
        baseline = seed_baseline_correlation(
            gt_run["gt_model"],
            gt_run["params"],
            eval_stimuli,
            seed_models_dir=seed_models_dir,
            gt_models_dir=gt_models_dir,
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
            gt_models_dir=gt_models_dir,
            predict_max_draws=predict_max_draws,
        )
        new_run = {
            **gt_run,
            "trajectory": trajectory,
            "baseline": baseline,
            "fitted_baseline": fitted_baseline,
        }
        if rebuilt is not None:
            new_run["n_eval_stimuli"] = len(rebuilt["stimuli"])
            new_run["n_eval_dropped"] = rebuilt["n_dropped"]
        new_runs.append(new_run)

    enriched = {**result, "gt_runs": new_runs}
    if eval_pool_override is not None:
        enriched["eval_pool"] = {**dict(result.get("eval_pool", {})), **eval_pool}
    return enriched
