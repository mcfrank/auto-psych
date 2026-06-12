"""Adaptive (sequential) EIG-driven recovery on the pure-Python model families.

Bayesian optimal experimental design for recovery diagnostics: keep an exact
posterior over a grid of model parameters, repeatedly pick the candidate
stimulus with the highest expected information gain *under the current
posterior*, simulate the true response, update the posterior, and repeat.

Two loops:

* :func:`run_adaptive_parameter_recovery` — recover one model's parameters. The
  EIG is the information a response carries about the parameters; the loop
  drives the parameter posterior toward the true values.
* :func:`run_adaptive_model_recovery` — recover which model generated the data.
  Each candidate model keeps its own parameter grid; the EIG is the information
  about *model identity*, and a model posterior is updated from the marginal
  likelihood each round.

The same grid machinery also evaluates *fixed* (non-sequential) designs:
:func:`compare_parameter_recovery` / :func:`compare_model_recovery` choose two
stimulus sets from one candidate pool — the top-k by expected information gain
under the prior ("eig") and a uniform draw of the same size ("random") — and
recover the same sampled ground truths with both, so the arms differ only in
which stimulus set was chosen.

Everything is computed from the families' ``predict_left`` on a parameter grid
(no MCMC), so the loops are exact, fast, and deterministic given ``seed``. This
is the fast design-evaluation counterpart to the one-shot PyMC recovery.
"""

from __future__ import annotations

import importlib
import itertools
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from src.subjective_randomness.recover import sample_true_params

_EPS = 1e-12


def _load_family(model_name: str):
    return importlib.import_module(
        f"src.subjective_randomness.model_families.{model_name}"
    )


def _binary_entropy_array(p: np.ndarray) -> np.ndarray:
    """Binary entropy (bits), elementwise; 0 at the endpoints."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _entropy_bits(weights: np.ndarray) -> float:
    """Shannon entropy (bits) of a normalized weight vector."""
    w = weights[weights > 0]
    return float(-(w * np.log2(w)).sum())


def _build_grid(
    module: Any, points_per_dim: int
) -> Tuple[List[str], List[Dict[str, float]], Dict[str, np.ndarray]]:
    """Regular grid over the family's ``PARAM_BOUNDS``.

    Returns the parameter names, one param-dict per grid point, and per-parameter
    arrays of the grid values (aligned with the grid-point ordering).
    """
    bounds = module.PARAM_BOUNDS
    names = list(bounds)
    axes = [np.linspace(bounds[n][0], bounds[n][1], points_per_dim) for n in names]
    grid_dicts = [
        {n: float(v) for n, v in zip(names, point)}
        for point in itertools.product(*axes)
    ]
    grid_values = {n: np.array([d[n] for d in grid_dicts]) for n in names}
    return names, grid_dicts, grid_values


def _prediction_matrix(
    module: Any,
    grid_dicts: Sequence[Mapping[str, float]],
    candidates: Sequence[Mapping[str, str]],
) -> np.ndarray:
    """``p_left`` for every (grid point, candidate stimulus), shape (n_grid, n_cand)."""
    matrix = np.empty((len(grid_dicts), len(candidates)))
    for gi, params in enumerate(grid_dicts):
        for ci, stim in enumerate(candidates):
            matrix[gi, ci] = module.predict_left(stim, params)
    return matrix


def _eig_per_candidate(pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """EIG (bits) about the grid variable for each candidate, given ``weights``.

    ``pred`` is (n_grid, n_cand) of p_left. The response at candidate ``c`` is
    ``Bernoulli(p̄_c)`` with ``p̄_c = Σ_g w_g pred[g,c]``; the mutual information
    between the grid variable and the response is ``H(p̄_c) − Σ_g w_g H(pred[g,c])``.
    """
    p_bar = weights @ pred  # (n_cand,)
    conditional = weights @ _binary_entropy_array(pred)  # (n_cand,)
    return _binary_entropy_array(p_bar) - conditional


def _log_likelihood(pred_col: np.ndarray, n_left: int, n: int) -> np.ndarray:
    """Log Binomial-kernel likelihood per grid point for ``n_left`` of ``n`` left."""
    p = np.clip(pred_col, _EPS, 1.0 - _EPS)
    return n_left * np.log(p) + (n - n_left) * np.log(1.0 - p)


def run_adaptive_parameter_recovery(
    model_name: str,
    true_params: Mapping[str, float],
    candidate_pool: Sequence[Mapping[str, str]],
    *,
    n_rounds: int,
    n_participants: int = 1,
    points_per_dim: int = 7,
    seed: int = 0,
) -> Dict[str, Any]:
    """Adaptively recover one family's parameters via sequential EIG selection.

    Each round selects the unused candidate with the highest parameter-EIG under
    the current grid posterior, simulates ``n_participants`` Bernoulli responses
    from ``true_params``, and updates the posterior. Returns the selected
    stimuli, the posterior mean/sd per parameter, the uninformed prior mean (a
    baseline), and the grid-posterior entropy before and after.
    """
    module = _load_family(model_name)
    candidates = list(candidate_pool)
    names, grid_values, pred = _parameter_design(module, candidates, points_per_dim)
    return _parameter_recovery_run(
        module,
        names,
        grid_values,
        pred,
        candidates,
        model_name=model_name,
        true_params=true_params,
        n_rounds=n_rounds,
        n_participants=n_participants,
        points_per_dim=points_per_dim,
        seed=seed,
    )


def _parameter_design(
    module: Any, candidates: Sequence[Mapping[str, str]], points_per_dim: int
) -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray]:
    """Grid + prediction matrix for one family over a fixed candidate pool.

    The expensive, truth-independent part of a recovery run; build it once and
    reuse it across repeated runs on the same pool.
    """
    names, grid_dicts, grid_values = _build_grid(module, points_per_dim)
    return names, grid_values, _prediction_matrix(module, grid_dicts, candidates)


def _parameter_recovery_run(
    module: Any,
    names: List[str],
    grid_values: Dict[str, np.ndarray],
    pred: np.ndarray,
    candidates: List[Dict[str, str]],
    *,
    model_name: str,
    true_params: Mapping[str, float],
    n_rounds: int,
    n_participants: int,
    points_per_dim: int,
    seed: int,
) -> Dict[str, Any]:
    """One parameter-recovery loop on a precomputed grid design."""
    if n_rounds > len(candidates):
        raise ValueError(
            f"n_rounds ({n_rounds}) exceeds the candidate pool size "
            f"({len(candidates)}); selection is without replacement."
        )

    weights = np.full(pred.shape[0], 1.0 / pred.shape[0])
    prior_mean = {n: float(weights @ grid_values[n]) for n in names}
    prior_entropy = _entropy_bits(weights)

    rng = np.random.default_rng(seed)
    true_p = np.array([module.predict_left(s, true_params) for s in candidates])
    used = np.zeros(len(candidates), dtype=bool)
    selected: List[Dict[str, Any]] = []

    for _ in range(n_rounds):
        eig = _eig_per_candidate(pred, weights)
        eig[used] = -np.inf
        c = int(np.argmax(eig))
        used[c] = True
        n_left = int(rng.binomial(n_participants, true_p[c]))
        log_post = np.log(np.clip(weights, 1e-300, None)) + _log_likelihood(
            pred[:, c], n_left, n_participants
        )
        log_post -= log_post.max()
        weights = np.exp(log_post)
        weights /= weights.sum()
        selected.append({**dict(candidates[c]), "eig": float(eig[c])})

    posterior_mean = {n: float(weights @ grid_values[n]) for n in names}
    posterior_sd = {
        n: float(np.sqrt(weights @ (grid_values[n] - posterior_mean[n]) ** 2))
        for n in names
    }
    return {
        "model": model_name,
        "true_params": dict(true_params),
        "n_rounds": n_rounds,
        "n_participants": n_participants,
        "points_per_dim": points_per_dim,
        "selected_stimuli": selected,
        "prior_mean": prior_mean,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "estimate_error": {
            n: posterior_mean[n] - float(true_params[n]) for n in names
        },
        "prior_entropy_bits": prior_entropy,
        "final_entropy_bits": _entropy_bits(weights),
    }


def run_adaptive_model_recovery(
    candidate_pool: Sequence[Mapping[str, str]],
    *,
    true_model: str,
    true_params: Mapping[str, float],
    model_names: Sequence[str],
    n_rounds: int,
    n_participants: int = 1,
    points_per_dim: int = 7,
    seed: int = 0,
) -> Dict[str, Any]:
    """Adaptively recover which model generated the data via sequential EIG.

    Each candidate model keeps its own parameter grid (posterior). Every round
    selects the unused candidate with the highest model-identity EIG under the
    current beliefs, simulates the true model's response, and updates both each
    model's parameter posterior and the model posterior (from each model's
    marginal likelihood for the observed response). Returns the model posterior,
    the recovered (MAP) model, and the selected stimuli.
    """
    candidates = list(candidate_pool)
    modules = {m: _load_family(m) for m in model_names}
    pred = _model_design(modules, candidates, points_per_dim)
    return _model_recovery_run(
        modules,
        pred,
        candidates,
        true_model=true_model,
        true_params=true_params,
        model_names=list(model_names),
        n_rounds=n_rounds,
        n_participants=n_participants,
        points_per_dim=points_per_dim,
        seed=seed,
    )


def _model_design(
    modules: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    points_per_dim: int,
) -> Dict[str, np.ndarray]:
    """Per-model prediction matrices over a fixed candidate pool.

    The expensive, truth-independent part of a model-recovery run; build it
    once and reuse it across repeated runs on the same pool.
    """
    pred = {}
    for m, module in modules.items():
        _, grid_dicts, _ = _build_grid(module, points_per_dim)
        pred[m] = _prediction_matrix(module, grid_dicts, candidates)
    return pred


def _model_recovery_run(
    modules: Mapping[str, Any],
    pred: Mapping[str, np.ndarray],
    candidates: List[Dict[str, str]],
    *,
    true_model: str,
    true_params: Mapping[str, float],
    model_names: List[str],
    n_rounds: int,
    n_participants: int,
    points_per_dim: int,
    seed: int,
) -> Dict[str, Any]:
    """One model-recovery loop on precomputed per-model grid designs."""
    if true_model not in model_names:
        raise ValueError(
            f"true_model {true_model!r} is not among model_names {list(model_names)}."
        )
    if n_rounds > len(candidates):
        raise ValueError(
            f"n_rounds ({n_rounds}) exceeds the candidate pool size "
            f"({len(candidates)}); selection is without replacement."
        )

    grid_weights = {
        m: np.full(pred[m].shape[0], 1.0 / pred[m].shape[0]) for m in model_names
    }

    log_model_weight = {m: 0.0 for m in model_names}  # uniform model prior

    def model_posterior() -> Dict[str, float]:
        logs = np.array([log_model_weight[m] for m in model_names])
        probs = np.exp(logs - logs.max())
        probs /= probs.sum()
        return {m: float(p) for m, p in zip(model_names, probs)}

    rng = np.random.default_rng(seed)
    true_p = np.array(
        [modules[true_model].predict_left(s, true_params) for s in candidates]
    )
    used = np.zeros(len(candidates), dtype=bool)
    selected: List[Dict[str, Any]] = []

    for _ in range(n_rounds):
        probs = model_posterior()
        pred_per_model = {m: grid_weights[m] @ pred[m] for m in model_names}  # (n_cand,)
        p_bar = sum(probs[m] * pred_per_model[m] for m in model_names)
        conditional = sum(
            probs[m] * _binary_entropy_array(pred_per_model[m]) for m in model_names
        )
        eig = _binary_entropy_array(p_bar) - conditional
        eig[used] = -np.inf
        c = int(np.argmax(eig))
        used[c] = True
        n_left = int(rng.binomial(n_participants, true_p[c]))

        for m in model_names:
            log_post = np.log(np.clip(grid_weights[m], 1e-300, None)) + _log_likelihood(
                pred[m][:, c], n_left, n_participants
            )
            # Marginal likelihood of this response under model m (evidence step).
            log_model_weight[m] += float(logsumexp(log_post))
            log_post -= log_post.max()
            w = np.exp(log_post)
            grid_weights[m] = w / w.sum()
        selected.append({**dict(candidates[c]), "eig": float(eig[c])})

    posterior = model_posterior()
    recovered = max(posterior, key=posterior.get)
    return {
        "true_model": true_model,
        "model_names": list(model_names),
        "n_rounds": n_rounds,
        "n_participants": n_participants,
        "points_per_dim": points_per_dim,
        "model_posterior": posterior,
        "recovered_model": recovered,
        "recovered_correct": recovered == true_model,
        "selected_stimuli": selected,
    }


def run_adaptive_model_confusion(
    candidate_pool: Sequence[Mapping[str, str]],
    *,
    generating_params: Mapping[str, Mapping[str, float]],
    model_names: Sequence[str],
    n_rounds: int,
    n_participants: int = 1,
    points_per_dim: int = 7,
    seed: int = 0,
) -> Dict[str, Any]:
    """Adaptive model recovery with each generating model as truth, in turn.

    Runs one model-recovery loop per entry in ``generating_params`` (each with
    its own selected stimuli, on one shared precomputed grid design) and
    assembles a confusion-shaped result: one ``generating`` entry per true model
    carrying the recovered model and the model posterior. Every run shares the
    same candidate pool but selects its own sequence of stimuli.
    """
    candidates = list(candidate_pool)
    modules = {m: _load_family(m) for m in model_names}
    pred = _model_design(modules, candidates, points_per_dim)

    generating: List[Dict[str, Any]] = []
    for gen_model, params in generating_params.items():
        run = _model_recovery_run(
            modules,
            pred,
            candidates,
            true_model=gen_model,
            true_params=params,
            model_names=list(model_names),
            n_rounds=n_rounds,
            n_participants=n_participants,
            points_per_dim=points_per_dim,
            seed=seed,
        )
        generating.append(
            {
                "generating_model": gen_model,
                "recovered_model": run["recovered_model"],
                "recovered_correct": run["recovered_correct"],
                "model_posterior": run["model_posterior"],
                "selected_stimuli": run["selected_stimuli"],
            }
        )
    return {
        "model_names": list(model_names),
        "n_rounds": n_rounds,
        "n_participants": n_participants,
        "points_per_dim": points_per_dim,
        "generating": generating,
    }


# ── EIG-optimized vs. random stimulus sets, compared on the same truths ──

# Truth draws use their own seed stream so they stay decoupled from the
# response-simulation seeds (`seed + repeat`) used inside each run.
_COMPARISON_TRUTH_SEED_OFFSET = 70000


def _require_positive_repeats(n_repeats: int) -> None:
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")


def _require_valid_set_size(n_stimuli: int, pool_size: int) -> None:
    if n_stimuli < 1:
        raise ValueError(f"n_stimuli must be >= 1, got {n_stimuli}.")
    if n_stimuli > pool_size:
        raise ValueError(
            f"n_stimuli ({n_stimuli}) exceeds the candidate pool size ({pool_size})."
        )


def _stimulus_sets(
    scores: np.ndarray, n_stimuli: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """The two fixed designs: top-``n_stimuli`` by prior EIG vs. a uniform draw.

    The "eig" set is ordered most-informative-first; the "random" set is a
    same-size draw (without replacement) from the same pool.
    """
    return {
        "eig": np.argsort(-scores)[:n_stimuli],
        "random": rng.choice(scores.size, size=n_stimuli, replace=False),
    }


def _annotate_set(
    candidates: Sequence[Mapping[str, str]], scores: np.ndarray, indices: np.ndarray
) -> List[Dict[str, Any]]:
    return [{**dict(candidates[i]), "eig": float(scores[i])} for i in indices]


def _batch_log_likelihood(
    pred_subset: np.ndarray, counts: np.ndarray, n_participants: int
) -> np.ndarray:
    """Summed log Binomial-kernel likelihood per grid point for a fixed set.

    ``pred_subset`` is (n_grid, k) of p_left for the set's stimuli; ``counts``
    holds the observed left-choices (out of ``n_participants``) per stimulus.
    """
    p = np.clip(pred_subset, _EPS, 1.0 - _EPS)
    return (counts * np.log(p) + (n_participants - counts) * np.log(1.0 - p)).sum(
        axis=1
    )


def _batch_posterior_weights(
    pred_subset: np.ndarray, counts: np.ndarray, n_participants: int
) -> np.ndarray:
    """Grid posterior from one batch of counts, starting from a uniform prior."""
    log_post = _batch_log_likelihood(pred_subset, counts, n_participants)
    log_post -= log_post.max()
    weights = np.exp(log_post)
    return weights / weights.sum()


def _model_log_evidence(
    pred_subset: np.ndarray, counts: np.ndarray, n_participants: int
) -> float:
    """Log marginal likelihood of the counts under one model's uniform grid prior."""
    log_lik = _batch_log_likelihood(pred_subset, counts, n_participants)
    return float(logsumexp(log_lik) - np.log(log_lik.size))


def _family_bounds(module: Any) -> Dict[str, Tuple[float, float]]:
    return {n: (float(lo), float(hi)) for n, (lo, hi) in module.PARAM_BOUNDS.items()}


def _pearson_r(xs: np.ndarray, ys: np.ndarray) -> Any:
    """Pearson correlation; None when undefined (n < 2 or either side constant)."""
    if xs.size < 2 or np.unique(xs).size == 1 or np.unique(ys).size == 1:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _parameter_arm_summary(
    runs: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Truth-vs-estimate recovery quality per parameter for one arm's runs."""
    summary = {}
    for name in names:
        trues = np.array([r["true_params"][name] for r in runs])
        ests = np.array([r["posterior_mean"][name] for r in runs])
        errors = ests - trues
        summary[name] = {
            "pearson_r": _pearson_r(trues, ests),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(np.mean(errors)),
            "mean_posterior_sd": float(
                np.mean([r["posterior_sd"][name] for r in runs])
            ),
        }
    return summary


def compare_parameter_recovery(
    model_name: str,
    candidate_pool: Sequence[Mapping[str, str]],
    *,
    n_repeats: int,
    n_stimuli: int,
    n_participants: int = 1,
    points_per_dim: int = 7,
    seed: int = 0,
) -> Dict[str, Any]:
    """Parameter recovery with an EIG-optimized vs. a random stimulus set.

    Scores every pool stimulus by its parameter-EIG under a uniform prior on
    the family's grid, then recovers each of ``n_repeats`` ground truths
    (sampled from ``PARAM_BOUNDS``) twice — once on the top-``n_stimuli`` set
    ("eig") and once on a uniform same-size draw from the pool ("random") — so
    within a repeat the arms differ only in which stimulus set was used.
    Returns each arm's chosen stimuli plus its runs and a per-parameter
    summary (pearson_r, rmse, bias, mean posterior sd).
    """
    _require_positive_repeats(n_repeats)
    candidates = list(candidate_pool)
    _require_valid_set_size(n_stimuli, len(candidates))
    module = _load_family(model_name)
    names, grid_values, pred = _parameter_design(module, candidates, points_per_dim)
    bounds = _family_bounds(module)

    prior = np.full(pred.shape[0], 1.0 / pred.shape[0])
    # Mutual information is nonnegative; clip the float noise so near-zero
    # scores cannot dip below 0.
    scores = np.maximum(_eig_per_candidate(pred, prior), 0.0)
    sets = _stimulus_sets(scores, n_stimuli, np.random.default_rng(seed))

    arms: Dict[str, Dict[str, Any]] = {
        arm: {
            "stimuli": _annotate_set(candidates, scores, indices),
            "mean_stimulus_eig": float(scores[indices].mean()),
            "runs": [],
        }
        for arm, indices in sets.items()
    }
    for repeat in range(n_repeats):
        truth = sample_true_params(
            bounds, random.Random(seed + _COMPARISON_TRUTH_SEED_OFFSET + repeat)
        )
        true_p = np.array([module.predict_left(s, truth) for s in candidates])
        for arm, indices in sets.items():
            rng = np.random.default_rng(seed + repeat)
            counts = rng.binomial(n_participants, true_p[indices])
            weights = _batch_posterior_weights(
                pred[:, indices], counts, n_participants
            )
            posterior_mean = {n: float(weights @ grid_values[n]) for n in names}
            posterior_sd = {
                n: float(
                    np.sqrt(weights @ (grid_values[n] - posterior_mean[n]) ** 2)
                )
                for n in names
            }
            arms[arm]["runs"].append(
                {
                    "repeat": repeat,
                    "true_params": truth,
                    "posterior_mean": posterior_mean,
                    "posterior_sd": posterior_sd,
                    "final_entropy_bits": _entropy_bits(weights),
                }
            )
    for arm in arms.values():
        arm["summary"] = _parameter_arm_summary(arm["runs"], names)

    return {
        "model": model_name,
        "n_repeats": n_repeats,
        "n_stimuli": n_stimuli,
        "n_participants": n_participants,
        "points_per_dim": points_per_dim,
        "seed": seed,
        "pool_size": len(candidates),
        "param_ranges": {n: [lo, hi] for n, (lo, hi) in bounds.items()},
        "arms": arms,
    }


def _model_arm_summary(
    runs: List[Dict[str, Any]], model_names: Sequence[str]
) -> Dict[str, Any]:
    """Accuracy, mean posterior on truth, and mean-posterior confusion."""
    confusion = {}
    for gen in model_names:
        gen_runs = [r for r in runs if r["generating_model"] == gen]
        confusion[gen] = {
            m: float(np.mean([r["model_posterior"][m] for r in gen_runs]))
            for m in model_names
        }
    return {
        "accuracy": float(np.mean([r["recovered_correct"] for r in runs])),
        "mean_true_posterior": float(
            np.mean([r["model_posterior"][r["generating_model"]] for r in runs])
        ),
        "confusion": confusion,
    }


def compare_model_recovery(
    candidate_pool: Sequence[Mapping[str, str]],
    *,
    model_names: Sequence[str],
    n_repeats: int,
    n_stimuli: int,
    n_participants: int = 1,
    points_per_dim: int = 7,
    seed: int = 0,
) -> Dict[str, Any]:
    """Model recovery with an EIG-optimized vs. a random stimulus set.

    Scores every pool stimulus by the mutual information between model
    identity and the response under each family's grid prior predictive
    (uniform model prior), picks the top-``n_stimuli`` set ("eig") and a
    uniform same-size draw ("random"), and recovers model identity on both
    sets for every generating model with sampled generating parameters
    (``n_repeats`` truths per generator, paired across arms). Returns each
    arm's chosen stimuli plus its runs, accuracy, mean posterior on the true
    model, and a mean-posterior confusion matrix.
    """
    _require_positive_repeats(n_repeats)
    candidates = list(candidate_pool)
    _require_valid_set_size(n_stimuli, len(candidates))
    modules = {m: _load_family(m) for m in model_names}
    pred = _model_design(modules, candidates, points_per_dim)

    # Prior-predictive p_left per model (uniform over its grid), then the
    # model-identity EIG per stimulus under a uniform model prior. Mutual
    # information is nonnegative; clip the float noise.
    prior_pred = {m: pred[m].mean(axis=0) for m in model_names}
    p_bar = sum(prior_pred[m] for m in model_names) / len(model_names)
    scores = np.maximum(
        _binary_entropy_array(p_bar)
        - sum(_binary_entropy_array(prior_pred[m]) for m in model_names)
        / len(model_names),
        0.0,
    )
    sets = _stimulus_sets(scores, n_stimuli, np.random.default_rng(seed))

    arm_runs: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in sets}
    for repeat in range(n_repeats):
        for gen_index, gen_model in enumerate(model_names):
            truth = sample_true_params(
                _family_bounds(modules[gen_model]),
                random.Random(
                    seed
                    + _COMPARISON_TRUTH_SEED_OFFSET
                    + repeat * len(model_names)
                    + gen_index
                ),
            )
            true_p = np.array(
                [modules[gen_model].predict_left(s, truth) for s in candidates]
            )
            for arm, indices in sets.items():
                rng = np.random.default_rng(seed + repeat)
                counts = rng.binomial(n_participants, true_p[indices])
                log_evidence = np.array(
                    [
                        _model_log_evidence(pred[m][:, indices], counts, n_participants)
                        for m in model_names
                    ]
                )
                probs = np.exp(log_evidence - log_evidence.max())
                probs /= probs.sum()
                posterior = {m: float(p) for m, p in zip(model_names, probs)}
                recovered = max(posterior, key=posterior.get)
                arm_runs[arm].append(
                    {
                        "repeat": repeat,
                        "generating_model": gen_model,
                        "true_params": truth,
                        "recovered_model": recovered,
                        "recovered_correct": recovered == gen_model,
                        "model_posterior": posterior,
                    }
                )

    return {
        "model_names": list(model_names),
        "n_repeats": n_repeats,
        "n_stimuli": n_stimuli,
        "n_participants": n_participants,
        "points_per_dim": points_per_dim,
        "seed": seed,
        "pool_size": len(candidates),
        "arms": {
            arm: {
                "stimuli": _annotate_set(candidates, scores, indices),
                "mean_stimulus_eig": float(scores[indices].mean()),
                "runs": arm_runs[arm],
                **_model_arm_summary(arm_runs[arm], model_names),
            }
            for arm, indices in sets.items()
        },
    }
