"""Bayesian model posterior over PyMC cognitive models.

For each model, ELPD-LOO is computed on the pooled observed responses; these
log-scores are softmaxed (a uniform model prior) to form the posterior over
models:

    P(model | data) ∝ exp(elpd_loo(model))

Output JSON:
    {
      "posteriors":      {"alternation": 0.82, "representativeness": 0.18},
      "elpd_loo":        {"alternation": -38.1, "representativeness": -45.7},
      "n_trials":        300
    }
"""

from __future__ import annotations

import csv

import numpy as np
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


from src.model_comparison.cv import LOO, Selection, cv_pointwise


def _flag_not_converged(name: str, row: Dict[str, Any]) -> None:
    """Set a comparison row's ``not_converged`` from its convergence problems,
    and say so out loud when it has any."""
    row["not_converged"] = bool(row["convergence_problems"])
    if row["not_converged"]:
        print(
            f"  [warn] {name} did not converge: "
            f"{'; '.join(row['convergence_problems'])}. It cannot be "
            "exported or pruned.",
            file=sys.stderr,
            flush=True,
        )


def compare_table(
    responses_path: Path,
    models_dir: Path,
    *,
    cache_dir: Optional[Path] = None,
    names: Optional[Sequence[str]] = None,
    selection: Selection = LOO,
    **fit_kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """Distinguishability diagnostic via ``arviz.compare`` (PSIS-LOO), or by
    grouped cross-validation (``selection``, ``src.model_comparison.cv``).

    With ``selection.criterion == "grouped_cv"`` the same fields are returned
    from each model's out-of-fold log predictive density: ``elpd_loo`` holds
    the ELPD-CV (the field name is kept so every reader works unchanged),
    ``dse`` and ``dse_clustered`` are the trial-level and stimulus-clustered
    standard errors of the pointwise differences, ``weight`` is a softmax of
    ELPD-CV (a report field; the stacking weights need PSIS-LOO),
    ``loo_unreliable`` is False (CV has no importance-sampling verdict), and
    convergence comes from the full-data fit. Every row records
    ``criterion`` and, for CV, ``n_cv_points``.

    ``names`` restricts the comparison to those manifest models (every
    ``elpd_diff`` and ``dse`` is then against the best of them); ``None``
    compares every manifest model.

    Fits each manifest model (reusing the in-process / on-disk fit cache, so this
    adds no MCMC when called after ``model_posterior``), then runs
    ``az.compare`` to rank them and report, per model:

    - ``rank``: 0 = best by ELPD-LOO
    - ``elpd_loo``: the model's ELPD
    - ``elpd_diff``: ELPD difference from the best model (0 for the best)
    - ``dse``: standard error of that difference (0 for the best). Two models are
      only meaningfully distinguishable when ``elpd_diff`` is large relative to
      ``dse`` (a rough rule of thumb is ``elpd_diff > 2 * dse``).
    - ``weight``: Akaike-style stacking weight from ``az.compare``.
    - ``dse_clustered``: the standard error of ``elpd_diff`` with each
      stimulus (unordered pair) as one observation
      (``src.models.clustered_se``); 0 for the best. Pruning uses it: the
      trial-level ``dse`` treats the 40 correlated responses to a pair as
      independent and is about half as large.
    - ``loo_unreliable``: True when this model's PSIS-LOO estimate is
      untrustworthy — more than a tolerated proportion of its trials have a
      high Pareto k, trials whose LOO term is exact excluded (see
      ``src.models.loo_reliability``; this is deliberately NOT arviz's blanket
      any-k>0.7 ``warning``). Its ELPD (and the posterior built from it) should
      then be read with caution: the inner loop excludes such a model from
      export and the registry zeroes its design prior.
    - ``n_bad_k``, ``frac_bad_k``, ``n_exact_loo_points``, ``max_pareto_k``:
      the evidence behind that verdict, recorded for audit.
    - ``not_converged`` / ``convergence_problems``: whether the MCMC fit failed
      the convergence gate (divergences, R-hat, bulk ESS; see
      ``pymc_inference.convergence_problems``), and why. Like an unreliable
      PSIS-LOO, a non-converged model is never exported or pruned.

    Each fit's PSIS-LOO is computed once (``FittedModel.loo_diagnostics``) and
    handed to ``az.compare`` as pointwise ``ELPDData``, so nothing is rescored.

    Returns a plain dict keyed by model name (JSON-serialisable).
    """
    import arviz as az  # type: ignore
    from src.models.clustered_se import cluster_dse, stimulus_clusters  # type: ignore
    from src.models.loo_reliability import describe_unreliable  # type: ignore
    from src.models.model_manifest import read_loadable_model_names  # type: ignore
    from src.models.pymc_inference import convergence_problems_of, fit_models_cached  # type: ignore

    responses_path = Path(responses_path)
    models_dir = Path(models_dir)

    model_names = read_loadable_model_names(models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")
    if names is not None:
        unknown = sorted(set(names) - set(model_names))
        if unknown or not names:
            raise ValueError(
                f"compare_table: {unknown or 'no names'} not among the loadable "
                f"models of {models_dir} ({model_names})."
            )
        model_names = [name for name in model_names if name in set(names)]

    fits = fit_models_cached(
        model_names,
        models_dir=models_dir,
        responses_path=responses_path,
        cache_dir=cache_dir,
        **fit_kwargs,
    )
    if selection.criterion == "grouped_cv":
        return _cv_compare_table(model_names, models_dir, responses_path, fits, selection, cache_dir, fit_kwargs)
    diagnostics = {name: fits[name].loo_diagnostics() for name in model_names}

    cmp = az.compare({name: d.loo for name, d in diagnostics.items()}, ic="loo")
    best = cmp.index[0]
    with responses_path.open(encoding="utf-8", newline="") as f:
        clusters = stimulus_clusters(list(csv.DictReader(f)))
    pointwise = {name: np.asarray(d.loo.loo_i) for name, d in diagnostics.items()}

    out: Dict[str, Dict[str, float]] = {}
    for rank, (name, row) in enumerate(cmp.iterrows()):
        diag = diagnostics[name]
        out[name] = {
            "rank": rank,
            "elpd_loo": float(row["elpd_loo"]),
            "elpd_diff": float(row["elpd_diff"]),
            "dse": float(row["dse"]),
            "dse_clustered": (
                0.0
                if name == best
                else cluster_dse(pointwise[best], pointwise[name], clusters)
            ),
            "weight": float(row["weight"]),
            "loo_unreliable": diag.unreliable,
            "n_bad_k": diag.n_bad_k,
            "frac_bad_k": diag.frac_bad_k,
            "n_exact_loo_points": diag.n_exact,
            "max_pareto_k": diag.max_pareto_k,
            "convergence_problems": convergence_problems_of(fits[name]),
            "criterion": "loo",
        }
        _flag_not_converged(name, out[name])
        if diag.unreliable:
            print(
                f"  [warn] PSIS-LOO for {describe_unreliable(name, diag)} Any "
                "posterior built from it may be off.",
                file=sys.stderr,
                flush=True,
            )
    return out


def _cv_compare_table(
    model_names: Sequence[str],
    models_dir: Path,
    responses_path: Path,
    fits: Mapping[str, Any],
    selection: Selection,
    cache_dir: Optional[Path],
    fit_kwargs: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """``compare_table``'s rows from grouped CV (see its docstring)."""
    from src.models.clustered_se import cluster_dse, stimulus_clusters  # type: ignore
    from src.models.pymc_inference import convergence_problems_of  # type: ignore

    lpd, rows = cv_pointwise(model_names, models_dir, responses_path, selection, cache_dir=cache_dir, **fit_kwargs)
    elpd = {name: float(lpd[name].sum()) for name in model_names}
    ranked = sorted(model_names, key=lambda name: -elpd[name])
    best = ranked[0]
    clusters = stimulus_clusters(rows)
    top = max(elpd.values())
    weights = {name: math.exp(elpd[name] - top) for name in model_names}
    total = sum(weights.values())
    out: Dict[str, Dict[str, Any]] = {}
    for rank, name in enumerate(ranked):
        diff = lpd[best] - lpd[name]
        out[name] = {
            "rank": rank,
            "elpd_loo": elpd[name],
            "elpd_diff": elpd[best] - elpd[name],
            "dse": 0.0 if name == best else float(np.sqrt(len(diff) * np.var(diff))),
            "dse_clustered": 0.0 if name == best else cluster_dse(lpd[best], lpd[name], clusters),
            "weight": weights[name] / total,
            "loo_unreliable": False,
            "convergence_problems": convergence_problems_of(fits[name]),
            "criterion": "grouped_cv",
            "n_cv_points": int(len(diff)),
        }
        _flag_not_converged(name, out[name])
    return out


def model_posterior(
    responses_path: Path,
    models_dir: Path,
    *,
    cache_dir: Optional[Path] = None,
    selection: Selection = LOO,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """Compute the Bayesian posterior over PyMC cognitive models.

    Scored by ELPD-LOO, or by ELPD from grouped cross-validation when
    ``selection`` says so (still reported under ``elpd_loo``; ``criterion``
    says which).

    Fits each model in `<models_dir>/models_manifest.yaml` to the pooled
    responses at `responses_path` (MCMC), scores each by ELPD-LOO, and
    softmaxes the scores to produce a normalized posterior.

    Extra ``fit_kwargs`` (e.g. ``draws``, ``tune``, ``chains``) are forwarded
    to the MCMC fit of every model.

    Caveat (overconfidence): the posterior is a softmax of total ELPD-LOO, which
    sums pointwise log-densities over every trial. A per-trial advantage of even a
    fraction of a nat, multiplied by hundreds of trials, drives the softmax to a
    near one-hot vector — and this does NOT account for the standard error of the
    ELPD difference. So ``posteriors`` will routinely read ~1.0 for one model even
    when ``compare_table``'s ``dse`` says the models are statistically
    indistinguishable (``elpd_diff`` < ~2·dse). Treat ``posteriors`` as a ranking
    that is overconfident about its certainty; use the ``dse`` column of
    ``compare_table`` for the standard-error-aware view of distinguishability.

    Returns a dict with `posteriors`, `elpd_loo`, `n_trials` and `criterion`.
    """
    from src.models.model_manifest import read_loadable_model_names  # type: ignore
    from src.model_comparison import likelihood as _likelihood  # type: ignore

    responses_path = Path(responses_path)
    models_dir = Path(models_dir)

    model_names = read_loadable_model_names(models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")

    if selection.criterion == "grouped_cv":
        lpd, _ = cv_pointwise(model_names, models_dir, responses_path, selection, cache_dir=cache_dir, **fit_kwargs)
        elpd: Dict[str, float] = {m: float(lpd[m].sum()) for m in model_names}
    else:
        elpd = {
            m: _likelihood.log_likelihood(
                m, responses_path, models_dir, cache_dir=cache_dir, **fit_kwargs
            )
            for m in model_names
        }

    # A non-finite ELPD (a model assigning ~0 probability to an observed outcome,
    # or a degenerate fit) would silently poison the softmax below: a single NaN
    # makes every posterior NaN, and -inf would zero a model without a trace. Fail
    # loudly with the offending model(s) rather than emit a corrupt posterior.
    nonfinite = [m for m in model_names if not math.isfinite(elpd[m])]
    if nonfinite:
        raise ValueError(
            f"Non-finite ELPD-LOO for model(s) {nonfinite} "
            f"({ {m: elpd[m] for m in nonfinite} }); refusing to build a posterior "
            "a NaN/inf would silently corrupt. Inspect the model(s)' likelihood."
        )

    log_scores = elpd
    max_score = max(log_scores.values())
    sum_exp = sum(math.exp(s - max_score) for s in log_scores.values())
    log_norm = max_score + math.log(sum_exp)
    posteriors = {m: round(math.exp(log_scores[m] - log_norm), 6) for m in model_names}

    with responses_path.open(encoding="utf-8", newline="") as f:
        n_trials = sum(1 for _ in csv.DictReader(f))

    result: Dict[str, Any] = {
        "posteriors": posteriors,
        "elpd_loo": {m: round(elpd[m], 4) for m in model_names},
        "n_trials": n_trials,
        "criterion": selection.criterion,
    }
    return result
