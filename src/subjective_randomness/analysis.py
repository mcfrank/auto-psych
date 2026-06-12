"""Summarize and analyze subjective-randomness recovery results.

Two analyses, one per recovery script:

* `parameter_recovery_summary` — given a Bayesian (PyMC) report from
  `pymc_recover.py`, report per-parameter recovery quality: bias, RMSE,
  spread of estimates, and 95% credible-interval coverage of the true value.

* `model_recovery_summary` — given a closed-ended confusion result from
  `model_recovery.py`, report per-generating-model and overall recovery: which
  model wins by posterior and by ELPD-LOO, and how often that is the true model.

Both consume the JSON these scripts already write; this module never runs MCMC.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np

from src.subjective_randomness.tidy import parameter_recovery_tidy_rows


def _ci_coverage_95(report: Mapping[str, Any], param: str) -> float | None:
    """Fraction of repeats whose 95% credible interval contains that run's truth.

    The truth comes from the run's own ``true_params`` (sampled-truth reports)
    or the report-level ``true_params`` (fixed-truth reports). Returns ``None``
    only when a run's posterior summary carries no interval (legacy reports
    written before q025/q975 were recorded), so the caller can distinguish
    "0% coverage" from "not applicable".
    """
    inside = 0
    n = 0
    for run in report["runs"]:
        entry = run["posterior"][param]
        if "q025" not in entry or "q975" not in entry:
            return None
        true_value = float(run.get("true_params", report.get("true_params"))[param])
        n += 1
        if entry["q025"] <= true_value <= entry["q975"]:
            inside += 1
    return inside / n if n else None


def parameter_recovery_summary(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per-parameter recovery quality for one recovery report.

    One row per parameter with the mean recovered estimate, bias (mean signed
    error), RMSE, the spread of estimates across repeats, the repeat count,
    the Pearson correlation between ground truth and estimate, and 95% CI
    coverage. Fields that are undefined for a given report shape are ``None``:
    ``true_value`` when truths vary across repeats (sampled-truth reports),
    ``pearson_r`` when they do not (fixed-truth reports), and
    ``ci_coverage_95`` when the posterior summaries carry no intervals.
    """
    tidy = parameter_recovery_tidy_rows(report)
    model = report["model"]
    params = list(dict.fromkeys(r["parameter"] for r in tidy))
    summary: List[Dict[str, Any]] = []
    for param in params:
        rows = [r for r in tidy if r["parameter"] == param]
        trues = np.array([r["true_value"] for r in rows], dtype="float64")
        estimates = np.array([r["estimate"] for r in rows], dtype="float64")
        errors = estimates - trues
        constant_truth = np.unique(trues).size == 1
        # Judge variance by distinct values, not std(): a constant truth like
        # 0.4 accumulates ~1e-17 of float noise in std() and would otherwise
        # yield a garbage near-zero correlation instead of "undefined".
        correlation_defined = (
            trues.size >= 2
            and not constant_truth
            and np.unique(estimates).size > 1
        )
        summary.append(
            {
                "model": model,
                "parameter": param,
                "true_value": float(trues[0]) if constant_truth else None,
                "mean_estimate": float(estimates.mean()),
                "bias": float(errors.mean()),
                "rmse": float(np.sqrt((errors**2).mean())),
                "estimate_sd": float(estimates.std()),
                "n_repeats": int(estimates.size),
                "pearson_r": (
                    float(np.corrcoef(trues, estimates)[0, 1])
                    if correlation_defined
                    else None
                ),
                "ci_coverage_95": _ci_coverage_95(report, param),
            }
        )
    return summary


def _distinguishability(
    entry: Mapping[str, Any], true_model: str
) -> Dict[str, Any]:
    """Distinguishability fields for one generating model from its `comparison`.

    Uses the inner loop's PSIS-LOO comparison table (per recovered model:
    ``rank``, ``elpd_diff`` and ``dse`` relative to the ELPD-best model). The
    ELPD winner is "clearly" ahead only when the runner-up's ``elpd_diff``
    exceeds ``2 * dse`` — otherwise the top models are statistically tied and
    "winning" is close to a coin flip. Returns ``None`` fields when no
    comparison table is present (older results), so absence is never read as a
    clear recovery.
    """
    comparison = entry.get("comparison")
    none_fields = {
        "winner_by_elpd": None,
        "winner_margin": None,
        "winner_margin_dse": None,
        "winner_distinguishable": None,
        "true_model_elpd_diff": None,
        "true_model_dse": None,
        "recovery_clear": None,
    }
    if not comparison:
        return none_fields

    by_rank = sorted(comparison.items(), key=lambda kv: kv[1]["rank"])
    winner_by_elpd = by_rank[0][0]
    runner_up = by_rank[1][1] if len(by_rank) > 1 else None
    margin = float(runner_up["elpd_diff"]) if runner_up else None
    margin_dse = float(runner_up["dse"]) if runner_up else None
    # A positive dse is required to call the gap significant.
    distinguishable = (
        margin is not None and margin_dse is not None
        and margin_dse > 0 and margin > 2 * margin_dse
    )
    true_row = comparison.get(true_model, {})
    return {
        "winner_by_elpd": winner_by_elpd,
        "winner_margin": margin,
        "winner_margin_dse": margin_dse,
        "winner_distinguishable": distinguishable,
        "true_model_elpd_diff": (
            float(true_row["elpd_diff"]) if "elpd_diff" in true_row else None
        ),
        "true_model_dse": float(true_row["dse"]) if "dse" in true_row else None,
        # A "clear" recovery: the ELPD winner *is* the true model and it is
        # statistically distinguishable from the runner-up.
        "recovery_clear": (winner_by_elpd == true_model) and distinguishable,
    }


def model_recovery_summary(confusion: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-model and overall recovery metrics for a closed-ended confusion result.

    For each generating model, identify the best-fitting model by posterior and
    by ELPD-LOO, and whether that is the true (generating) model. When the
    inner loop's ``comparison`` table is present, also judge whether each
    recovery is statistically *clear* (the winner is distinguishable from the
    runner-up by ``elpd_diff > 2 * dse``) rather than a near-tie. Aggregate into
    posterior/ELPD accuracy, the mean posterior on the true model, and (when
    comparison data exists) the rate of clear recoveries.
    """
    generating = confusion["generating"]
    if not generating:
        raise ValueError("Confusion result has no generating models to summarize.")

    per_model: List[Dict[str, Any]] = []
    for entry in generating:
        true_model = entry["generating_model"]
        posteriors = entry["posteriors"]
        elpd = entry["elpd_loo"]
        best_by_posterior = max(posteriors, key=posteriors.get)
        best_by_elpd = max(elpd, key=elpd.get)  # higher ELPD-LOO is better
        per_model.append(
            {
                "generating_model": true_model,
                "true_posterior": float(posteriors[true_model]),
                "best_by_posterior": best_by_posterior,
                "best_by_elpd": best_by_elpd,
                "correct_posterior": best_by_posterior == true_model,
                "correct_elpd": best_by_elpd == true_model,
                **_distinguishability(entry, true_model),
            }
        )

    n = len(per_model)
    has_comparison = all(r["winner_distinguishable"] is not None for r in per_model)
    return {
        "n_models": n,
        "posterior_accuracy": sum(r["correct_posterior"] for r in per_model) / n,
        "elpd_accuracy": sum(r["correct_elpd"] for r in per_model) / n,
        "mean_true_posterior": sum(r["true_posterior"] for r in per_model) / n,
        "has_comparison": has_comparison,
        "clear_recovery_rate": (
            sum(bool(r["recovery_clear"]) for r in per_model) / n
            if has_comparison
            else None
        ),
        "per_model": per_model,
    }
