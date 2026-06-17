"""Posterior-predictive critique of a fitted PyMC cognitive model (CriticAL).

Given an incumbent model that has already been fit to the pooled responses, this
module evaluates LLM-proposed *test statistics* — each a Python function
``test_statistic(df) -> float`` over a trial-level dataframe — under a
posterior-predictive check:

1. ``build_critique_frames`` builds the observed ("human") dataframe and
   ``n_replicates`` model dataframes. Each model dataframe is the observed
   dataframe with its response column overwritten by one posterior-predictive
   draw of the response (``FittedModel.sample_synthetic_responses``). Feature
   columns are identical across all frames, so a statistic that conditions on
   stimulus features sees the same design under the model and the data.
2. ``evaluate_test_statistic`` computes the statistic on the observed frame and
   on every replicate, then forms the two-sided empirical p-value

       p = min(1, 2 · min((n_ge + 1)/(n + 1), (n_le + 1)/(n + 1)))

   (the plus-one keeps finite Monte-Carlo samples from yielding an exact zero)
   and a z-score of the observed value against the replicate distribution.
3. ``fdr_adjust`` applies Benjamini–Hochberg FDR across the proposed statistics
   — exploratory discrepancy screening, where FDR is far more powerful than
   Bonferroni.

A statistic whose FDR-adjusted p-value is at or below ``significance_alpha`` is a
*significant discrepancy*: concrete evidence of how the incumbent model fails to
reproduce the data, which the next round of candidate models should address.

This is the statistical core only. The critique agent writes the test-statistic
files and the natural-language synthesis; this module computes the evidence.

CLI (run by the critique agent over a directory of test-statistic files)::

    python3 -m src.critique.ppc \\
        --responses      model_loop/responses.csv \\
        --model          bayesian_fair_coin \\
        --models-dir     model_loop/models \\
        --test-stats-dir iter_0/critique/test_stats \\
        --out            iter_0/critique/ppc_results.json \\
        --cache-dir      .mcmc_cache
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────


@dataclass
class TestStatistic:
    """One LLM-proposed test statistic: a ``test_statistic(df) -> float`` function."""

    __test__ = False  # not a pytest test class despite the "Test" prefix

    name: str
    code: str
    description: str = ""


@dataclass
class TestStatisticResult:
    """The outcome of evaluating one test statistic under the PPC.

    ``p_value`` is the two-sided empirical p-value; ``p_value_one_sided`` is the
    upper-tail value (how often the model meets or exceeds the observed). Both
    are unadjusted; ``p_value_adjusted`` is set by :func:`fdr_adjust`. ``error``
    is non-None when the statistic's code raised or returned a non-finite value,
    in which case the p-values are NaN and the statistic is excluded from FDR.
    """

    __test__ = False  # not a pytest test class despite the "Test" prefix

    test_statistic: TestStatistic
    t_observed: float
    t_null: List[float]
    p_value: float
    p_value_adjusted: float
    p_value_one_sided: float = float("nan")
    z_score: float = float("nan")
    p_value_is_floor: bool = False
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Test-statistic loading and execution
# ─────────────────────────────────────────────


_NAME_RE = re.compile(r"^#\s*name:\s*(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"^#\s*description:\s*(.+?)\s*$", re.MULTILINE)


def load_test_statistic_file(path: Path) -> TestStatistic:
    """Load a ``test_statistic(df)`` from a ``.py`` file.

    The file must define a callable named ``test_statistic``. The optional
    ``# name:`` / ``# description:`` header comments (the format the critique
    agent writes) populate the metadata; ``name`` defaults to the file stem.
    """
    path = Path(path)
    code = path.read_text(encoding="utf-8")
    if "def test_statistic" not in code:
        raise ValueError(
            f"{path} must define a function named 'test_statistic(df)'"
        )
    name_match = _NAME_RE.search(code)
    desc_match = _DESC_RE.search(code)
    name = name_match.group(1).strip() if name_match else path.stem
    description = desc_match.group(1).strip() if desc_match else ""
    return TestStatistic(name=name, code=code, description=description)


def _execute_test_statistic(code: str, df: Any) -> float:
    """Execute test-statistic ``code`` against a dataframe and return the scalar.

    The code runs with ``np``, ``pd``, ``math`` and ``json`` in scope (the same
    surface the critique agent is told it may use). The dataframe is copied so a
    statistic that mutates it cannot leak into the next evaluation.
    """
    import pandas as pd  # local import: pandas is only needed at evaluation time

    namespace: Dict[str, Any] = {
        "np": np,
        "pd": pd,
        "math": math,
        "json": json,
        "__builtins__": __builtins__,
    }
    exec(code, namespace)  # noqa: S102 — executing agent-authored statistic code
    fn = namespace.get("test_statistic")
    if not callable(fn):
        raise ValueError("code must define a callable named 'test_statistic'")
    return float(fn(df.copy()))


# ─────────────────────────────────────────────
# Posterior-predictive evaluation
# ─────────────────────────────────────────────


def evaluate_test_statistic(
    test_statistic: TestStatistic,
    human_df: Any,
    model_dfs: List[Any],
) -> TestStatisticResult:
    """Evaluate one statistic on the observed data and the PPC replicates.

    Returns a :class:`TestStatisticResult` with the observed value, the null
    distribution over replicates, the two-sided empirical p-value, and a z-score.
    A statistic whose code raises, or which is non-finite anywhere, returns a
    result with ``error`` set and NaN p-values (excluded from FDR downstream).
    """
    try:
        t_obs = _execute_test_statistic(test_statistic.code, human_df)
        t_null = [_execute_test_statistic(test_statistic.code, df) for df in model_dfs]
    except Exception as exc:  # an agent-authored statistic that does not run
        return TestStatisticResult(
            test_statistic=test_statistic,
            t_observed=float("nan"),
            t_null=[],
            p_value=float("nan"),
            p_value_adjusted=float("nan"),
            error=f"{type(exc).__name__}: {exc}",
        )

    t_null_arr = np.asarray(t_null, dtype=float)
    if not np.all(np.isfinite(t_null_arr)) or not math.isfinite(t_obs):
        return TestStatisticResult(
            test_statistic=test_statistic,
            t_observed=t_obs,
            t_null=list(t_null_arr),
            p_value=float("nan"),
            p_value_adjusted=float("nan"),
            error="non-finite test statistic value",
        )

    n = len(t_null_arr)
    n_ge = int(np.sum(t_null_arr >= t_obs))
    n_le = int(np.sum(t_null_arr <= t_obs))
    # Plus-one correction: finite Monte-Carlo samples never give an exact zero
    # p-value, and this defines the floor when the observed value falls outside
    # every replicate.
    p_one = (n_ge + 1) / (n + 1)
    p_two = min(1.0, 2.0 * min((n_ge + 1) / (n + 1), (n_le + 1) / (n + 1)))

    null_mean = float(t_null_arr.mean())
    null_std = float(t_null_arr.std())
    if null_std > 0.0:
        z_score = (t_obs - null_mean) / null_std
    elif t_obs == null_mean:
        z_score = 0.0
    else:
        # Degenerate (deterministic) reference: any discrepancy is infinitely far.
        z_score = math.inf if t_obs > null_mean else -math.inf

    return TestStatisticResult(
        test_statistic=test_statistic,
        t_observed=t_obs,
        t_null=list(t_null_arr),
        p_value=p_two,
        p_value_adjusted=p_two,  # adjusted in place by fdr_adjust
        p_value_one_sided=p_one,
        z_score=z_score,
        p_value_is_floor=min(n_ge, n_le) == 0,
    )


def fdr_adjust(results: List[TestStatisticResult]) -> List[TestStatisticResult]:
    """In-place Benjamini–Hochberg FDR adjustment over the valid results.

    FDR control is far more powerful than Bonferroni when many statistics are
    screened — appropriate here, since this is exploratory discrepancy screening
    rather than confirmatory testing. Errored / non-finite results get NaN.
    """
    for r in results:
        if r.error is not None or math.isnan(r.p_value):
            r.p_value_adjusted = float("nan")

    valid = [r for r in results if r.error is None and not math.isnan(r.p_value)]
    m = len(valid)
    if m == 0:
        return results

    ordered = sorted(valid, key=lambda r: r.p_value)
    running_min = 1.0
    for rank in range(m, 0, -1):
        r = ordered[rank - 1]
        running_min = min(running_min, r.p_value * m / rank)
        r.p_value_adjusted = running_min
    return results


# ─────────────────────────────────────────────
# Frame building (PyMC posterior-predictive replicates)
# ─────────────────────────────────────────────


def build_critique_frames(
    fitted: Any,
    responses_path: Path,
    *,
    n_replicates: int,
    seed: int = 42,
) -> Tuple[Any, List[Any]]:
    """Build the observed frame and the posterior-predictive replicate frames.

    ``fitted`` is a :class:`src.models.pymc_inference.FittedModel`. The observed
    frame is the full responses CSV. Each replicate frame is a copy whose
    observed-response column is overwritten with one posterior-predictive draw
    of the response, so a test statistic conditions on the *same* feature columns
    under the model and the data and only the responses differ.

    ``n_replicates`` is capped at the fitted posterior's draw count (chains ×
    draws); a smaller pool is used with a printed note rather than an error.
    """
    import pandas as pd

    from src.models.pymc_inference import make_stim_data, observed_response_data

    responses_path = Path(responses_path)
    human_df = pd.read_csv(responses_path)
    if human_df.empty:
        raise ValueError(f"No response rows in {responses_path}")

    response_col = observed_response_data(fitted.model)
    if response_col not in human_df.columns:
        raise ValueError(
            f"Response column {response_col!r} (the model's observed pm.Data) is not "
            f"in {responses_path}; columns: {list(human_df.columns)}"
        )

    rows = human_df.to_dict("records")
    stim_data = make_stim_data(fitted.model, rows)

    capacity = _posterior_capacity(fitted)
    n_use = n_replicates
    if capacity is not None and n_replicates > capacity:
        print(
            f"  [critique] requested {n_replicates} PPC replicates but the posterior "
            f"only holds {capacity} draws; using {capacity}",
            flush=True,
        )
        n_use = capacity

    synthetic = fitted.sample_synthetic_responses(
        stim_data, n_datasets=n_use, seed=seed
    )
    synthetic = np.asarray(synthetic)
    if synthetic.shape[1] != len(human_df):
        raise ValueError(
            f"PPC returned {synthetic.shape[1]} responses per replicate but the data "
            f"has {len(human_df)} trials"
        )

    model_dfs: List[Any] = []
    for i in range(synthetic.shape[0]):
        rep = human_df.copy()
        rep[response_col] = synthetic[i]
        model_dfs.append(rep)
    return human_df, model_dfs


def _posterior_capacity(fitted: Any) -> Optional[int]:
    """Number of posterior draws available for PPC (chains × draws), or None.

    Returns None when the fit carries no idata (e.g. a test stub), in which case
    the caller trusts the requested replicate count.
    """
    idata = getattr(fitted, "idata", None)
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None
    return int(posterior.sizes["chain"]) * int(posterior.sizes["draw"])


# ─────────────────────────────────────────────
# Directory evaluation (what the agent / CLI calls)
# ─────────────────────────────────────────────


def _result_to_dict(res: TestStatisticResult, alpha: float) -> Dict[str, Any]:
    """JSON-serialisable summary of a result (the full null vector is omitted)."""
    t_null = np.asarray(res.t_null, dtype=float) if res.t_null else np.array([])
    significant = (
        res.error is None
        and not math.isnan(res.p_value_adjusted)
        and res.p_value_adjusted <= alpha
    )
    return {
        "name": res.test_statistic.name,
        "description": res.test_statistic.description,
        "t_observed": res.t_observed,
        "null_mean": float(t_null.mean()) if t_null.size else float("nan"),
        "null_std": float(t_null.std()) if t_null.size else float("nan"),
        "n_replicates": int(t_null.size),
        "z_score": res.z_score,
        "p_value": res.p_value,
        "p_value_one_sided": res.p_value_one_sided,
        "p_value_adjusted": res.p_value_adjusted,
        "p_value_is_floor": res.p_value_is_floor,
        "significant": bool(significant),
        "error": res.error,
        "code": res.test_statistic.code,
    }


def evaluate_test_stat_dir(
    fitted: Any,
    responses_path: Path,
    test_stats_dir: Path,
    *,
    n_replicates: int,
    seed: int = 42,
    significance_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate every ``test_stats_dir/*.py`` statistic under the PPC.

    Builds the observed + replicate frames once, runs every statistic, applies
    Benjamini–Hochberg FDR, and returns a JSON-serialisable dict with one entry
    per statistic (sorted most-discrepant first) plus a significance count. Fails
    loudly if the directory holds no test-statistic files.
    """
    test_stats_dir = Path(test_stats_dir)
    stat_files = sorted(test_stats_dir.glob("*.py"))
    if not stat_files:
        raise ValueError(f"No test-statistic .py files in {test_stats_dir}")

    human_df, model_dfs = build_critique_frames(
        fitted, responses_path, n_replicates=n_replicates, seed=seed
    )

    statistics = [load_test_statistic_file(p) for p in stat_files]
    results = [evaluate_test_statistic(ts, human_df, model_dfs) for ts in statistics]
    fdr_adjust(results)

    rows = [_result_to_dict(r, significance_alpha) for r in results]
    # Most discrepant first: significant before not, then by |z|.
    rows.sort(
        key=lambda d: (
            not d["significant"],
            -(abs(d["z_score"]) if math.isfinite(d["z_score"]) else math.inf),
        )
    )
    return {
        "model": fitted.name,
        "n_test_statistics": len(rows),
        "n_replicates": len(model_dfs),
        "significance_alpha": significance_alpha,
        "n_significant": sum(1 for d in rows if d["significant"]),
        "results": rows,
    }


def run_ppc_for_model(
    model_name: str,
    models_dir: Path,
    responses_path: Path,
    test_stats_dir: Path,
    *,
    cache_dir: Optional[Path] = None,
    n_replicates: int = 200,
    seed: int = 42,
    significance_alpha: float = 0.05,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit (or reuse a cached fit of) ``model_name`` and run the PPC critique.

    The fit reuses the same MCMC cache the inner loop wrote, so when called right
    after the inner loop scored the model set this adds no new sampling.
    """
    from src.models.pymc_inference import fit_models_cached

    fit_kwargs = fit_kwargs or {}
    fits = fit_models_cached(
        [model_name],
        models_dir=Path(models_dir),
        responses_path=Path(responses_path),
        cache_dir=cache_dir,
        **fit_kwargs,
    )
    return evaluate_test_stat_dir(
        fits[model_name],
        responses_path,
        test_stats_dir,
        n_replicates=n_replicates,
        seed=seed,
        significance_alpha=significance_alpha,
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


@dataclass
class Args:
    """Run a posterior-predictive critique of one fitted PyMC model."""

    responses: Path
    """Path to the pooled responses CSV the model was fit on."""
    model: str
    """Incumbent model name (``<model>.py`` must exist in --models-dir)."""
    models_dir: Path
    """Directory holding the model set (the inner loop's ``model_loop/models``)."""
    test_stats_dir: Path
    """Directory of ``test_statistic(df)`` ``.py`` files the agent wrote."""
    out: Optional[Path] = None
    """Write the results JSON here (default: stdout)."""
    cache_dir: Optional[Path] = None
    """MCMC fit cache to reuse (share the inner loop's so no refit is needed)."""
    n_replicates: int = 200
    """Posterior-predictive replicates forming each statistic's null distribution."""
    significance_alpha: float = 0.05
    """FDR-adjusted threshold for flagging a statistic as a significant discrepancy."""
    seed: int = 42
    """Seed for posterior-predictive sampling."""
    draws: int = 2000
    """MCMC draws (only used if the fit is not already cached)."""
    tune: int = 2000
    """MCMC tuning steps (only used if the fit is not already cached)."""
    chains: int = 4
    """MCMC chains (only used if the fit is not already cached)."""


def main(args: Args) -> None:
    if not args.responses.exists():
        print(f"Error: responses not found: {args.responses}", file=sys.stderr)
        sys.exit(1)

    result = run_ppc_for_model(
        args.model,
        args.models_dir,
        args.responses,
        args.test_stats_dir,
        cache_dir=args.cache_dir,
        n_replicates=args.n_replicates,
        seed=args.seed,
        significance_alpha=args.significance_alpha,
        fit_kwargs={"draws": args.draws, "tune": args.tune, "chains": args.chains},
    )

    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(
            f"Wrote {args.out} — {result['n_significant']}/{result['n_test_statistics']} "
            f"test statistics significant at FDR α={result['significance_alpha']} "
            f"({result['n_replicates']} PPC replicates)",
            flush=True,
        )
    else:
        print(output)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
