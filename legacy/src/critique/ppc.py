"""
Posterior predictive check (PPC) helper for cognitive model criticism.

For a given model and test statistic, resamples synthetic datasets from the
model and computes an empirical p-value:

    p = #{resamples where T(synthetic) >= T(observed)} / n_samples

Stimuli are derived directly from the responses file(s) — no separate stimuli
argument is needed.

Usage (CLI):
    python3 -m src.critique.ppc \\
        --exp-dir cc_pipeline/projects/subjective_randomness/experiment1 \\
        --model representativeness \\
        --stat-file critique/test_stats/balance_residual.py \\
        --n-samples 500

    # Pool responses across experiments:
    python3 -m src.critique.ppc \\
        --exp-dir cc_pipeline/projects/subjective_randomness/experiment2 \\
        --model representativeness \\
        --stat-file critique/test_stats/balance_residual.py \\
        --responses exp1/data/responses.csv exp2/data/responses.csv \\
        --n-samples 500

Prints a JSON object to stdout:
    {
      "model": "representativeness",
      "stat": "balance_residual",
      "t_observed": 0.123,
      "t_mean_null": 0.045,
      "p_value": 0.032,
      "n_samples": 500,
      "significant_at_0.05": true
    }
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_test_stat(stat_file: Path):
    """Load a test statistic function from a .py file. The file must define a
    callable named 'test_stat(rows: list[dict]) -> float'."""
    spec = importlib.util.spec_from_file_location("_test_stat", stat_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {stat_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "test_stat", None)
    if not callable(fn):
        raise ValueError(f"{stat_file} must define a callable named 'test_stat'")
    return fn


def aggregate_rows(
    response_rows: List[Dict[str, Any]],
    stimulus_col_a: str = "sequence_a",
    stimulus_col_b: str = "sequence_b",
    response_col: str = "chose_left",
) -> List[Dict[str, Any]]:
    """Aggregate response rows by (stimulus_col_a, stimulus_col_b) — mirrors _aggregate_csv."""
    from collections import defaultdict
    key_to_rows: Dict[tuple, list] = defaultdict(list)
    for r in response_rows:
        key = (r[stimulus_col_a], r[stimulus_col_b])
        key_to_rows[key].append(r)
    result = []
    for (sa, sb), group in sorted(key_to_rows.items()):
        lefts = [int(r.get(response_col, 0)) for r in group]
        result.append({
            "sequence_a": sa,
            "sequence_b": sb,
            "chose_left_pct": sum(lefts) / len(lefts) if lefts else 0.0,
            "n": len(lefts),
            "lm_code_translation_list": [r["lm_code_translation"] for r in group if r.get("lm_code_translation")],
        })
    return result


def run_ppc(
    exp_dir: Path,
    model_name: str,
    stat_file: Path,
    n_samples: int = 500,
    theorist_dir: Optional[Path] = None,
    responses_paths: Optional[List[Path]] = None,
    stimulus_col_a: str = "sequence_a",
    stimulus_col_b: str = "sequence_b",
    response_col: str = "chose_left",
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Posterior predictive check for one PyMC model × one test statistic.

    1. Fit the model (MCMC) on the pooled observed responses.
    2. Sample `n_samples` posterior-predictive datasets — one synthetic
       response array per draw, attached to the observed rows so the
       stimulus identifier columns (stimulus_col_a, stimulus_col_b) are
       preserved for aggregation.
    3. Compute the test statistic on each synthetic aggregate; empirical
       one-tailed p-value compares to t_observed.

    `n_samples` must be ≤ chains × draws of the fitted model's posterior;
    raises if more are requested.
    """
    from src.models.pymc_inference import (  # type: ignore
        fit_models_cached,
        make_stim_data,
        observed_response_data,
    )

    exp_dir = Path(exp_dir)
    if theorist_dir is None:
        theorist_dir = exp_dir / "cognitive_models"

    test_stat_fn = load_test_stat(Path(stat_file))

    # Load and pool observed responses (write a single pooled CSV for the fitter)
    if responses_paths:
        observed_rows: List[Dict[str, Any]] = []
        for p in responses_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"responses.csv not found at {p}")
            observed_rows.extend(csv.DictReader(open(p, encoding="utf-8")))
    else:
        responses_path = exp_dir / "data" / "responses.csv"
        if not responses_path.exists():
            raise FileNotFoundError(f"responses.csv not found at {responses_path}")
        observed_rows = list(csv.DictReader(open(responses_path, encoding="utf-8")))
    if not observed_rows:
        raise ValueError("No observed responses; cannot run PPC.")

    observed_agg = aggregate_rows(observed_rows, stimulus_col_a, stimulus_col_b, response_col)
    t_observed = float(test_stat_fn(observed_agg))

    # If multiple response files, write a pooled CSV for the fitter.
    if responses_paths and len(responses_paths) > 1:
        import tempfile
        pooled_path = Path(tempfile.mkstemp(prefix="ppc_pooled_", suffix=".csv")[1])
        fieldnames = list(observed_rows[0].keys())
        with pooled_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(observed_rows)
        fit_csv = pooled_path
    elif responses_paths:
        fit_csv = Path(responses_paths[0])
    else:
        fit_csv = exp_dir / "data" / "responses.csv"

    fits = fit_models_cached(
        [model_name], models_dir=theorist_dir, responses_path=fit_csv, cache_dir=cache_dir,
    )
    fitted = fits[model_name]
    pymc_response_col = observed_response_data(fitted.model)

    # Build stim_data for posterior-predictive sampling: one row per observed trial,
    # so synthetic responses align 1:1 with observed_rows and inherit their
    # stimulus_col_a/b values for aggregation.
    stim_data = make_stim_data(fitted.model, observed_rows)
    synthetic = fitted.sample_synthetic_responses(stim_data, n_datasets=n_samples)
    # shape: (n_samples, n_observed_trials)

    null_values = []
    for i in range(n_samples):
        synth_responses = synthetic[i]
        synthetic_rows = [
            {**observed_rows[j], response_col: int(synth_responses[j])}
            for j in range(len(observed_rows))
        ]
        synthetic_agg = aggregate_rows(synthetic_rows, stimulus_col_a, stimulus_col_b, response_col)
        null_values.append(float(test_stat_fn(synthetic_agg)))

    n_extreme = sum(1 for t in null_values if t >= t_observed)
    p_value = (n_extreme + 1) / (n_samples + 1)

    return {
        "model": model_name,
        "stat": Path(stat_file).stem,
        "t_observed": t_observed,
        "t_mean_null": sum(null_values) / len(null_values) if null_values else 0.0,
        "t_sd_null": (
            (sum((t - sum(null_values) / len(null_values)) ** 2 for t in null_values) / len(null_values)) ** 0.5
            if null_values else 0.0
        ),
        "p_value": p_value,
        "n_samples": n_samples,
        "n_extreme": n_extreme,
        "significant_at_0.05": p_value < 0.05,
        "pymc_response_col": pymc_response_col,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run posterior predictive check for a model × test statistic")
    parser.add_argument("--exp-dir", required=True, help="Path to experimentN/ directory")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--stat-file", required=True, help="Path to test statistic .py file")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of resamples (default: 500)")
    parser.add_argument("--theorist-dir", default=None, help="Override path to cognitive_models/ dir")
    parser.add_argument(
        "--responses", nargs="+", default=None,
        help="Response CSV(s) to use instead of exp-dir/data/responses.csv (pooled if multiple)",
    )
    parser.add_argument("--stimulus-col-a", default="sequence_a")
    parser.add_argument("--stimulus-col-b", default="sequence_b")
    parser.add_argument("--response-col", default="chose_left")
    parser.add_argument("--cache-dir", default=None, help="Directory to persist .nc model fits (optional)")
    args = parser.parse_args()

    theorist_dir = Path(args.theorist_dir) if args.theorist_dir else None
    result = run_ppc(
        exp_dir=Path(args.exp_dir),
        model_name=args.model,
        stat_file=Path(args.stat_file),
        n_samples=args.n_samples,
        theorist_dir=theorist_dir,
        responses_paths=[Path(p) for p in args.responses] if args.responses else None,
        stimulus_col_a=args.stimulus_col_a,
        stimulus_col_b=args.stimulus_col_b,
        response_col=args.response_col,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
