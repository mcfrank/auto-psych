"""ELPD-LOO of a PyMC cognitive model on observed responses.

Each cognitive model is a PyMC model that defines, at module level, `pm.Model()`
with `pm.Data` containers (one per CSV column the model needs) and a Bernoulli
(or Categorical) likelihood for the observed response. This module fits the
model via MCMC, then returns the cross-validated ELPD (`arviz.loo`) — a more
honest per-model score for Bayesian model comparison than summed log-likelihood
at the posterior mean.

Usage (CLI):
    python3 -m src.model_comparison.likelihood \\
        --responses  EXP_DIR/data/responses.csv \\
        --model      alternation \\
        --models-dir EXP_DIR/cognitive_models

Prints JSON:
    {
      "model": "alternation",
      "elpd_loo": -42.31,
      "n_trials": 150
    }
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def log_likelihood(
    model_name: str,
    responses_path: Path,
    models_dir: Path,
    *,
    cache_dir: Optional[Path] = None,
    **fit_kwargs: object,
) -> float:
    """Return ELPD-LOO of `model_name` fit on `responses_path`.

    Higher is better. Same units as summed pointwise log-likelihood, so
    softmaxing across models for Bayesian model comparison is meaningful.
    Cached in-process per (model file content, CSV content); if `cache_dir`
    is given, also persisted to `<cache_dir>/<name>.<fingerprint>.nc`.
    Extra ``fit_kwargs`` (e.g. ``draws``, ``tune``, ``chains``) are forwarded
    to the MCMC fit.
    """
    from src.models.pymc_inference import fit_models_cached  # type: ignore

    fits = fit_models_cached(
        [model_name],
        models_dir=models_dir,
        responses_path=responses_path,
        cache_dir=cache_dir,
        **fit_kwargs,
    )
    return fits[model_name].elpd_loo()


@dataclass
class Args:
    """ELPD-LOO of a PyMC cognitive model on observed responses."""

    responses: Path
    """Path to responses.csv."""
    model: str
    """Model name."""
    models_dir: Path
    """Path to the cognitive_models/ directory."""
    cache_dir: Optional[Path] = None
    """Optional directory to persist .nc fits."""


def main(args: Args) -> None:
    if not args.responses.exists():
        print(f"Error: {args.responses} not found", file=sys.stderr)
        sys.exit(1)

    elpd = log_likelihood(
        model_name=args.model,
        responses_path=args.responses,
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
    )
    n_trials = sum(1 for _ in csv.DictReader(args.responses.open(encoding="utf-8")))

    print(json.dumps({
        "model": args.model,
        "elpd_loo": round(elpd, 4),
        "n_trials": n_trials,
    }, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
