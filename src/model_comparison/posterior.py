"""
Bayesian model posterior from individual response data with uniform prior.

With a uniform prior over models, the posterior is simply the normalised
likelihood: P(model | data) ∝ P(data | model). Log-sum-exp is used for
numerical stability. Pass multiple response files to pool data across
experiments before computing the posterior.

Usage (CLI):
    python3 -m src.model_comparison.posterior \\
        --responses  EXP1/data/responses.csv EXP2/data/responses.csv \\
        --models-dir EXP_DIR/cognitive_models \\
        --out        EXP_DIR/critique/model_posterior.json

Prints (and optionally writes) JSON:
    {
      "posteriors":      {"alternation": 0.82, "representativeness": 0.18},
      "log_likelihoods": {"alternation": -38.1, "representativeness": -45.7},
      "n_trials": 300
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def model_posterior(
    response_rows: List[Dict[str, Any]],
    models_dir: Path,
) -> Dict[str, Any]:
    """
    Compute the Bayesian posterior over models given individual response data,
    using a uniform prior over all models in the manifest.

    response_rows: list of dicts with sequence_a, sequence_b, chose_left (0 or 1).
                   Pool rows from multiple experiments before calling.
    models_dir: cognitive_models/ directory for the current experiment.

    Returns a dict with keys: posteriors, log_likelihoods, n_trials.
    """
    import yaml  # type: ignore
    from src.models.loader import get_model_names_from_manifest  # type: ignore
    from src.model_comparison.likelihood import log_likelihood  # type: ignore

    manifest_path = models_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"models_manifest.yaml not found at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    model_names = get_model_names_from_manifest(manifest, models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")

    # Compute log-likelihoods from individual responses
    log_liks: Dict[str, float] = {
        m: log_likelihood(m, response_rows, models_dir) for m in model_names
    }

    # Uniform prior → posterior ∝ likelihood; normalise with log-sum-exp
    max_ll = max(log_liks.values())
    sum_exp = sum(math.exp(ll - max_ll) for ll in log_liks.values())
    log_norm = max_ll + math.log(sum_exp)
    posteriors = {m: round(math.exp(log_liks[m] - log_norm), 6) for m in model_names}

    return {
        "posteriors": posteriors,
        "log_likelihoods": {m: round(log_liks[m], 4) for m in model_names},
        "n_trials": len(response_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian model posterior from individual response data (uniform prior)"
    )
    parser.add_argument(
        "--responses", required=True, nargs="+",
        help="Path(s) to responses.csv — multiple files are pooled",
    )
    parser.add_argument("--models-dir", required=True, help="Path to cognitive_models/ directory")
    parser.add_argument("--out", default=None, help="Write JSON to this file (default: stdout)")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for p in args.responses:
        path = Path(p)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        rows.extend(csv.DictReader(path.open(encoding="utf-8")))

    result = model_posterior(response_rows=rows, models_dir=Path(args.models_dir))

    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        best = max(result["posteriors"], key=lambda m: result["posteriors"][m])
        print(
            f"Wrote model_posterior.json — best: {best} "
            f"(posterior={result['posteriors'][best]:.3f}, "
            f"log_lik={result['log_likelihoods'][best]:.1f}, "
            f"n_trials={result['n_trials']})",
            flush=True,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
