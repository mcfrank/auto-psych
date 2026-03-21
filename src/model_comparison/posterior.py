"""
Bayesian model posterior from individual response data.

With a uniform prior the posterior is the normalised likelihood:
P(model | data) ∝ P(data | model).

With --complexity-prior CONST the prior is:
P(model) ∝ exp(CONST * complexity(model))
where complexity = non-blank, non-comment lines in the model's .py file.
Negative CONST penalises complex models (Occam's razor); positive CONST
favours richer models. Log-sum-exp is used throughout for numerical stability.
Pass multiple response files to pool data across experiments.

Usage (CLI):
    python3 -m src.model_comparison.posterior \\
        --responses  EXP1/data/responses.csv EXP2/data/responses.csv \\
        --models-dir EXP_DIR/cognitive_models \\
        --out        EXP_DIR/critique/model_posterior.json

    # With complexity prior (penalise complex models):
    python3 -m src.model_comparison.posterior \\
        --responses  EXP1/data/responses.csv \\
        --models-dir EXP_DIR/cognitive_models \\
        --complexity-prior -0.01 \\
        --out        EXP_DIR/critique/model_posterior.json

Prints (and optionally writes) JSON:
    {
      "posteriors":      {"alternation": 0.82, "representativeness": 0.18},
      "log_likelihoods": {"alternation": -38.1, "representativeness": -45.7},
      "n_trials": 300
      // if --complexity-prior is non-zero:
      "complexity_prior_const": -0.01,
      "complexities": {"alternation": 42, "representativeness": 87}
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def model_complexity(model_name: str, models_dir: Path) -> int:
    """Count non-blank, non-comment lines in a model's .py file."""
    model_file = models_dir / f"{model_name}.py"
    if not model_file.exists():
        return 0
    count = 0
    for line in model_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def model_posterior(
    response_rows: List[Dict[str, Any]],
    models_dir: Path,
    complexity_prior_const: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute the Bayesian posterior over models given individual response data.

    response_rows: list of dicts with sequence_a, sequence_b, chose_left (0 or 1).
                   Pool rows from multiple experiments before calling.
    models_dir: cognitive_models/ directory for the current experiment.
    complexity_prior_const: if non-zero, log-prior for each model is
        complexity_prior_const * complexity(model), where complexity is the
        number of non-blank, non-comment lines in the model's .py file.
        Negative values penalise complex models; 0.0 gives a uniform prior.

    Returns a dict with keys: posteriors, log_likelihoods, n_trials.
    If complexity_prior_const != 0, also includes complexity_prior_const and complexities.
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

    # Compute log-priors
    if complexity_prior_const != 0.0:
        complexities: Optional[Dict[str, int]] = {
            m: model_complexity(m, models_dir) for m in model_names
        }
        log_priors = {m: complexity_prior_const * complexities[m] for m in model_names}
    else:
        complexities = None
        log_priors = {m: 0.0 for m in model_names}

    # Posterior ∝ likelihood × prior; normalise with log-sum-exp
    log_scores = {m: log_liks[m] + log_priors[m] for m in model_names}
    max_score = max(log_scores.values())
    sum_exp = sum(math.exp(s - max_score) for s in log_scores.values())
    log_norm = max_score + math.log(sum_exp)
    posteriors = {m: round(math.exp(log_scores[m] - log_norm), 6) for m in model_names}

    result: Dict[str, Any] = {
        "posteriors": posteriors,
        "log_likelihoods": {m: round(log_liks[m], 4) for m in model_names},
        "n_trials": len(response_rows),
    }
    if complexities is not None:
        result["complexity_prior_const"] = complexity_prior_const
        result["complexities"] = complexities
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian model posterior from individual response data"
    )
    parser.add_argument(
        "--responses", required=True, nargs="+",
        help="Path(s) to responses.csv — multiple files are pooled",
    )
    parser.add_argument("--models-dir", required=True, help="Path to cognitive_models/ directory")
    parser.add_argument("--out", default=None, help="Write JSON to this file (default: stdout)")
    parser.add_argument(
        "--complexity-prior", type=float, default=0.0, metavar="CONST",
        help=(
            "Log-prior per model = CONST * complexity, where complexity = non-blank "
            "non-comment lines in the model .py file. Negative CONST penalises complex "
            "models (Occam's razor). Default: 0.0 (uniform prior)."
        ),
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for p in args.responses:
        path = Path(p)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        rows.extend(csv.DictReader(path.open(encoding="utf-8")))

    result = model_posterior(
        response_rows=rows,
        models_dir=Path(args.models_dir),
        complexity_prior_const=args.complexity_prior,
    )

    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        best = max(result["posteriors"], key=lambda m: result["posteriors"][m])
        prior_note = (
            f", complexity_prior_const={result['complexity_prior_const']}"
            if "complexity_prior_const" in result else ""
        )
        print(
            f"Wrote model_posterior.json — best: {best} "
            f"(posterior={result['posteriors'][best]:.3f}, "
            f"log_lik={result['log_likelihoods'][best]:.1f}, "
            f"n_trials={result['n_trials']}{prior_note})",
            flush=True,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
