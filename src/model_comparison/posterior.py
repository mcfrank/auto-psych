"""Bayesian model posterior over PyMC cognitive models.

For each model, ELPD-LOO is computed on the pooled observed responses; these
log-scores are softmaxed (with an optional complexity prior) to form the
posterior over models.

    P(model | data) ∝ exp(elpd_loo(model)) * P(model)

With --complexity-prior CONST, log-prior = CONST * complexity(model), where
complexity = non-whitespace, non-comment chars in the model's .py file.
Negative CONST penalises complex models (Occam's razor).

Multiple --responses files are pooled into a single observed set before
fitting (one MCMC run per model on pooled data).

Usage (CLI):
    python3 -m src.model_comparison.posterior \\
        --responses  EXP1/data/responses.csv EXP2/data/responses.csv \\
        --models-dir EXP_DIR/cognitive_models \\
        --out        EXP_DIR/critique/model_posterior.json

Output JSON:
    {
      "posteriors":      {"alternation": 0.82, "representativeness": 0.18},
      "elpd_loo":        {"alternation": -38.1, "representativeness": -45.7},
      "n_trials":        300,
      // if --complexity-prior is non-zero:
      "complexity_prior_const": -0.01,
      "complexities":    {"alternation": 312, "representativeness": 748}
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def model_complexity(model_name: str, models_dir: Path) -> int:
    """Non-whitespace, non-comment character count in `<model_name>.py`."""
    model_file = models_dir / f"{model_name}.py"
    if not model_file.exists():
        return 0
    count = 0
    for line in model_file.read_text(encoding="utf-8").splitlines():
        code_part = line.split("#")[0]
        count += sum(1 for c in code_part if not c.isspace())
    return count


def _pool_response_csvs(paths: List[Path]) -> Path:
    """Concatenate multiple CSVs into one temporary file with a single header.

    Returns the path to the pooled CSV. All inputs must share the same header.
    """
    if len(paths) == 1:
        return paths[0]
    header: Optional[List[str]] = None
    pooled_rows: List[List[str]] = []
    for p in paths:
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            this_header = next(reader)
            if header is None:
                header = this_header
            elif this_header != header:
                raise ValueError(
                    f"Cannot pool {p}: header {this_header} differs from first file's header {header}"
                )
            for row in reader:
                pooled_rows.append(row)
    if header is None:
        raise ValueError("No response files provided")
    tmp = Path(tempfile.mkstemp(prefix="pooled_responses_", suffix=".csv")[1])
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(pooled_rows)
    return tmp


def model_posterior(
    responses_path: Path,
    models_dir: Path,
    *,
    complexity_prior_const: float = 0.0,
    cache_dir: Optional[Path] = None,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """Compute the Bayesian posterior over PyMC cognitive models.

    Fits each model in `<models_dir>/models_manifest.yaml` to the pooled
    responses at `responses_path` (MCMC), scores each by ELPD-LOO, and
    softmaxes the scores (with optional complexity prior) to produce a
    normalized posterior.

    Extra ``fit_kwargs`` (e.g. ``draws``, ``tune``, ``chains``) are forwarded
    to the MCMC fit of every model.

    Returns a dict with `posteriors`, `elpd_loo`, `n_trials`, and (if
    complexity_prior_const != 0) `complexity_prior_const` + `complexities`.
    """
    import yaml  # type: ignore
    from src.models.theorist.loader import get_model_names_from_manifest  # type: ignore
    from src.model_comparison import likelihood as _likelihood  # type: ignore

    responses_path = Path(responses_path)
    models_dir = Path(models_dir)

    manifest_path = models_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"models_manifest.yaml not found at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    model_names = get_model_names_from_manifest(manifest, models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")

    elpd: Dict[str, float] = {
        m: _likelihood.log_likelihood(
            m, responses_path, models_dir, cache_dir=cache_dir, **fit_kwargs
        )
        for m in model_names
    }

    if complexity_prior_const != 0.0:
        complexities: Optional[Dict[str, int]] = {
            m: model_complexity(m, models_dir) for m in model_names
        }
        log_priors = {m: complexity_prior_const * complexities[m] for m in model_names}
    else:
        complexities = None
        log_priors = {m: 0.0 for m in model_names}

    log_scores = {m: elpd[m] + log_priors[m] for m in model_names}
    max_score = max(log_scores.values())
    sum_exp = sum(math.exp(s - max_score) for s in log_scores.values())
    log_norm = max_score + math.log(sum_exp)
    posteriors = {m: round(math.exp(log_scores[m] - log_norm), 6) for m in model_names}

    n_trials = sum(1 for _ in csv.DictReader(responses_path.open(encoding="utf-8")))

    result: Dict[str, Any] = {
        "posteriors": posteriors,
        "elpd_loo": {m: round(elpd[m], 4) for m in model_names},
        "n_trials": n_trials,
    }
    if complexities is not None:
        result["complexity_prior_const"] = complexity_prior_const
        result["complexities"] = complexities
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian posterior over PyMC cognitive models"
    )
    parser.add_argument(
        "--responses", required=True, nargs="+",
        help="Path(s) to responses.csv — multiple files are pooled (must share header)",
    )
    parser.add_argument("--models-dir", required=True, help="Path to cognitive_models/ directory")
    parser.add_argument("--out", default=None, help="Write JSON to this file (default: stdout)")
    parser.add_argument(
        "--complexity-prior", type=float, default=0.0, metavar="CONST",
        help=(
            "Log-prior per model = CONST * complexity, where complexity = non-whitespace "
            "non-comment chars in the model .py file. Negative CONST penalises complex "
            "models (Occam's razor). Default: 0.0 (uniform prior)."
        ),
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Optional directory to persist .nc fits (default: in-process cache only)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.responses]
    for p in paths:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    pooled = _pool_response_csvs(paths)

    result = model_posterior(
        responses_path=pooled,
        models_dir=Path(args.models_dir),
        complexity_prior_const=args.complexity_prior,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
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
            f"elpd_loo={result['elpd_loo'][best]:.1f}, "
            f"n_trials={result['n_trials']}{prior_note})",
            flush=True,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
