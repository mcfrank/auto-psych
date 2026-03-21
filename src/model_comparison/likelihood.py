"""
Log-likelihood of individual response data under a cognitive model.

For each trial, the model predicts P(chose_left | stimulus). The log-likelihood
contribution is simply log P(observed_response | model, stimulus):

    log p_left   if chose_left == 1
    log p_right  if chose_left == 0

Summed across all trials. No aggregation needed — the likelihood comes directly
from the model's predictions on individual responses.

Usage (CLI):
    python3 -m src.model_comparison.likelihood \\
        --responses  EXP_DIR/data/responses.csv \\
        --model      alternation \\
        --models-dir EXP_DIR/cognitive_models

Prints JSON:
    {
      "model": "alternation",
      "log_likelihood": -42.31,
      "n_trials": 150
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

RESPONSE_OPTIONS = ["left", "right"]
_CLIP = 1e-9  # clip predicted probability away from 0/1


def log_likelihood(
    model_name: str,
    response_rows: List[Dict[str, Any]],
    models_dir: Path,
    model_registry: Optional[Dict] = None,
) -> float:
    """
    Compute log P(responses | model) by summing log P(response_i | model, stimulus_i)
    across all individual trials.

    response_rows: list of dicts with sequence_a, sequence_b, chose_left (0 or 1)
    models_dir: cognitive_models/ directory
    model_registry: optional {name: callable} dict (e.g. ground-truth models)
    """
    from src.models.randomness import get_model_predictions  # type: ignore

    # Cache predictions per stimulus to avoid redundant model calls
    pred_cache: Dict[tuple, float] = {}
    ll = 0.0

    for row in response_rows:
        stimulus = (row["sequence_a"], row["sequence_b"])
        if stimulus not in pred_cache:
            if model_registry and model_name in model_registry:
                preds = {model_name: model_registry[model_name](stimulus, RESPONSE_OPTIONS)}
            else:
                preds = get_model_predictions(stimulus, RESPONSE_OPTIONS, [model_name], models_dir)
            p = preds.get(model_name, {}).get("left", 0.5)
            pred_cache[stimulus] = max(_CLIP, min(1 - _CLIP, p))

        p_left = pred_cache[stimulus]
        chose_left = int(row["chose_left"])
        ll += math.log(p_left) if chose_left else math.log(1 - p_left)

    return ll


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute log-likelihood of individual responses under a model"
    )
    parser.add_argument("--responses", required=True, help="Path to responses.csv")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--models-dir", required=True, help="Path to cognitive_models/ directory")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Error: {responses_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(responses_path.open(encoding="utf-8")))

    ll = log_likelihood(
        model_name=args.model,
        response_rows=rows,
        models_dir=Path(args.models_dir),
    )

    print(json.dumps({
        "model": args.model,
        "log_likelihood": round(ll, 4),
        "n_trials": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
