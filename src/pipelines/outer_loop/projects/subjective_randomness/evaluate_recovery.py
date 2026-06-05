#!/usr/bin/env python3
"""Evaluate hidden-ground-truth recovery for subjective randomness.

The outer loop can generate data from a callable ground-truth model that is not
listed among the seed PyMC families. After each experiment exports an
``inner_loop_model`` into ``cognitive_models/``, this script refits that exported
model on the experiment's pooled feature CSV and compares its heldout
``p_left`` predictions to the hidden ground truth.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.models.pymc_inference import fit_model, load_pymc_model, make_stim_data
from src.pipelines.outer_loop.orchestrator import (
    experiment_dir,
    get_ground_truth_models,
    outer_project_dir,
)

RESPONSE_OPTIONS = ["left", "right"]
Stimulus = Tuple[str, str]


def _load_featurizer():
    path = Path(__file__).resolve().parent / "preprocess.py"
    spec = importlib.util.spec_from_file_location("_subjective_randomness_preprocess", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load featurizer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.featurize_stimulus


featurize_stimulus = _load_featurizer()


def parse_experiments(value: str) -> List[int]:
    """Parse ``N``, ``A-B``, or comma-separated experiment ids."""
    value = value.strip()
    if "-" in value and "," not in value:
        start_s, end_s = value.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start < 1 or end < start:
            raise ValueError("Experiment range must be A-B with B >= A >= 1")
        return list(range(start, end + 1))
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(n < 1 for n in out):
        raise ValueError("Experiments must be positive integers")
    return out


def random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("HT") for _ in range(length))


def make_heldout_stimuli(
    *,
    n_pairs: int,
    min_length: int,
    max_length: int,
    seed: int,
) -> List[Stimulus]:
    """Create deterministic heldout H/T sequence pairs."""
    rng = random.Random(seed)
    stimuli: List[Stimulus] = []
    seen: set[Stimulus] = set()
    while len(stimuli) < n_pairs:
        len_a = rng.randint(min_length, max_length)
        len_b = rng.randint(min_length, max_length)
        pair = (random_sequence(rng, len_a), random_sequence(rng, len_b))
        if pair not in seen and pair[0] != pair[1]:
            stimuli.append(pair)
            seen.add(pair)
    return stimuli


def ground_truth_p_left(
    fn: Callable[[Stimulus, List[str]], Dict[str, float]],
    stimuli: List[Stimulus],
) -> np.ndarray:
    return np.array(
        [float(fn(stimulus, RESPONSE_OPTIONS)[RESPONSE_OPTIONS[0]]) for stimulus in stimuli],
        dtype="float64",
    )


def feature_rows(stimuli: List[Stimulus]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sequence_a, sequence_b in stimuli:
        row: Dict[str, Any] = {
            "sequence_a": sequence_a,
            "sequence_b": sequence_b,
            "chose_left": 0,
        }
        row.update(featurize_stimulus(sequence_a, sequence_b))
        rows.append(row)
    return rows


def metrics(p_true: np.ndarray, p_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-9
    pred = np.clip(p_pred.astype("float64"), eps, 1.0 - eps)
    true = np.clip(p_true.astype("float64"), eps, 1.0 - eps)
    return {
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mae": float(np.mean(np.abs(pred - true))),
        "cross_entropy": float(-np.mean(true * np.log(pred) + (1.0 - true) * np.log(1.0 - pred))),
        "kl": float(
            np.mean(
                true * np.log(true / pred)
                + (1.0 - true) * np.log((1.0 - true) / (1.0 - pred))
            )
        ),
    }


def posterior_summary(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "model_loop" / "model_posterior.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    post = data.get("posteriors") or {}
    if not post:
        return data
    data["best_model"] = max(post, key=post.get)
    data["best_posterior"] = post[data["best_model"]]
    return data


def evaluate_experiment(
    *,
    project_id: str,
    exp_num: int,
    p_true: np.ndarray,
    rows: List[Dict[str, Any]],
    draws: int,
    tune: int,
    chains: int,
) -> Dict[str, Any]:
    exp_dir = experiment_dir(project_id, exp_num)
    models_dir = exp_dir / "cognitive_models"
    responses_path = exp_dir / "model_loop" / "responses.csv"
    exported_model = models_dir / "inner_loop_model.py"

    result: Dict[str, Any] = {
        "experiment": exp_num,
        "experiment_dir": str(exp_dir),
        "model": "inner_loop_model",
        "posterior_summary": posterior_summary(exp_dir),
    }
    if not exported_model.exists():
        result["error"] = f"Missing exported model: {exported_model}"
        return result
    if not responses_path.exists():
        result["error"] = f"Missing feature responses CSV: {responses_path}"
        return result

    model = load_pymc_model("inner_loop_model", models_dir)
    stim_data = make_stim_data(model, rows)
    fitted = fit_model(
        "inner_loop_model",
        models_dir,
        responses_path,
        cache_dir=exp_dir / "model_loop" / "evaluation_cache",
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
    )
    p_pred = np.asarray(fitted.predict_p_left(stim_data), dtype="float64")
    result.update(metrics(p_true, p_pred))
    result["mean_p_true"] = float(np.mean(p_true))
    result["mean_p_pred"] = float(np.mean(p_pred))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how well exported outer-loop models recover a hidden ground truth."
    )
    parser.add_argument("--project", default="subjective_randomness")
    parser.add_argument("--ground-truth-model", required=True)
    parser.add_argument("--experiments", default="1-3", help="N, A-B, or comma-separated ids")
    parser.add_argument("--n-heldout", type=int, default=200)
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    registry = get_ground_truth_models(args.project)
    if args.ground_truth_model not in registry:
        allowed = sorted(registry)
        raise SystemExit(
            f"Unknown ground-truth model {args.ground_truth_model!r}. Allowed: {allowed}"
        )

    exp_ids = parse_experiments(args.experiments)
    stimuli = make_heldout_stimuli(
        n_pairs=args.n_heldout,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )
    p_true = ground_truth_p_left(registry[args.ground_truth_model], stimuli)
    rows = feature_rows(stimuli)

    results = [
        evaluate_experiment(
            project_id=args.project,
            exp_num=exp_num,
            p_true=p_true,
            rows=rows,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
        )
        for exp_num in exp_ids
    ]
    payload = {
        "project": args.project,
        "project_dir": str(outer_project_dir(args.project)),
        "ground_truth_model": args.ground_truth_model,
        "n_heldout": len(stimuli),
        "heldout_seed": args.seed,
        "experiments": results,
    }

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)

    print("\nRecovery summary:")
    for row in results:
        if row.get("error"):
            print(f"  experiment {row['experiment']}: {row['error']}")
            continue
        best = row.get("posterior_summary", {}).get("best_model", "?")
        print(
            f"  experiment {row['experiment']}: "
            f"rmse={row['rmse']:.4f} mae={row['mae']:.4f} "
            f"kl={row['kl']:.4f} model_loop_best={best}"
        )


if __name__ == "__main__":
    main()
