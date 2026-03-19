"""
Posterior predictive check (PPC) helper for cognitive model criticism.

For a given model and test statistic, resamples synthetic datasets from the
model and computes an empirical p-value:

    p = #{resamples where T(synthetic) >= T(observed)} / n_samples

Usage (CLI):
    python3 -m src.critique.ppc \\
        --exp-dir cc_pipeline/projects/subjective_randomness/experiment1 \\
        --model representativeness \\
        --stat-file critique/test_stats/balance_residual.py \\
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


def aggregate_rows(response_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate response rows by (sequence_a, sequence_b) — mirrors _aggregate_csv."""
    from collections import defaultdict
    key_to_lefts: Dict[tuple, list] = defaultdict(list)
    for r in response_rows:
        key = (r["sequence_a"], r["sequence_b"])
        key_to_lefts[key].append(int(r.get("chose_left", 0)))
    result = []
    for (sa, sb), lefts in sorted(key_to_lefts.items()):
        result.append({
            "sequence_a": sa,
            "sequence_b": sb,
            "chose_left_pct": sum(lefts) / len(lefts) if lefts else 0.0,
            "n": len(lefts),
        })
    return result


def run_ppc(
    exp_dir: Path,
    model_name: str,
    stat_file: Path,
    n_samples: int = 500,
    theorist_dir: Optional[Path] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a posterior predictive check for one model × one test statistic.
    The model is assumed to be a theorist model in theorist_dir (cognitive_models/).
    Pass project_id if the model might be in the project's ground_truth_models.py instead.
    Returns a dict with t_observed, p_value, etc.
    """
    from src.agents.collect import _generate_from_models  # type: ignore

    exp_dir = Path(exp_dir)
    if theorist_dir is None:
        theorist_dir = exp_dir / "cognitive_models"

    # Resolve model registry: theorist dir (default) or project ground truth
    model_registry = None
    if project_id:
        from src.models.ground_truth import get_ground_truth_models  # type: ignore
        gt = get_ground_truth_models(project_id)
        if model_name in gt:
            model_registry = gt
            theorist_dir = None  # use registry, not .py files

    # Load test statistic
    test_stat_fn = load_test_stat(Path(stat_file))

    # Load observed aggregate
    responses_path = exp_dir / "data" / "responses.csv"
    if not responses_path.exists():
        raise FileNotFoundError(f"responses.csv not found at {responses_path}")
    observed_rows = list(csv.DictReader(open(responses_path, encoding="utf-8")))
    observed_agg = aggregate_rows(observed_rows)

    # Infer n_participants from observed data
    participant_ids = {r["participant_id"] for r in observed_rows}
    n_participants = len(participant_ids)

    # Load stimuli
    stimuli_path = exp_dir / "design" / "stimuli.json"
    if not stimuli_path.exists():
        raise FileNotFoundError(f"stimuli.json not found at {stimuli_path}")
    stimuli = json.loads(stimuli_path.read_text(encoding="utf-8"))

    # Compute observed test statistic
    t_observed = float(test_stat_fn(observed_agg))

    # Resample: pass single-element list to pin all participants to this model
    null_values = []
    for _ in range(n_samples):
        synthetic_rows = _generate_from_models(
            stimuli=stimuli,
            model_names=[model_name],
            n_participants=n_participants,
            theorist_dir=theorist_dir,
            model_registry=model_registry,
        )
        synthetic_agg = aggregate_rows(synthetic_rows)
        t_synthetic = float(test_stat_fn(synthetic_agg))
        null_values.append(t_synthetic)

    # Empirical p-value (one-tailed: how often does the model produce values >= observed)
    n_extreme = sum(1 for t in null_values if t >= t_observed)
    # +1 in numerator and denominator avoids p=0 with finite samples
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run posterior predictive check for a model × test statistic")
    parser.add_argument("--exp-dir", required=True, help="Path to experimentN/ directory")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--stat-file", required=True, help="Path to test statistic .py file")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of resamples (default: 500)")
    parser.add_argument("--theorist-dir", default=None, help="Override path to cognitive_models/ dir")
    parser.add_argument("--project-id", default=None, help="Project ID for ground truth model lookup")
    args = parser.parse_args()

    theorist_dir = Path(args.theorist_dir) if args.theorist_dir else None
    result = run_ppc(
        exp_dir=Path(args.exp_dir),
        model_name=args.model,
        stat_file=Path(args.stat_file),
        n_samples=args.n_samples,
        theorist_dir=theorist_dir,
        project_id=args.project_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
