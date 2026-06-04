"""Run PyMC parameter recovery for subjective-randomness model families.

This bridges the existing pure-Python model families and the PyMC adapters:

1. Simulate choices from the pure-Python reference model in `../model_families/`.
2. Featurize those rows with `../preprocess_data.py`.
3. Fit the matching PyMC adapter in `../pymc_model_families/`.
4. Compare posterior parameter summaries to the known true parameters.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = REPO_ROOT / "cc_pipeline" / "projects" / "subjective_randomness"
PYMC_MODELS_DIR = PROJECT_DIR / "pymc_model_families"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .simulate import generate_rows, load_stimuli
except ImportError:  # pragma: no cover - supports direct script execution
    from simulate import generate_rows, load_stimuli

from cc_pipeline.projects.subjective_randomness.preprocess_data import featurize_stimulus


def resolve_path(path_value: str | Path, config_path: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists() or config_path is None:
        return repo_path
    return (config_path.parent / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def model_name_from_module(module_path: str, model_module: Any | None = None) -> str:
    if model_module is not None and hasattr(model_module, "MODEL_NAME"):
        return str(model_module.MODEL_NAME)
    return module_path.split(".")[-1]


def featurize_response_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add numeric PyMC feature columns to simulated response rows."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        features = featurize_stimulus(str(row["sequence_a"]), str(row["sequence_b"]))
        out.append({**dict(row), **features})
    return out


def write_feature_rows(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    """Write simulated rows with feature columns to CSV."""
    if not rows:
        raise ValueError("Cannot write an empty simulated response table")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "participant_id",
        "trial_index",
        "sequence_a",
        "sequence_b",
        "chose_left",
        "chose_right",
        "model",
        "true_params",
    ]
    feature_cols = [
        "n_a", "h_a", "alts_a", "max_run_a",
        "n_b", "h_b", "alts_b", "max_run_b",
        "p_a", "p_alts_a", "max_run_norm_a", "imbalance_a", "periodicity_a",
        "p_b", "p_alts_b", "max_run_norm_b", "imbalance_b", "periodicity_b",
    ]
    extra_cols = sorted(set().union(*(r.keys() for r in rows)) - set(base_cols) - set(feature_cols))
    fieldnames = base_cols + feature_cols + extra_cols

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def posterior_summary(idata: Any, param_names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Summarize posterior parameters as mean, sd, and central 95% interval."""
    summary: Dict[str, Dict[str, float]] = {}
    posterior = idata.posterior
    for name in param_names:
        if name not in posterior:
            continue
        values = np.asarray(posterior[name].values, dtype=float).reshape(-1)
        if values.size == 0:
            continue
        summary[name] = {
            "mean": round(float(np.mean(values)), 6),
            "sd": round(float(np.std(values, ddof=0)), 6),
            "q025": round(float(np.quantile(values, 0.025)), 6),
            "q975": round(float(np.quantile(values, 0.975)), 6),
        }
    return summary


def summarize_recovery(
    true_params: Mapping[str, float],
    posterior_summaries: Sequence[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Any]:
    """Aggregate repeated posterior means against known true params."""
    out: Dict[str, Any] = {}
    for name, true_value in true_params.items():
        means = [
            float(summary[name]["mean"])
            for summary in posterior_summaries
            if name in summary and "mean" in summary[name]
        ]
        if not means:
            continue
        mean_est = sum(means) / len(means)
        out[name] = {
            "true": float(true_value),
            "mean_posterior_mean": round(mean_est, 6),
            "bias": round(mean_est - float(true_value), 6),
            "rmse": round(math.sqrt(sum((m - float(true_value)) ** 2 for m in means) / len(means)), 6),
            "min_posterior_mean": round(min(means), 6),
            "max_posterior_mean": round(max(means), 6),
        }
    return out


def run_pymc_recovery(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    repeats_override: int | None = None,
    pymc_models_dir: Path = PYMC_MODELS_DIR,
    work_dir: Path | None = None,
    cache_dir: Path | None = None,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    cores: int = 1,
) -> Dict[str, Any]:
    """Simulate from the reference model and fit the matching PyMC adapter."""
    from src.models.pymc_inference import fit_model  # type: ignore

    model_module = importlib.import_module(str(config["model_module"]))
    model_name = model_name_from_module(str(config["model_module"]), model_module)
    pymc_models_dir = Path(pymc_models_dir)
    if not (pymc_models_dir / f"{model_name}.py").exists():
        raise FileNotFoundError(
            f"No PyMC adapter found for {model_name!r}: {pymc_models_dir / f'{model_name}.py'}"
        )

    stimuli = load_stimuli(resolve_path(str(config["stimuli_path"]), config_path))
    true_params = {k: float(v) for k, v in (config.get("true_params") or {}).items()}
    sim_cfg = config.get("simulation") or {}
    n_repeats = repeats_override if repeats_override is not None else int(sim_cfg.get("n_repeats", 20))
    n_participants = int(sim_cfg.get("n_participants", 20))
    seed = int(sim_cfg.get("seed", 1))

    owns_tmp_dir = work_dir is None
    tmp_context = tempfile.TemporaryDirectory(prefix="subjective_randomness_pymc_recovery_") if owns_tmp_dir else None
    base_work_dir = Path(tmp_context.name) if tmp_context is not None else Path(work_dir)  # type: ignore[arg-type]
    base_work_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    posterior_summaries = []
    try:
        for repeat in range(n_repeats):
            rows = generate_rows(model_module, stimuli, true_params, n_participants, seed + repeat)
            feature_rows = featurize_response_rows(rows)
            responses_path = base_work_dir / f"{model_name}_repeat{repeat:03d}_responses.csv"
            write_feature_rows(feature_rows, responses_path)

            fitted = fit_model(
                model_name,
                models_dir=pymc_models_dir,
                responses_path=responses_path,
                cache_dir=cache_dir,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=seed + 10000 + repeat,
            )
            param_summary = posterior_summary(fitted.idata, true_params.keys())
            posterior_summaries.append(param_summary)
            runs.append(
                {
                    "repeat": repeat,
                    "responses_path": str(responses_path) if not owns_tmp_dir else None,
                    "fit_fingerprint": fitted.fingerprint,
                    "posterior": param_summary,
                }
            )
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()

    return {
        "model": model_name,
        "model_module": config["model_module"],
        "pymc_models_dir": str(pymc_models_dir),
        "stimuli_path": config["stimuli_path"],
        "n_stimuli": len(stimuli),
        "n_participants": n_participants,
        "n_repeats": n_repeats,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "true_params": true_params,
        "summary": summarize_recovery(true_params, posterior_summaries),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyMC parameter recovery for a subjective-randomness model family")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--pymc-models-dir", default=str(PYMC_MODELS_DIR))
    parser.add_argument("--work-dir", default=None, help="Optional directory to keep simulated featurized CSVs")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for PyMC .nc fit cache")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    result = run_pymc_recovery(
        load_config(config_path),
        config_path,
        repeats_override=args.n_repeats,
        pymc_models_dir=resolve_path(args.pymc_models_dir),
        work_dir=resolve_path(args.work_dir) if args.work_dir else None,
        cache_dir=resolve_path(args.cache_dir) if args.cache_dir else None,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
    )
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote PyMC recovery report for {result['model']} "
        f"({result['n_repeats']} repeats, {result['n_stimuli']} stimuli) to {out_path}"
    )


if __name__ == "__main__":
    main()
