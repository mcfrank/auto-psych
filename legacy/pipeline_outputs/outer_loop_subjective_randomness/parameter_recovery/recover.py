"""Run repeated simulation-and-fit checks for parameter recovery."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .fit import fit_rows
    from .simulate import generate_rows, load_stimuli
except ImportError:  # pragma: no cover - supports direct script execution
    from fit import fit_rows
    from simulate import generate_rows, load_stimuli


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


def _summarize(true_params: Mapping[str, float], estimates: List[Mapping[str, float]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, true_value in true_params.items():
        values = [float(est[name]) for est in estimates if name in est]
        if not values:
            continue
        mean_est = sum(values) / len(values)
        bias = mean_est - float(true_value)
        rmse = math.sqrt(sum((v - float(true_value)) ** 2 for v in values) / len(values))
        summary[name] = {
            "true": float(true_value),
            "mean_estimate": round(mean_est, 6),
            "bias": round(bias, 6),
            "rmse": round(rmse, 6),
            "min_estimate": round(min(values), 6),
            "max_estimate": round(max(values), 6),
        }
    return summary


def run_recovery(config: Mapping[str, Any], config_path: Path, repeats_override: int | None = None) -> Dict[str, Any]:
    model = importlib.import_module(config["model_module"])
    stimuli = load_stimuli(resolve_path(config["stimuli_path"], config_path))
    true_params = {k: float(v) for k, v in (config.get("true_params") or {}).items()}
    sim_cfg = config.get("simulation") or {}
    fit_cfg = config.get("fit") or {}
    n_repeats = repeats_override if repeats_override is not None else int(sim_cfg.get("n_repeats", 20))
    n_participants = int(sim_cfg.get("n_participants", 20))
    seed = int(sim_cfg.get("seed", 1))

    runs = []
    estimates = []
    for repeat in range(n_repeats):
        rows = generate_rows(model, stimuli, true_params, n_participants, seed + repeat)
        fit = fit_rows(
            model,
            rows,
            n_starts=int(fit_cfg.get("n_starts", 24)),
            max_iters=int(fit_cfg.get("max_iters", 160)),
            seed=seed + 10000 + repeat,
            fixed_params=fit_cfg.get("fixed_params") or {},
        )
        estimates.append(fit["params"])
        runs.append(
            {
                "repeat": repeat,
                "fit": fit,
            }
        )

    return {
        "model": getattr(model, "MODEL_NAME", config["model_module"].split(".")[-1]),
        "model_module": config["model_module"],
        "stimuli_path": config["stimuli_path"],
        "n_stimuli": len(stimuli),
        "n_participants": n_participants,
        "n_repeats": n_repeats,
        "true_params": true_params,
        "summary": _summarize(true_params, estimates),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parameter recovery for a subjective-randomness model family")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--n-repeats", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    result = run_recovery(load_config(config_path), config_path, args.n_repeats)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote recovery report for {result['model']} "
        f"({result['n_repeats']} repeats, {result['n_stimuli']} stimuli) to {out_path}"
    )


if __name__ == "__main__":
    main()
