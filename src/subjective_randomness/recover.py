"""Run repeated simulation-and-fit checks for parameter recovery."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.subjective_randomness.config import resolve_path
from src.subjective_randomness.fit import fit_rows
from src.subjective_randomness.simulate import generate_rows, load_stimuli


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
