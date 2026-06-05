"""Fit subjective-randomness model-family parameters by maximum likelihood."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_choice_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse repeated participant rows to one row per stimulus with counts."""
    counts: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["sequence_a"]), str(row["sequence_b"]))
        if key not in counts:
            counts[key] = {
                "sequence_a": key[0],
                "sequence_b": key[1],
                "left_count": 0,
                "right_count": 0,
            }
        if int(row["chose_left"]):
            counts[key]["left_count"] += 1
        else:
            counts[key]["right_count"] += 1
    return list(counts.values())


def negative_log_likelihood(model_module, rows: Iterable[Mapping[str, Any]], params: Mapping[str, float]) -> float:
    nll = 0.0
    for row in rows:
        stimulus = (str(row["sequence_a"]), str(row["sequence_b"]))
        p_left = max(1e-9, min(1.0 - 1e-9, float(model_module.predict_left(stimulus, params))))
        if "left_count" in row or "right_count" in row:
            left_count = int(row.get("left_count", 0))
            right_count = int(row.get("right_count", 0))
            nll -= left_count * math.log(p_left)
            nll -= right_count * math.log(1.0 - p_left)
        else:
            chose_left = int(row["chose_left"])
            nll -= math.log(p_left if chose_left else 1.0 - p_left)
    return nll


def _random_params(bounds: Mapping[str, tuple[float, float]], rng: random.Random) -> Dict[str, float]:
    return {k: rng.uniform(lo, hi) if hi > lo else lo for k, (lo, hi) in bounds.items()}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _coordinate_search(
    model_module,
    rows: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, tuple[float, float]],
    start: Mapping[str, float],
    max_iters: int,
    rng: random.Random,
) -> tuple[Dict[str, float], float]:
    params = {k: _clamp(float(start[k]), *bounds[k]) for k in bounds}
    best = negative_log_likelihood(model_module, rows, params)
    step = {k: max((hi - lo) / 4.0, 1e-6) for k, (lo, hi) in bounds.items()}
    names = list(bounds)

    for _ in range(max_iters):
        rng.shuffle(names)
        improved = False
        for name in names:
            lo, hi = bounds[name]
            if hi <= lo:
                continue
            current = params[name]
            candidates = []
            for direction in (1.0, -1.0):
                proposal = dict(params)
                proposal[name] = _clamp(current + direction * step[name], lo, hi)
                candidates.append(proposal)
            for proposal in candidates:
                value = negative_log_likelihood(model_module, rows, proposal)
                if value + 1e-9 < best:
                    params = proposal
                    best = value
                    improved = True
                    break
        if not improved:
            for name in step:
                step[name] *= 0.5
            if max(step.values()) < 1e-4:
                break
    return params, best


def fit_rows(
    model_module,
    rows: Sequence[Mapping[str, Any]],
    n_starts: int = 24,
    max_iters: int = 160,
    seed: int = 0,
    fixed_params: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    fit_data = aggregate_choice_rows(rows)
    fixed = {k: float(v) for k, v in (fixed_params or {}).items()}
    raw_bounds = getattr(model_module, "PARAM_BOUNDS")
    bounds = {k: (float(v[0]), float(v[1])) for k, v in raw_bounds.items() if k not in fixed}
    defaults = dict(getattr(model_module, "DEFAULT_PARAMS", {}))

    def with_fixed(params: Mapping[str, float]) -> Dict[str, float]:
        merged = dict(params)
        merged.update(fixed)
        return merged

    starts: List[Dict[str, float]] = []
    default_start = {k: float(defaults.get(k, (lo + hi) / 2.0)) for k, (lo, hi) in bounds.items()}
    starts.append(default_start)
    for _ in range(max(0, n_starts - 1)):
        starts.append(_random_params(bounds, rng))

    best_params: Dict[str, float] | None = None
    best_nll = math.inf
    for start in starts:
        candidate, nll = _coordinate_search(model_module, fit_data, bounds, start, max_iters, rng)
        full_candidate = with_fixed(candidate)
        full_nll = negative_log_likelihood(model_module, fit_data, full_candidate)
        if full_nll < best_nll:
            best_nll = full_nll
            best_params = full_candidate
    assert best_params is not None
    return {
        "model": getattr(model_module, "MODEL_NAME", model_module.__name__.split(".")[-1]),
        "params": {k: round(v, 6) for k, v in sorted(best_params.items())},
        "negative_log_likelihood": round(best_nll, 6),
        "n_trials": len(rows),
        "n_stimuli": len(fit_data),
        "n_starts": n_starts,
        "max_iters": max_iters,
    }
