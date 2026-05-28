"""Model–data correlation for PyMC cognitive models.

Pearson r between each model's posterior-mean P(response==1) per unique stimulus
and the observed proportion of response==1 per unique stimulus. The unique
stimuli are derived by grouping the trial-level CSV on the model's pm.Data
feature columns (all pm.Data containers except the observed-response one).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists. Returns 0.0 if undefined."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((v - mean_x) ** 2 for v in x)
    var_y = sum((v - mean_y) ** 2 for v in y)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    return cov / (var_x * var_y) ** 0.5


def model_data_correlations(
    model_names: List[str],
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Pearson r per model between posterior-mean response probability and
    observed response rate, computed at the unique-stimulus level.

    Stimuli are grouped by the tuple of pm.Data feature-column values
    (everything except the observed-response container). For each model,
    posterior-predictive mean is computed at one row per unique stimulus, then
    correlated with the observed response rate at that stimulus.
    """
    from src.models.pymc_inference import (  # type: ignore
        fit_models_cached,
        make_stim_data,
        observed_response_data,
        pm_data_inputs,
    )

    if not model_names:
        return {}

    responses_path = Path(responses_path)
    rows = list(csv.DictReader(responses_path.open(encoding="utf-8")))
    if not rows:
        return {m: 0.0 for m in model_names}

    fits = fit_models_cached(model_names, models_dir=models_dir, responses_path=responses_path, cache_dir=cache_dir)
    if not fits:
        return {m: 0.0 for m in model_names}

    # Use the first fitted model to identify stimulus feature columns vs response column.
    first_model = next(iter(fits.values())).model
    response_col = observed_response_data(first_model)
    feature_cols = [c for c in pm_data_inputs(first_model) if c != response_col]

    # Group trial-level rows by the feature-column tuple → observed response rate per stimulus.
    groups: Dict[tuple, List[int]] = defaultdict(list)
    feature_rows: Dict[tuple, Dict[str, str]] = {}
    for r in rows:
        key = tuple(r[c] for c in feature_cols)
        groups[key].append(int(float(r[response_col])))
        feature_rows.setdefault(key, {c: r[c] for c in feature_cols})

    unique_keys = list(groups.keys())
    observed = [sum(groups[k]) / len(groups[k]) for k in unique_keys]
    # Need a response column entry for set_data; the value is ignored for p_left predictions.
    pred_rows = [{**feature_rows[k], response_col: "0"} for k in unique_keys]

    correlations: Dict[str, float] = {}
    for m in model_names:
        if m not in fits:
            correlations[m] = 0.0
            continue
        stim_data = make_stim_data(fits[m].model, pred_rows)
        p_response_arr = fits[m].predict_p_left(stim_data)
        correlations[m] = round(pearson_r(list(p_response_arr), observed), 4)
    return correlations
