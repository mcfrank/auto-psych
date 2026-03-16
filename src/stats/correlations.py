"""
Model–data correlation utilities.

Pearson r between each model's predicted P(left) and observed proportion chose left
per stimulus. Used by the data_analyst (run-only aggregate) and interpreter (merged
aggregate); correlations are always computed in code and injected into the interpreter
prompt—the LLM does not compute them.
"""

from pathlib import Path
from typing import Dict, List, Optional

from src.models.randomness import get_model_predictions


def pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists. Returns 0.0 if undefined (e.g. constant input)."""
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
    aggregate_lines: List[str],
    model_names: List[str],
    theorist_dir: Optional[Path],
    response_options: List[str],
) -> Dict[str, float]:
    """
    Pearson r between each model's predicted P(left) and observed chose_left_pct per stimulus.

    aggregate_lines: CSV rows (header + rows with seq_a, seq_b, chose_left_pct, n).
    Returns one correlation per model (rounded to 4 decimals).
    """
    observed = []
    stimuli = []
    for line in aggregate_lines[1:]:  # skip header
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                seq_a, seq_b, pct = parts[0], parts[1], float(parts[2])
                observed.append(pct)
                stimuli.append((seq_a, seq_b))
            except ValueError:
                continue
    if not stimuli or not model_names:
        return {m: 0.0 for m in model_names}
    model_predictions: Dict[str, List[float]] = {m: [] for m in model_names}
    for stim in stimuli:
        preds = get_model_predictions(stim, response_options, model_names, theorist_dir)
        for m in model_names:
            model_predictions[m].append(preds.get(m, {}).get("left", 0.5))
    return {m: round(pearson_r(model_predictions[m], observed), 4) for m in model_names}
