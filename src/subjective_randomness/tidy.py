"""Reshape parameter-recovery reports into tidy (long-format) rows for plotting.

`pymc_recover.py` writes nested JSON reports; visualization tools (ggplot,
seaborn, pandas) want one row per observation instead. This module flattens a
report into one row per (parameter, repeat): the true value, the recovered
estimate (`run["posterior"][param]["mean"]`), and their error.

Parameter fitting is Bayesian-only, so the PyMC posterior shape is the only
one supported; anything else fails loudly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

TIDY_COLUMNS = ["model", "parameter", "repeat", "true_value", "estimate", "error"]


def _estimate_for_param(run: Mapping[str, Any], param: str) -> float:
    """Pull a single repeat's posterior-mean estimate for `param` from one run.

    Fails loudly if the run carries no PyMC `posterior` (e.g. reports from the
    deleted max-likelihood path).
    """
    if "posterior" in run:
        return float(run["posterior"][param]["mean"])
    raise KeyError(
        f"Run {run.get('repeat')!r} has no 'posterior' entry; cannot extract an "
        f"estimate for {param!r}. Parameter fitting is Bayesian-only — "
        "regenerate the report with pymc_recover.py."
    )


def parameter_recovery_tidy_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a recovery report into one row per (parameter, repeat).

    The ground truth for a row comes from the run's own `true_params`
    (sampled-truth reports) when present, else from the report-level
    `true_params` (fixed-truth reports). A report with neither fails loudly.
    """
    model = report["model"]
    shared_truth = report.get("true_params")
    rows: List[Dict[str, Any]] = []
    for run in report["runs"]:
        repeat = run["repeat"]
        true_params = run.get("true_params", shared_truth)
        if true_params is None:
            raise KeyError(
                f"Run {repeat!r} has no 'true_params' and the report carries no "
                "top-level 'true_params'; cannot pair estimates with ground truth."
            )
        for param, true_value in true_params.items():
            estimate = _estimate_for_param(run, param)
            rows.append(
                {
                    "model": model,
                    "parameter": param,
                    "repeat": repeat,
                    "true_value": float(true_value),
                    "estimate": estimate,
                    "error": estimate - float(true_value),
                }
            )
    return rows


def write_tidy_csv(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    columns: Sequence[str] = TIDY_COLUMNS,
) -> None:
    """Write tidy rows to a CSV with a fixed column order.

    Fails loudly if any row is missing a declared column.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            missing = [c for c in columns if c not in row]
            if missing:
                raise KeyError(f"Tidy row missing columns {missing}: {dict(row)}")
            writer.writerow({c: row[c] for c in columns})
