"""CLI: summarize and analyze subjective-randomness recovery results.

Auto-detects the result type from the JSON it is given:

* a *parameter-recovery* report (`pymc_recover.py`) -> per-parameter bias,
  RMSE, estimate spread, truth/estimate Pearson correlation, and 95%
  credible-interval coverage; optional figure. Sampled-truth reports (the
  default) get a ground-truth vs. recovered correlation scatter per parameter;
  fixed-truth reports get estimate spread around the true value.

* a *closed-ended model-recovery* confusion result (`model_recovery.py`) ->
  per-generating-model recovery (best model by posterior and by ELPD-LOO) plus
  overall accuracy; optional generating x recovered posterior confusion heatmap.

Usage:
    uv run python scripts/subjective_randomness/analyze_recovery.py \\
        --results data/subjective_randomness/model_recovery/confusion.json \\
        --out-csv data/subjective_randomness/model_recovery/summary.csv \\
        --figure data/subjective_randomness/model_recovery/confusion.png
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.analysis import (  # noqa: E402
    model_recovery_summary,
    parameter_recovery_summary,
)
from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.reporting import (  # noqa: E402
    MODEL_SUMMARY_COLUMNS,
    PARAM_SUMMARY_COLUMNS,
    model_recovery_text,
    parameter_recovery_text,
    plot_model_recovery,
    plot_parameter_recovery,
)
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402


@dataclass
class Args:
    """Summarize a recovery result (parameter recovery or model recovery)."""

    results: Path
    """Path to a recovery result JSON (auto-detected: param recovery vs. confusion)."""
    out_csv: Optional[Path] = None
    """Optional path to write the tidy per-row summary as CSV."""
    figure: Optional[Path] = None
    """Optional path to write a summary figure (PNG)."""


def _detect_kind(data: Mapping[str, Any]) -> str:
    """Classify a result JSON, failing loudly on anything unrecognized."""
    if "generating" in data and "seed_models" in data:
        return "model_recovery"
    if "runs" in data and ("true_params" in data or "param_ranges" in data):
        return "parameter_recovery"
    raise ValueError(
        "Unrecognized results file: expected a model-recovery confusion result "
        "(keys 'generating'/'seed_models') or a parameter-recovery report "
        f"(key 'runs' plus 'true_params' or 'param_ranges'). Got keys: {sorted(data)}."
    )


def main(args: Args) -> None:
    results_path = resolve_path(args.results)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    kind = _detect_kind(data)

    if kind == "parameter_recovery":
        rows = parameter_recovery_summary(data)
        print("\n" + parameter_recovery_text(data))
        csv_columns = PARAM_SUMMARY_COLUMNS
        plot = plot_parameter_recovery
    else:
        rows = model_recovery_summary(data)["per_model"]
        print("\n" + model_recovery_text(data))
        csv_columns = MODEL_SUMMARY_COLUMNS
        plot = plot_model_recovery

    if args.out_csv is not None:
        out_csv = resolve_path(args.out_csv)
        write_tidy_csv(rows, out_csv, columns=csv_columns)
        print(f"\nWrote summary CSV to {out_csv}")

    if args.figure is not None:
        figure_path = resolve_path(args.figure)
        plot(data, figure_path)
        print(f"Wrote figure to {figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
