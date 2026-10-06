"""Pooled holdout-recovery curves over inner-loop scoring steps (matplotlib).

For each stimulus-selection design (one holdout test-retest run root), pool its
per-``run<r>/<gt_model>/holdout.json`` trajectories with the repo's own
``aggregate_holdout_trajectories`` and draw the best-model recovery vs. the
inner-loop scoring step — one panel per held-out ground-truth model, the mean
across repeats with a ``±`` spread error bar, the Bayesian-model-average curve
dashed, and the two flat seed-model baselines.

This is the matplotlib twin of ``scripts/analysis/plot_holdout_combined.py``
(which renders the same aggregate via plotnine); it exists so the figures can be
produced in an environment without plotnine/statsmodels.

Usage (Sherlock dev node, NOT login):
    eltest_venv3/bin/python \\
        scripts/subjective_randomness/plot_holdout_curves.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pyprojroot import here  # noqa: E402

from src.subjective_randomness.reporting import (  # noqa: E402
    aggregate_holdout_trajectories,
)

_RUN_ROOT = Path(os.environ["SCRATCH"]) / "auto-psych"

# Ordered condition label -> that design's holdout test-retest run root.
DEFAULT_RUNS: Dict[str, Path] = {
    "32_eig": _RUN_ROOT / "holdout_faithful_test_retest_v8",
    "64_random": _RUN_ROOT / "holdout_faithful_64random",
    "64_random_eig": _RUN_ROOT / "holdout_faithful_32eig_32random",
    "64_eig": _RUN_ROOT / "holdout_faithful_64eig",
}

# Per-metric axis styling (mirrors reporting._HOLDOUT_METRIC_SPECS).
_METRIC_STYLE = {
    "pearson_r": {
        "ylabel": "Pearson r vs. ground-truth p_left",
        "ylim": (-1.05, 1.05),
        "zero_line": True,
    },
    "rmse": {
        "ylabel": "RMSE vs. ground-truth p_left",
        "ylim": None,  # autoscale up from 0
        "zero_line": False,
    },
}

BEST_COLOR = "#4878CF"
BMA_COLOR = "#D65F5F"
SEED_FIT_COLOR = "#EE854A"
BASELINE_COLOR = "#6ACC65"


@dataclass
class Args:
    """Draw pooled recovery curves over inner-loop steps for each design."""

    runs: Dict[str, Path] = field(default_factory=lambda: dict(DEFAULT_RUNS))
    """Ordered mapping of condition label -> holdout test-retest run root."""
    out_dir: Path = Path("data/analysis/design_recovery_comparison/recovery_curves")
    """Directory (repo-relative) for the per-condition figures."""
    error: str = "sem"
    """Spread the error bars show: sem, std, or ci95."""


def _load_run_results(runs_root: Path) -> List[dict]:
    files = sorted(runs_root.glob("run*/*/holdout.json"))
    if not files:
        raise FileNotFoundError(
            f"No run files matched {runs_root}/run*/*/holdout.json — nothing to pool."
        )
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def _plot_condition(label: str, aggregated: dict, metric: str, out_path: Path) -> None:
    style = _METRIC_STYLE[metric]
    panels = aggregated["gt_models"]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False, sharey=True)

    plotted = []
    for ax, panel in zip(axes[0], panels):
        for series_key, color, ls, marker, lbl in (
            ("best", BEST_COLOR, "-", "o", "best model"),
            ("bma", BMA_COLOR, "--", "s", "Bayesian model average"),
        ):
            points = panel.get(series_key) or []
            if points:
                xs = [p["global_step"] for p in points]
                ys = [p["mean"] for p in points]
                errs = [p["err"] for p in points]
                plotted.extend(ys)
                ax.errorbar(
                    xs, ys, yerr=errs, color=color, linestyle=ls, marker=marker,
                    markersize=4, capsize=2, linewidth=1.4, label=lbl,
                )
        for run_key, color, ls, lbl in (
            ("fitted_baseline", SEED_FIT_COLOR, ":", "seed models (fit to all data)"),
            ("baseline", BASELINE_COLOR, "-.", "seed models (default params)"),
        ):
            stats = panel["baselines"].get(run_key)
            if stats is not None:
                plotted.append(stats["mean"])
                ax.axhline(stats["mean"], color=color, linestyle=ls, linewidth=1.3, label=lbl)
        for x in panel.get("experiment_boundaries", []):
            ax.axvline(x - 0.5, color="grey", linestyle=":", linewidth=0.8)
        if style["zero_line"]:
            ax.axhline(0.0, color="grey", linewidth=0.5)
        n_runs = panel.get("n_runs", "?")
        ax.set_title(f"{panel['gt_model']}  (n={n_runs})")
        ax.set_xlabel("inner-loop scoring step")

    if style["ylim"] is not None:
        axes[0][0].set_ylim(*style["ylim"])
    elif plotted:
        axes[0][0].set_ylim(0.0, max(plotted) * 1.15)
    axes[0][0].set_ylabel(style["ylabel"])
    axes[0][0].legend(loc="lower right" if metric == "pearson_r" else "upper right", fontsize=8)
    fig.suptitle(
        f"Holdout recovery over inner-loop steps — design: {label}  "
        f"(mean ± {aggregated['error']} across repeats)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args: Args) -> None:
    out_root = here() / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    for label, runs_root in args.runs.items():
        results = _load_run_results(runs_root)
        n_files = sum(len(r["gt_runs"]) for r in results)
        print(f"[{label}] pooled {n_files} run files from {runs_root}")
        for metric in ("pearson_r", "rmse"):
            aggregated = aggregate_holdout_trajectories(
                results, metric=metric, error=args.error
            )
            out_path = out_root / label / f"curves_{metric}.png"
            _plot_condition(label, aggregated, metric, out_path)
            print(f"    wrote {out_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
