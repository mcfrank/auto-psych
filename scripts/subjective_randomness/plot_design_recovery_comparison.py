"""CLI: compare model-recovery quality across stimulus-selection designs.

Reads the ``test_retest.json`` summary of several finished holdout-recovery
test-retest runs — each run being one *stimulus-selection design* (pure EIG,
pure random, or a mix, at a given per-experiment stimulus budget) — and draws a
single figure comparing how well the agentic loop recovered each held-out
ground-truth model under each design.

Recovery metric: the Pearson correlation between the loop's best-fitting
discovered model and the held-out ground-truth model, evaluated on the shared
exhaustive stimulus pool, aggregated (mean +/- SD, with the individual repeats
overlaid) over the test-retest repeats recorded for that design.

Usage:
    plotvenv/bin/python \\
        scripts/subjective_randomness/plot_design_recovery_comparison.py \\
        --out-dir data/analysis/design_recovery_comparison
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from pyprojroot import here  # noqa: E402

# Each design's test-retest summary lives in its own $SCRATCH work root. Fail
# loudly (KeyError) if $SCRATCH is unset rather than silently guessing a path.
_RUN_ROOT = Path(os.environ["SCRATCH"]) / "auto-psych"

# Ordered condition -> test_retest.json. Order = the x-axis / legend order:
# the 32-stimulus EIG design first, then the three 64-stimulus designs.
DEFAULT_RUNS: dict[str, Path] = {
    "32 EIG": _RUN_ROOT / "holdout_faithful_test_retest_v8" / "test_retest.json",
    "64 random": _RUN_ROOT / "holdout_faithful_64random" / "test_retest.json",
    "64 random+EIG": _RUN_ROOT / "holdout_faithful_32eig_32random" / "test_retest.json",
    "64 EIG": _RUN_ROOT / "holdout_faithful_64eig" / "test_retest.json",
}

# One fixed colorblind-safe hue per condition (dataviz categorical slots 1/2/3/7;
# validated all-pairs: worst CVD dE 9.2, worst normal-vision dE 16.3, light mode).
CONDITION_COLORS: dict[str, str] = {
    "32 EIG": "#2a78d6",        # blue
    "64 random": "#eb6834",     # orange
    "64 random+EIG": "#1baf7a", # aqua
    "64 EIG": "#4a3aa7",        # violet
}

# Fixed ground-truth order + display names (kept short; long ones wrap).
GT_ORDER: list[str] = [
    "falk_konold_dp",
    "finite_experience_occurrence",
    "local_representativeness",
    "motif_stack",
]
GT_LABELS: dict[str, str] = {
    "falk_konold_dp": "Falk–Konold\nDP",
    "finite_experience_occurrence": "Finite experience\n(occurrence)",
    "local_representativeness": "Local\nrepresentativeness",
    "motif_stack": "Motif stack",
}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e2"


@dataclass
class Args:
    """Compare recovery across stimulus-selection designs."""

    runs: dict[str, Path] = field(default_factory=lambda: dict(DEFAULT_RUNS))
    """Ordered mapping of condition label -> that design's test_retest.json."""
    out_dir: Path = Path("data/analysis/design_recovery_comparison")
    """Directory (repo-relative) for the figure, CSV, and summary JSON."""
    metric: str = "pearson_r"
    """Recovery metric label (only pearson_r is recorded by the runs)."""


def _resolve(path: Path) -> Path:
    """Repo-relative paths resolve against the project root; absolute stay put."""
    return path if path.is_absolute() else here() / path


def load_per_gt(path: Path, metric: str) -> dict[str, dict]:
    """Load one run's per-ground-truth recovery block, failing loud on drift."""
    data = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if data["metric"] != metric:
        raise ValueError(f"{path}: metric is {data['metric']!r}, expected {metric!r}")
    per_gt = data["per_gt_model"]
    missing = [gt for gt in GT_ORDER if gt not in per_gt]
    if missing:
        raise ValueError(f"{path}: missing ground-truth models {missing}")
    return per_gt


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_SECONDARY)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))


def plot_comparison(runs_data: dict[str, dict[str, dict]], metric: str) -> plt.Figure:
    """Two panels: per-GT grouped bars (left) and the per-design mean (right)."""
    conditions = list(runs_data.keys())
    n_cond = len(conditions)

    fig, (ax_main, ax_mean) = plt.subplots(
        1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [2.7, 1.0]}
    )

    # --- Panel A: recovery per ground-truth model, one bar group per model ----
    x = np.arange(len(GT_ORDER))
    group_span = 0.82
    bar_w = group_span / n_cond
    # Deterministic within-bar jitter for the overlaid repeat points.
    jitter = np.linspace(-bar_w * 0.28, bar_w * 0.28, 5)

    for i, cond in enumerate(conditions):
        color = CONDITION_COLORS[cond]
        per_gt = runs_data[cond]
        offsets = x + (i - (n_cond - 1) / 2) * bar_w
        means = np.array([per_gt[gt]["mean"] for gt in GT_ORDER])
        sds = np.array([per_gt[gt]["sd"] for gt in GT_ORDER])

        ax_main.bar(
            offsets, means, bar_w * 0.86, color=color, edgecolor="white", lw=0.6,
            label=cond, zorder=2,
        )
        ax_main.errorbar(
            offsets, means, yerr=sds, fmt="none", ecolor=INK_SECONDARY,
            elinewidth=1.0, capsize=2.5, zorder=3,
        )
        for xoff, gt in zip(offsets, GT_ORDER):
            vals = np.asarray(per_gt[gt]["values"], dtype=float)
            pts = jitter[: len(vals)]
            ax_main.scatter(
                xoff + pts, vals, s=11, color="white", edgecolor=INK,
                linewidths=0.7, zorder=4,
            )

    _style_axis(ax_main)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels([GT_LABELS[gt] for gt in GT_ORDER], fontsize=9)
    ax_main.set_ylabel(f"Recovery ({metric})", fontsize=11, color=INK)
    ax_main.set_title(
        "Recovery of each held-out ground-truth model", fontsize=12, color=INK,
        pad=8,
    )
    ax_main.legend(
        title="Stimulus design", frameon=False, fontsize=9, title_fontsize=9,
        loc="lower left", ncol=2, labelcolor=INK,
    )

    # --- Panel B: mean recovery per design (averaged over the 4 GT models) ----
    xc = np.arange(n_cond)
    for i, cond in enumerate(conditions):
        per_gt = runs_data[cond]
        gt_means = np.array([per_gt[gt]["mean"] for gt in GT_ORDER])
        overall = gt_means.mean()
        spread = gt_means.std(ddof=1)
        color = CONDITION_COLORS[cond]
        ax_mean.bar(i, overall, 0.66, color=color, edgecolor="white", lw=0.6, zorder=2)
        ax_mean.errorbar(
            i, overall, yerr=spread, fmt="none", ecolor=INK_SECONDARY,
            elinewidth=1.0, capsize=3, zorder=3,
        )
        # Overlay the 4 per-GT means so the average isn't read as the whole story.
        ax_mean.scatter(
            np.full_like(gt_means, i), gt_means, s=14, color="white",
            edgecolor=INK, linewidths=0.7, zorder=4,
        )
        ax_mean.text(
            i, overall + spread + 0.03, f"{overall:.2f}", ha="center", va="bottom",
            fontsize=9, color=INK,
        )

    _style_axis(ax_mean)
    ax_mean.set_xticks(xc)
    ax_mean.set_xticklabels(conditions, fontsize=9, rotation=20, ha="right")
    ax_mean.set_ylabel(f"Mean recovery ({metric})", fontsize=11, color=INK)
    ax_mean.set_title("Mean over ground-truth models", fontsize=12, color=INK, pad=8)

    fig.suptitle(
        "Model recovery by stimulus-selection design (holdout test-retest)",
        fontsize=14, color=INK, y=0.99, x=0.01, ha="left", weight="bold",
    )
    fig.text(
        0.01, 0.005,
        "Points = individual test-retest repeats; bars = mean; error bars = SD "
        "across repeats (panel A) / across ground-truth models (panel B).",
        fontsize=8, color=INK_SECONDARY, ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    return fig


def write_tidy_csv(runs_data: dict[str, dict[str, dict]], path: Path, metric: str) -> None:
    """One row per (design, ground-truth): mean, sd, n, and the raw repeats."""
    lines = ["design,ground_truth,metric,n_repeats,mean,sd,values"]
    for cond, per_gt in runs_data.items():
        for gt in GT_ORDER:
            block = per_gt[gt]
            vals = ";".join(f"{v:.6f}" for v in block["values"])
            lines.append(
                f"{cond},{gt},{metric},{block['n_runs']},"
                f"{block['mean']:.6f},{block['sd']:.6f},{vals}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: Args) -> None:
    runs_data = {label: load_per_gt(path, args.metric) for label, path in args.runs.items()}

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Text summary to stdout (each design's per-GT mean and overall mean).
    print(f"Recovery ({args.metric}) by design:")
    header = "  ground truth".ljust(30) + "".join(f"{c:>16}" for c in runs_data)
    print(header)
    for gt in GT_ORDER:
        row = f"  {gt}".ljust(30)
        for per_gt in runs_data.values():
            row += f"{per_gt[gt]['mean']:>16.3f}"
        print(row)
    row = "  MEAN over GTs".ljust(30)
    for per_gt in runs_data.values():
        row += f"{np.mean([per_gt[gt]['mean'] for gt in GT_ORDER]):>16.3f}"
    print(row)

    csv_path = out_dir / "design_recovery_comparison.csv"
    write_tidy_csv(runs_data, csv_path, args.metric)
    print(f"\nWrote tidy CSV to {csv_path}")

    fig = plot_comparison(runs_data, args.metric)
    fig_path = out_dir / "design_recovery_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote figure to {fig_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
