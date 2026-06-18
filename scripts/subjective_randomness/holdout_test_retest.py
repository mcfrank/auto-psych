"""CLI: test-retest reliability across repeated holdout-recovery runs.

Given a ``runs_root`` whose per-task tidy CSVs live at ``run<r>/<gt>/holdout.csv``
(the per-GT array layout) or ``run<r>/holdout.csv`` (one file per repeat),
summarise how stable the recovered fit is across the repeats. Each repeat ``r``
used a distinct ``--seed`` but otherwise identical config.

For every held-out ground-truth model we take the *final* trajectory step of
each repeat (the largest ``global_step``) and collect that repeat's best-model
Pearson r. From the resulting ``gt_model x repeat`` matrix we report:

* per ground-truth-model mean / sd / coefficient-of-variation across repeats,
* ICC(2,1) (two-way random, single measure, absolute agreement) treating the
  ground-truth models as targets and the repeats as repeated measurements,
* the mean pairwise across-repeat Pearson correlation, and
* best-model selection agreement (how often the repeats land on the same winner).

Usage:
    uv run python scripts/subjective_randomness/holdout_test_retest.py \\
        --runs-root $SCRATCH/auto-psych/holdout_test_retest \\
        --out      $SCRATCH/auto-psych/holdout_test_retest/test_retest.json \\
        --csv      $SCRATCH/auto-psych/holdout_test_retest/test_retest.csv \\
        --figure   $SCRATCH/auto-psych/holdout_test_retest/test_retest.png
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402


@dataclass
class Args:
    """Summarise test-retest reliability across repeated holdout-recovery runs."""

    runs_root: Path
    """Directory holding the per-repeat run folders (run1, run2, ...)."""
    out: Path
    """Output JSON path for the reliability summary."""
    csv: Optional[Path] = None
    """Optional CSV: one row per (gt_model, repeat) with the final-step metrics."""
    figure: Optional[Path] = None
    """Optional figure: per-gt-model final r across repeats."""
    tidy_name: str = "holdout.csv"
    """Name of the tidy trajectory CSV written under each run<r>/[<gt>]/ dir."""
    metric: str = "pearson_r"
    """Which trajectory column to treat as the recovered fit metric."""


def _final_rows_by_gt(tidy_path: Path, metric: str) -> dict[str, dict]:
    """Return, per gt_model, the trajectory row with the largest global_step."""
    best: dict[str, dict] = {}
    best_step: dict[str, float] = {}
    with tidy_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gt = row["gt_model"]
            try:
                step = float(row.get("global_step") or row.get("step") or 0)
            except ValueError:
                step = 0.0
            if gt not in best or step >= best_step[gt]:
                best[gt] = row
                best_step[gt] = step
    return best


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _icc_2_1(matrix: np.ndarray) -> Optional[float]:
    """ICC(2,1): two-way random effects, single measure, absolute agreement.

    Rows are targets (gt_models), columns are raters (runs). Requires a complete
    matrix with at least 2 targets and 2 raters.
    """
    if matrix.ndim != 2:
        return None
    n, k = matrix.shape
    if n < 2 or k < 2 or not np.isfinite(matrix).all():
        return None
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((matrix - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    if denom == 0:
        return None
    return float((ms_rows - ms_err) / denom)


def _mean_pairwise_corr(matrix: np.ndarray) -> Optional[float]:
    """Mean Pearson correlation between every pair of runs (matrix columns)."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return None
    corrs = []
    for a in range(k):
        for b in range(a + 1, k):
            xa, xb = matrix[:, a], matrix[:, b]
            if np.std(xa) == 0 or np.std(xb) == 0:
                continue
            corrs.append(float(np.corrcoef(xa, xb)[0, 1]))
    return float(np.mean(corrs)) if corrs else None


def main(args: Args) -> None:
    runs_root = resolve_path(args.runs_root)
    # Each task writes its tidy CSV at run<r>/<gt>/<tidy_name> (per-GT layout) or
    # run<r>/<tidy_name> (one file per repeat). Glob both; the repeat label is the
    # leading "run<N>" path component and the ground truth comes from the rows.
    # Precise depth globs (run<r>/<gt>/ and run<r>/) rather than a recursive **,
    # so we never descend into the big repo copies / MCMC caches on Lustre.
    csv_paths = sorted(
        set(runs_root.glob(f"run*/*/{args.tidy_name}"))
        | set(runs_root.glob(f"run*/{args.tidy_name}"))
    )
    if not csv_paths:
        raise SystemExit(f"No {args.tidy_name!r} under run*/ in {runs_root}")

    # gt_model -> run_label -> {metric, pearson_r_bma, best_model, global_step}
    per_gt: dict[str, dict[str, dict]] = defaultdict(dict)
    found_runs_set: set[str] = set()

    for csv_path in csv_paths:
        run_label = csv_path.relative_to(runs_root).parts[0]  # e.g. "run3"
        found_runs_set.add(run_label)
        for gt, row in _final_rows_by_gt(csv_path, args.metric).items():
            per_gt[gt][run_label] = {
                "metric": _as_float(row.get(args.metric)),
                "pearson_r_bma": _as_float(row.get("pearson_r_bma")),
                "best_model": row.get("best_model"),
                "global_step": row.get("global_step") or row.get("step"),
            }

    found_runs = sorted(found_runs_set)
    missing_runs: list[str] = []

    gt_models = sorted(per_gt)
    # Complete matrix (gt_models x runs) of the chosen metric, runs present in all.
    complete_runs = [
        r for r in found_runs
        if all(per_gt[gt].get(r, {}).get("metric") is not None for gt in gt_models)
    ]
    matrix = np.array(
        [[per_gt[gt][r]["metric"] for r in complete_runs] for gt in gt_models],
        dtype=float,
    ) if (gt_models and complete_runs) else np.empty((0, 0))

    per_gt_summary = {}
    for gt in gt_models:
        vals = np.array(
            [v["metric"] for v in per_gt[gt].values() if v["metric"] is not None],
            dtype=float,
        )
        winners = [v["best_model"] for v in per_gt[gt].values() if v["best_model"]]
        modal_winner, modal_count = (
            Counter(winners).most_common(1)[0] if winners else (None, 0)
        )
        mean = float(vals.mean()) if vals.size else None
        sd = float(vals.std(ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else None)
        per_gt_summary[gt] = {
            "n_runs": int(vals.size),
            "mean": mean,
            "sd": sd,
            "cv": (float(sd / mean) if (mean not in (None, 0.0) and sd is not None) else None),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "values": [round(v, 6) for v in vals.tolist()],
            "modal_best_model": modal_winner,
            "best_model_agreement": (modal_count / len(winners)) if winners else None,
        }

    summary = {
        "runs_root": str(runs_root),
        "metric": args.metric,
        "n_runs_found": len(found_runs),
        "runs_found": found_runs,
        "runs_missing_tidy": missing_runs,
        "gt_models": gt_models,
        "runs_in_complete_matrix": complete_runs,
        "icc_2_1": _icc_2_1(matrix) if matrix.size else None,
        "mean_pairwise_corr": _mean_pairwise_corr(matrix) if matrix.size else None,
        "per_gt_model": per_gt_summary,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote test-retest summary to {out_path}")
    print(f"  runs used: {len(found_runs)} ({', '.join(found_runs)})")
    if missing_runs:
        print(f"  runs missing {args.tidy_name}: {', '.join(missing_runs)}")
    icc = summary["icc_2_1"]
    mpc = summary["mean_pairwise_corr"]
    print(f"  ICC(2,1) = {'n/a' if icc is None else f'{icc:.3f}'}   "
          f"mean pairwise r = {'n/a' if mpc is None else f'{mpc:.3f}'}")
    for gt, s in per_gt_summary.items():
        mean = s["mean"]
        sd = s["sd"]
        print(
            f"  {gt}: {args.metric} mean="
            f"{'n/a' if mean is None else f'{mean:.3f}'} "
            f"sd={'n/a' if sd is None else f'{sd:.3f}'} "
            f"(n={s['n_runs']}), best-model agreement="
            f"{'n/a' if s['best_model_agreement'] is None else f'{s['best_model_agreement']:.2f}'}"
            f" -> {s['modal_best_model']}"
        )

    if args.csv is not None:
        csv_path = resolve_path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["gt_model", "run", args.metric, "pearson_r_bma", "best_model", "global_step"]
            )
            for gt in gt_models:
                for run_name, v in sorted(per_gt[gt].items()):
                    writer.writerow(
                        [gt, run_name, v["metric"], v["pearson_r_bma"],
                         v["best_model"], v["global_step"]]
                    )
        print(f"Wrote per-run CSV to {csv_path}")

    if args.figure is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping figure.", file=sys.stderr)
            return
        figure_path = resolve_path(args.figure)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(gt_models)), 5))
        for i, gt in enumerate(gt_models):
            vals = per_gt_summary[gt]["values"]
            ax.scatter([i] * len(vals), vals, alpha=0.7, zorder=3)
            mean = per_gt_summary[gt]["mean"]
            if mean is not None:
                ax.hlines(mean, i - 0.2, i + 0.2, color="black", zorder=4)
        ax.set_xticks(range(len(gt_models)))
        ax.set_xticklabels(gt_models, rotation=30, ha="right")
        ax.set_ylabel(f"final {args.metric}")
        title = "Holdout recovery test-retest"
        if summary["icc_2_1"] is not None:
            title += f"  (ICC(2,1)={summary['icc_2_1']:.3f})"
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        print(f"Wrote figure to {figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
