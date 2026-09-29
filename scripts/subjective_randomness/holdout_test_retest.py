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

It lists the sweep's expected cells that are partial (started, no
``holdout.json``) or missing (never started) — pass ``--n-repeats`` and
``--gt-models`` to state the expected grid — and, at the end of every
experiment, compares the loop's best model with the fitted-seed baseline fit
on that experiment's data over the same cells (``loop_vs_fitted_baseline``).

It also lists every cell and step whose metrics were computed on fewer held-out
pairs than the pool (``eval_exclusions``): a trajectory step where some model's
``p_left`` is undefined on some pairs (``n_eval_excluded`` in ``holdout.csv``)
and a fitted seed of the fitted-seed baseline likewise (from ``holdout.json``).
Those metrics are not on the same pairs as the no-learning baseline's.

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
from src.subjective_randomness.sweep_cells import survey_sweep  # noqa: E402


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
    n_repeats: Optional[int] = None
    """The sweep's repeat count, so cells that never started are listed as
    missing (default: inferred from the run<r>/ directories, and said so)."""
    gt_models: Optional[str] = None
    """The sweep's ground truths, space-separated (as GT_MODELS; default:
    inferred from the directories present, and said so)."""


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


def exclusion_report(csv_paths: list[Path], runs_root: Path) -> dict:
    """Every cell and step scored on fewer held-out pairs than its pool.

    Reads each tidy CSV's ``n_eval_excluded`` / ``eval_excluded_models``
    columns and, from the ``holdout.json`` beside it, each fitted seed's
    ``n_eval_excluded`` and the pool size. A CSV without the columns (written
    before 2026-09-27) is listed under ``cells_without_record``: whether its
    steps excluded pairs is not known from the CSV.
    """
    steps, seeds, unrecorded = [], [], []
    for csv_path in csv_paths:
        cell = str(csv_path.parent.relative_to(runs_root))
        result_path = csv_path.with_name("holdout.json")
        pool_sizes, seed_rows = {}, []
        if result_path.exists():
            for gt_run in json.loads(result_path.read_text(encoding="utf-8"))[
                "gt_runs"
            ]:
                pool_sizes[gt_run["gt_model"]] = gt_run["n_eval_stimuli"]
                for name, entry in gt_run["fitted_baseline"]["per_model"].items():
                    if entry.get("n_eval_excluded", 0) > 0:
                        seed_rows.append(
                            {
                                "cell": cell,
                                "gt_model": gt_run["gt_model"],
                                "seed": name,
                                "n_eval_excluded": entry["n_eval_excluded"],
                                "n_eval_stimuli": gt_run["n_eval_stimuli"],
                            }
                        )
        seeds += seed_rows
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "n_eval_excluded" not in (reader.fieldnames or []):
                unrecorded.append(cell)
                continue
            for row in reader:
                n = int(row["n_eval_excluded"])
                if n > 0:
                    steps.append(
                        {
                            "cell": cell,
                            "gt_model": row["gt_model"],
                            "experiment": int(row["experiment"]),
                            "step": int(row["step"]),
                            "global_step": int(row["global_step"]),
                            "best_model": row["best_model"],
                            "n_eval_excluded": n,
                            "n_eval_stimuli": pool_sizes.get(row["gt_model"]),
                            "models": row["eval_excluded_models"],
                        }
                    )
    return {
        "steps": steps,
        "fitted_seed_baseline": seeds,
        "cells_without_record": unrecorded,
    }


def print_exclusions(exclusions: dict) -> None:
    """The exclusion report, one line per affected step or seed."""

    def of(entry: dict) -> str:
        pool = entry["n_eval_stimuli"]
        return f"{entry['n_eval_excluded']} of {pool if pool is not None else '?'} held-out pairs"

    if not exclusions["steps"] and not exclusions["fitted_seed_baseline"]:
        print(
            "  eval exclusions: none (every step and fitted seed scored on its whole pool)"
        )
    for entry in exclusions["steps"]:
        print(
            f"  eval exclusions: {entry['cell']} experiment {entry['experiment']} step "
            f"{entry['step']}: {of(entry)} excluded (p_left undefined for "
            f"{entry['models']}); this step's metrics cover fewer pairs than the baselines'"
        )
    for entry in exclusions["fitted_seed_baseline"]:
        print(
            f"  eval exclusions: {entry['cell']} fitted-seed baseline, seed "
            f"{entry['seed']}: {of(entry)} excluded (its p_left is undefined there)"
        )
    if exclusions["cells_without_record"]:
        print(
            "  eval exclusions: not recorded in the CSVs of "
            f"{', '.join(exclusions['cells_without_record'])} (written before 2026-09-27)"
        )


# The fitted-seed baseline's field for each trajectory metric it has.
_FITTED_FIELDS = {"pearson_r": "elpd_best_r", "rmse": "elpd_best_rmse"}


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def loop_vs_fitted_baseline(results: dict[str, dict]) -> list[dict]:
    """At the end of every experiment, the loop's best model against the
    fitted-seed baseline fit on the same data, over the same cells.

    ``results`` maps a cell (``run<r>/<gt>``) to its ``holdout.json``. For each
    ground truth, experiment and metric: the cells where both the loop's value
    at the experiment's last step and the baseline's
    (``fitted_baseline_by_experiment``) are defined, their means and the mean
    paired difference (loop minus baseline), and every other cell with the
    reason it is left out. A result scored before per-experiment baselines
    existed has one only for its final experiment (``fitted_baseline``).
    """
    rows: list[dict] = []
    by_key: dict = defaultdict(lambda: {"cells": {}, "excluded": {}})
    for cell, result in sorted(results.items()):
        for gt_run in result["gt_runs"]:
            ends: dict[int, dict] = {}
            for row in gt_run["trajectory"]:
                if (
                    row["experiment"] not in ends
                    or row["step"] >= ends[row["experiment"]]["step"]
                ):
                    ends[row["experiment"]] = row
            final_experiment = max(ends) if ends else 0
            by_experiment = {
                int(e["experiment"]): e
                for e in gt_run.get("fitted_baseline_by_experiment") or []
            }
            for experiment, end in sorted(ends.items()):
                baseline = by_experiment.get(experiment)
                if (
                    baseline is None
                    and not by_experiment
                    and experiment == final_experiment
                ):
                    baseline = gt_run.get("fitted_baseline")
                for metric, field in _FITTED_FIELDS.items():
                    entry = by_key[(gt_run["gt_model"], experiment, metric)]
                    loop_value = end.get(metric)
                    base_value = (baseline or {}).get(field)
                    if not _finite(loop_value):
                        entry["excluded"][cell] = (
                            f"loop {metric} undefined at the end of the experiment"
                        )
                    elif baseline is None:
                        entry["excluded"][cell] = (
                            "no fitted-seed baseline for this experiment (scored before "
                            "per-experiment baselines; re-score it)"
                        )
                    elif not _finite(base_value):
                        entry["excluded"][cell] = (
                            baseline.get("elpd_best_reason")
                            or f"fitted-seed {field} undefined"
                        )
                    else:
                        entry["cells"][cell] = (float(loop_value), float(base_value))
    for (gt_model, experiment, metric), entry in sorted(by_key.items()):
        pairs = list(entry["cells"].values())
        loop = np.array([p[0] for p in pairs])
        base = np.array([p[1] for p in pairs])
        rows.append(
            {
                "gt_model": gt_model,
                "experiment": experiment,
                "metric": metric,
                "n_cells": len(pairs),
                "loop_mean": float(loop.mean()) if pairs else None,
                "fitted_baseline_mean": float(base.mean()) if pairs else None,
                "mean_difference": float((loop - base).mean()) if pairs else None,
                "cells": sorted(entry["cells"]),
                "excluded": dict(sorted(entry["excluded"].items())),
            }
        )
    return rows


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
    # Every expected run<r>/<gt> cell, so the cells without a result are
    # listed rather than silently absent from every number below. (The old
    # one-file-per-repeat layout has no cells to survey.)
    survey = None
    results: dict[str, dict] = {}
    if any(runs_root.glob(f"run*/*/{args.tidy_name}")):
        survey = survey_sweep(
            runs_root,
            n_repeats=args.n_repeats,
            gt_models=args.gt_models.split() if args.gt_models else None,
        )
        results = {
            label: json.loads((cell / "holdout.json").read_text(encoding="utf-8"))
            for label, cell in survey.complete.items()
        }

    EXTRA_METRICS = ("rmse", "kl_regret")

    # gt_model -> run_label -> {metric, pearson_r_bma, best_model, global_step, rmse, kl_regret}
    per_gt: dict[str, dict[str, dict]] = defaultdict(dict)
    found_runs_set: set[str] = set()

    for csv_path in csv_paths:
        run_label = csv_path.relative_to(runs_root).parts[0]  # e.g. "run3"
        found_runs_set.add(run_label)
        for gt, row in _final_rows_by_gt(csv_path, args.metric).items():
            entry = {
                "metric": _as_float(row.get(args.metric)),
                "pearson_r_bma": _as_float(row.get("pearson_r_bma")),
                "best_model": row.get("best_model"),
                "global_step": row.get("global_step") or row.get("step"),
            }
            for em in EXTRA_METRICS:
                entry[em] = _as_float(row.get(em))
            per_gt[gt][run_label] = entry

    found_runs = sorted(found_runs_set)
    # Expected cells without a tidy CSV (unfinished or never started).
    missing_runs = (
        sorted(
            label
            for label in (*survey.partial, *survey.missing, *survey.complete)
            if not (runs_root / label / args.tidy_name).exists()
        )
        if survey is not None
        else []
    )

    gt_models = sorted(per_gt)
    # Complete matrix (gt_models x runs) of the chosen metric, runs present in all.
    complete_runs = [
        r
        for r in found_runs
        if all(per_gt[gt].get(r, {}).get("metric") is not None for gt in gt_models)
    ]
    matrix = (
        np.array(
            [[per_gt[gt][r]["metric"] for r in complete_runs] for gt in gt_models],
            dtype=float,
        )
        if (gt_models and complete_runs)
        else np.empty((0, 0))
    )

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
        sd = (
            float(vals.std(ddof=1))
            if vals.size > 1
            else (0.0 if vals.size == 1 else None)
        )
        per_gt_summary[gt] = {
            "n_runs": int(vals.size),
            "mean": mean,
            "sd": sd,
            "cv": (
                float(sd / mean)
                if (mean not in (None, 0.0) and sd is not None)
                else None
            ),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "values": [round(v, 6) for v in vals.tolist()],
            "modal_best_model": modal_winner,
            "best_model_agreement": (modal_count / len(winners)) if winners else None,
        }

    per_metric: dict[str, dict] = {}
    for em in EXTRA_METRICS:
        em_per_gt: dict[str, dict] = {}
        for gt in gt_models:
            vals = np.array(
                [v[em] for v in per_gt[gt].values() if v.get(em) is not None],
                dtype=float,
            )
            mean_v = float(vals.mean()) if vals.size else None
            sd_v = (
                float(vals.std(ddof=1))
                if vals.size > 1
                else (0.0 if vals.size == 1 else None)
            )
            em_per_gt[gt] = {
                "n_runs": int(vals.size),
                "mean": mean_v,
                "sd": sd_v,
                "min": float(vals.min()) if vals.size else None,
                "max": float(vals.max()) if vals.size else None,
                "values": [round(v, 6) for v in vals.tolist()],
            }
        per_metric[em] = {"per_gt_model": em_per_gt}

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
        "per_metric": per_metric,
        "eval_exclusions": exclusion_report(csv_paths, runs_root),
        "cells": survey.as_dict() if survey is not None else None,
        "loop_vs_fitted_baseline": loop_vs_fitted_baseline(results),
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote test-retest summary to {out_path}")
    print(f"  runs used: {len(found_runs)} ({', '.join(found_runs)})")
    if survey is not None:
        print(
            f"  cells: {survey.n_expected} expected ({survey.expected_from}); "
            f"{len(survey.complete)} complete, {len(survey.partial)} partial, "
            f"{len(survey.missing)} missing"
        )
        for label, reason in sorted({**survey.partial, **survey.missing}.items()):
            print(f"    left out {label}: {reason}")
    if missing_runs:
        print(f"  cells missing {args.tidy_name}: {', '.join(missing_runs)}")
    for row in summary["loop_vs_fitted_baseline"]:
        if row["n_cells"] == 0:
            print(
                f"  {row['gt_model']} exp{row['experiment']} {row['metric']}: no cell has both"
            )
            continue
        print(
            f"  {row['gt_model']} end of exp{row['experiment']} {row['metric']}: loop "
            f"{row['loop_mean']:.4f} vs fitted seeds {row['fitted_baseline_mean']:.4f} "
            f"(diff {row['mean_difference']:+.4f}, same {row['n_cells']} cells; "
            f"{len(row['excluded'])} left out)"
        )
    icc = summary["icc_2_1"]
    mpc = summary["mean_pairwise_corr"]
    print(
        f"  ICC(2,1) = {'n/a' if icc is None else f'{icc:.3f}'}   "
        f"mean pairwise r = {'n/a' if mpc is None else f'{mpc:.3f}'}"
    )
    for gt, s in per_gt_summary.items():
        mean = s["mean"]
        sd = s["sd"]
        parts = [
            f"  {gt}: {args.metric} mean="
            f"{'n/a' if mean is None else f'{mean:.3f}'} "
            f"sd={'n/a' if sd is None else f'{sd:.3f}'} "
            f"(n={s['n_runs']})"
        ]
        for em in EXTRA_METRICS:
            em_s = per_metric.get(em, {}).get("per_gt_model", {}).get(gt, {})
            em_mean = em_s.get("mean")
            em_sd = em_s.get("sd")
            parts.append(
                f"{em} mean={'n/a' if em_mean is None else f'{em_mean:.4f}'} "
                f"sd={'n/a' if em_sd is None else f'{em_sd:.4f}'}"
            )
        parts.append(
            f"best-model agreement="
            f"{'n/a' if s['best_model_agreement'] is None else format(s['best_model_agreement'], '.2f')}"
            f" -> {s['modal_best_model']}"
        )
        print(", ".join(parts))
    print_exclusions(summary["eval_exclusions"])

    if args.csv is not None:
        csv_path = resolve_path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "gt_model",
                    "run",
                    args.metric,
                    *EXTRA_METRICS,
                    "pearson_r_bma",
                    "best_model",
                    "global_step",
                ]
            )
            for gt in gt_models:
                for run_name, v in sorted(per_gt[gt].items()):
                    writer.writerow(
                        [
                            gt,
                            run_name,
                            v["metric"],
                            *(v.get(em) for em in EXTRA_METRICS),
                            v["pearson_r_bma"],
                            v["best_model"],
                            v["global_step"],
                        ]
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
