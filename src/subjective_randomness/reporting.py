"""Human-readable text blocks and figures for recovery results.

* `parameter_recovery_text` / `plot_parameter_recovery` — per-parameter
  recovery quality for a PyMC parameter-recovery report. Sampled-truth reports
  get a ground-truth vs. recovered correlation scatter per parameter;
  fixed-truth reports get the estimate spread around the single true value.

* `model_recovery_text` / `plot_model_recovery` — per-generating-model
  recovery and the posterior confusion heatmap for a closed-ended
  `model_recovery.py` result.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean
from statistics import stdev as _stdev
from typing import Any, Iterable, List, Mapping, Sequence

from src.subjective_randomness.analysis import (
    model_recovery_summary,
    parameter_recovery_summary,
)
from src.subjective_randomness.tidy import parameter_recovery_tidy_rows

PARAM_SUMMARY_COLUMNS = [
    "model",
    "parameter",
    "true_value",
    "mean_estimate",
    "bias",
    "rmse",
    "estimate_sd",
    "pearson_r",
    "n_repeats",
    "ci_coverage_95",
]
MODEL_SUMMARY_COLUMNS = [
    "generating_model",
    "true_posterior",
    "best_by_posterior",
    "best_by_elpd",
    "correct_posterior",
    "correct_elpd",
    "winner_by_elpd",
    "winner_margin",
    "winner_margin_dse",
    "winner_distinguishable",
    "true_model_elpd_diff",
    "true_model_dse",
    "recovery_clear",
]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "n/a"
    return str(value)


def parameter_recovery_text(report: Mapping[str, Any]) -> str:
    """Per-parameter recovery quality as an aligned text table."""
    rows = parameter_recovery_summary(report)
    lines = [f"Parameter recovery — model: {report['model']}"]
    lines.append(f"  repeats: {report.get('n_repeats', rows[0]['n_repeats'])}")
    header = [
        "parameter",
        "true_value",
        "mean_estimate",
        "bias",
        "rmse",
        "pearson_r",
        "ci_coverage_95",
    ]
    lines.append("  " + "  ".join(f"{h:>14}" for h in header))
    for r in rows:
        lines.append("  " + "  ".join(f"{_fmt(r[h]):>14}" for h in header))
    return "\n".join(lines)


def recovery_note(row: Mapping[str, Any]) -> str:
    """A short per-row annotation flagging mis-recovery and/or statistical ties."""
    if not row["correct_posterior"]:
        if row["winner_distinguishable"] is False:
            return "   <- mis-recovered (but tied: not distinguishable)"
        return "   <- mis-recovered"
    if row["winner_distinguishable"] is False:
        return "   <- recovered, but tied with runner-up"
    return ""


def model_recovery_text(confusion: Mapping[str, Any]) -> str:
    """Closed-ended model-recovery metrics as an aligned text table."""
    summary = model_recovery_summary(confusion)
    n = summary["n_models"]
    lines = [
        f"Closed-ended model recovery — generator: {confusion.get('generator', '?')}"
    ]
    line = (
        f"  {n} models | posterior accuracy: {summary['posterior_accuracy']:.2f} "
        f"({sum(r['correct_posterior'] for r in summary['per_model'])}/{n}) | "
        f"ELPD-LOO accuracy: {summary['elpd_accuracy']:.2f} | "
        f"mean posterior on true model: {summary['mean_true_posterior']:.3f}"
    )
    if summary["has_comparison"]:
        n_clear = sum(bool(r["recovery_clear"]) for r in summary["per_model"])
        line += f" | clearly recovered: {summary['clear_recovery_rate']:.2f} ({n_clear}/{n})"
    lines.append(line)
    header = ["generating_model", "true_posterior", "best_by_elpd", "winner_margin"]
    lines.append("  " + "  ".join(f"{h:>20}" for h in header))
    for r in summary["per_model"]:
        lines.append(
            "  " + "  ".join(f"{_fmt(r[h]):>20}" for h in header) + recovery_note(r)
        )
    if summary["has_comparison"]:
        lines.append(
            "\n  note: `winner_margin` is the runner-up's elpd_diff (with dse). "
            "A recovery is only 'clear' when that margin exceeds ~2·dse; "
            "smaller margins mean the top models are statistically tied."
        )
    return "\n".join(lines)


def plot_parameter_recovery(report: Mapping[str, Any], out_path: Path) -> None:
    """Write the figure that suits the report's ground-truth structure.

    Sampled-truth reports (truths vary across repeats) get a ground-truth vs.
    recovered correlation scatter per parameter; fixed-truth reports get the
    spread of estimates around the single true value.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tidy = parameter_recovery_tidy_rows(report)
    params = list(dict.fromkeys(r["parameter"] for r in tidy))
    by_param = {p: [r for r in tidy if r["parameter"] == p] for p in params}
    truths_vary = any(
        len({r["true_value"] for r in rows}) > 1 for rows in by_param.values()
    )

    fig, axes = plt.subplots(
        1, len(params), figsize=(3.4 * len(params), 3.6), squeeze=False
    )
    if truths_vary:
        _draw_correlation_panels(report, axes[0], by_param)
    else:
        _draw_fixed_truth_panels(report, axes[0], by_param)
    fig.suptitle(f"Parameter recovery — {report['model']}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_correlation_panels(
    report: Mapping[str, Any], axes, by_param: Mapping[str, list]
) -> None:
    """One ground-truth (x) vs. recovered (y) scatter per parameter."""
    pearson = {
        row["parameter"]: row["pearson_r"] for row in parameter_recovery_summary(report)
    }
    for ax, (param, rows) in zip(axes, by_param.items()):
        trues = [r["true_value"] for r in rows]
        ests = [r["estimate"] for r in rows]
        lo = min(trues + ests)
        hi = max(trues + ests)
        pad = 0.05 * ((hi - lo) or 1.0)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="#999999",
            linestyle="--",
            linewidth=1,
            label="identity",
        )
        ax.scatter(trues, ests, alpha=0.6, color="#4878CF")
        r = pearson[param]
        ax.set_title(param if r is None else f"{param} (r = {r:.2f})")
        ax.set_xlabel("true value")
        ax.set_ylabel("recovered estimate")
    axes[-1].legend(loc="best", fontsize=8)


def _draw_fixed_truth_panels(
    report: Mapping[str, Any], axes, by_param: Mapping[str, list]
) -> None:
    """Estimate spread around the single true value, one panel per parameter."""
    import random

    rng = random.Random(0)  # deterministic horizontal jitter
    for ax, (param, rows) in zip(axes, by_param.items()):
        ests = [r["estimate"] for r in rows]
        xs = [1 + (rng.random() - 0.5) * 0.3 for _ in ests]
        ax.scatter(xs, ests, alpha=0.6, color="#4878CF", label="estimates")
        ax.axhline(rows[0]["true_value"], color="#D65F5F", linestyle="--", label="true")
        ax.set_title(param)
        ax.set_xticks([])
        ax.set_ylabel("recovered estimate")
    axes[-1].legend(loc="best", fontsize=8)


# Per-metric plotting spec for `plot_holdout_trajectories`. Each metric names
# the trajectory keys for the best-model and BMA series, the per-run baseline
# field, and how to label/scale the axis. The default-params (green) baseline
# only records `mean_r`, so it is absent on the RMSE figure (its `baseline_field`
# lookup returns None and the line is skipped).
_HOLDOUT_METRIC_SPECS = {
    "pearson_r": {
        "best_key": "pearson_r",
        "bma_key": "pearson_r_bma",
        "baseline_field": "mean_r",
        "fitted_baseline_field": "elpd_best_r",
        "ylabel": "Pearson r vs. ground-truth p_left (held-out stimuli)",
        "suptitle": (
            "Holdout recovery — best model vs. Bayesian model average "
            "(held-out model = ground truth)"
        ),
        "combined_suptitle": (
            "Holdout recovery — best-model correlation with held-out ground truth"
        ),
        "combined_ylabel": "Pearson r vs. ground truth",
        "higher_is_better": True,
        "ylim": (-1.05, 1.05),
        "zero_line": True,
        "legend_loc": "lower right",
    },
    "rmse": {
        "best_key": "rmse",
        "bma_key": "rmse_bma",
        "baseline_field": "mean_rmse",
        "fitted_baseline_field": "elpd_best_rmse",
        "ylabel": "RMSE vs. ground-truth p_left (held-out stimuli)",
        "suptitle": (
            "Holdout recovery — RMSE of best model vs. Bayesian model average "
            "(held-out model = ground truth; lower is better)"
        ),
        "combined_suptitle": (
            "Holdout recovery — best-model RMSE vs. held-out ground truth "
            "(lower is better)"
        ),
        "combined_ylabel": "RMSE vs. ground truth",
        "higher_is_better": False,
        "ylim": None,  # autoscale up from 0
        "zero_line": False,
        "legend_loc": "upper right",
    },
}


def plot_holdout_trajectories(
    result: Mapping[str, Any], out_path: Path, *, metric: str = "pearson_r"
) -> None:
    """Plot held-out recovery vs. inner-loop step, one panel per held-out model.

    Each panel shows two trajectories of the chosen ``metric`` against the
    ground truth's ``p_left`` on the held-out stimuli: the single best-fitting
    model (solid) and the posterior-weighted Bayesian model average (dashed).
    ``metric`` is ``"pearson_r"`` (higher is better, fixed [-1, 1] axis) or
    ``"rmse"`` (lower is better, axis autoscaled up from 0). Steps with an
    undefined value (None — e.g. a constant prediction makes correlation
    undefined) are skipped rather than plotted as zero. Dotted vertical lines
    mark outer-experiment boundaries. The fitted-seed baseline is drawn over
    each experiment's steps at its value for that experiment's data; the
    default-params baseline is a flat line.
    """
    if metric not in _HOLDOUT_METRIC_SPECS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of "
            f"{sorted(_HOLDOUT_METRIC_SPECS)}"
        )
    spec = _HOLDOUT_METRIC_SPECS[metric]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BEST_COLOR = "#4878CF"
    BMA_COLOR = "#D65F5F"
    SEED_FIT_COLOR = "#EE854A"
    BASELINE_COLOR = "#6ACC65"

    gt_runs = result["gt_runs"]
    n = len(gt_runs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False, sharey=True)
    plotted_values = []
    for ax, gt_run in zip(axes[0], gt_runs):
        trajectory = gt_run["trajectory"]
        for key, color, style, marker, label in (
            (spec["best_key"], BEST_COLOR, "-", "o", "best model"),
            (spec["bma_key"], BMA_COLOR, "--", "s", "Bayesian model average"),
        ):
            points = [
                (row["global_step"], row[key])
                for row in trajectory
                if row.get(key) is not None
            ]
            if points:
                xs, ys = zip(*points)
                plotted_values.extend(ys)
                ax.plot(
                    xs, ys, color=color, linestyle=style, marker=marker, label=label
                )
        # The fitted-seed baseline over each experiment's steps, fit on that
        # experiment's cumulative data (the data those steps were fit on). A
        # result scored before per-experiment baselines has only the final
        # data's, drawn over the final experiment's steps.
        by_experiment = gt_run.get("fitted_baseline_by_experiment") or []
        if not by_experiment and gt_run.get("fitted_baseline") and trajectory:
            by_experiment = [{**gt_run["fitted_baseline"],
                              "experiment": max(row["experiment"] for row in trajectory)}]
        drawn = False
        for entry in by_experiment:
            value = entry.get(spec["fitted_baseline_field"])
            steps = [row["global_step"] for row in trajectory
                     if row["experiment"] == entry["experiment"]]
            if value is None or not steps:
                continue
            plotted_values.append(value)
            ax.hlines(
                value, min(steps) - 0.4, max(steps) + 0.4,
                colors=SEED_FIT_COLOR, linestyles=":", linewidth=1.6,
                label=None if drawn else "ELPD-best seed model (fit to the experiment's data)",
            )
            drawn = True
        # The other seed models with default params: no data, so flat.
        for run_key, baseline_color, baseline_style, baseline_label in (
            ("baseline", BASELINE_COLOR, "-.", "seed models (default params)"),
        ):
            baseline_value = (gt_run.get(run_key) or {}).get(spec["baseline_field"])
            if baseline_value is not None:
                plotted_values.append(baseline_value)
                ax.axhline(
                    baseline_value,
                    color=baseline_color,
                    linestyle=baseline_style,
                    linewidth=1.3,
                    label=baseline_label,
                )
        for x in sorted(
            row["global_step"]
            for row in trajectory
            if row["step"] == 0 and row["experiment"] > 1
        ):
            ax.axvline(x - 0.5, color="grey", linestyle=":", linewidth=0.8)
        if spec["zero_line"]:
            ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_title(gt_run["gt_model"])
        ax.set_xlabel("inner-loop scoring step")

    # Axes share y, so one limit governs every panel. Fixed window for the
    # bounded correlation; autoscale up from 0 for RMSE so small differences
    # near the floor stay legible.
    if spec["ylim"] is not None:
        axes[0][0].set_ylim(*spec["ylim"])
    else:
        top = max(plotted_values) * 1.15 if plotted_values else 1.0
        axes[0][0].set_ylim(0.0, top)

    axes[0][0].set_ylabel(spec["ylabel"])
    axes[0][0].legend(loc=spec["legend_loc"], fontsize=8)
    fig.suptitle(spec["suptitle"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_ERROR_KINDS = ("sem", "std", "ci95")


def _is_finite(value: Any) -> bool:
    """True only for a real, finite number — not None, NaN, or +/-inf.

    Impossible-ground-truth recoveries can make a metric undefined (a constant
    prediction gives a NaN correlation) or unbounded; such points are dropped
    from a step's sample exactly like an explicit ``None``.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _summarize(values: List[float], error: str) -> Mapping[str, Any]:
    """Reduce a list of per-run values to a mean and a spread.

    The spread is the sample standard deviation (``std``), the standard error
    of the mean (``sem = std / sqrt(n)``), or a 95% normal interval half-width
    (``ci95 = 1.96 * sem``). With fewer than two values the spread is undefined
    and reported as 0.0 so a lone run still plots a point with no whisker.
    """
    n = len(values)
    avg = _mean(values)
    if n < 2:
        spread = 0.0
    else:
        std = _stdev(values)
        spread = {
            "std": std,
            "sem": std / math.sqrt(n),
            "ci95": 1.96 * std / math.sqrt(n),
        }[error]
    return {"mean": avg, "err": spread, "n": n}


def _best_seed_value(
    baseline: Mapping[str, Any], run_key: str, spec: Mapping[str, Any]
) -> Any:
    """A cell's seed-model baseline value for the combined figure, or None.

    The fit-to-all-data baseline is the ELPD-best seed — chosen on the
    training data, as the loop chooses its winner (``elpd_best_*``). It used to
    be the seed with the best held-out value against the ground truth, which
    only an oracle could choose. A result scored before ``elpd_best_*`` existed
    has no such baseline; that is said on stderr, not papered over.

    The default-params baseline stores only a bare Pearson r per model (so it
    has no RMSE) and no ELPD; for it "best" is still the max r, an explicitly
    optimistic reference.
    """
    per_model = baseline.get("per_model")
    if not per_model:
        return None
    metric_key = spec["best_key"]
    if run_key == "fitted_baseline":
        field = spec["fitted_baseline_field"]
        if field not in baseline:
            print(
                f"  [warn] a result has no ELPD-best seed ({field!r} missing: scored "
                "before it existed); its fitted-seed baseline is left out. "
                "Re-score it to include it.",
                file=sys.stderr,
                flush=True,
            )
            return None
        value = baseline[field]
        return value if _is_finite(value) else None
    if metric_key == "pearson_r":  # default-params baseline: bare Pearson r
        values = list(per_model.values())
    else:
        return None  # default-params baseline has no RMSE
    values = [v for v in values if _is_finite(v)]
    if not values:
        return None
    return max(values) if spec["higher_is_better"] else min(values)


def _experiment_rows(trajectory: Iterable[Mapping[str, Any]]) -> "dict[int, List[Mapping[str, Any]]]":
    """A cell's trajectory rows grouped by experiment, in step order."""
    by_experiment: "defaultdict[int, List[Mapping[str, Any]]]" = defaultdict(list)
    for row in trajectory:
        by_experiment[int(row["experiment"])].append(row)
    return {
        experiment: sorted(rows, key=lambda row: int(row["step"]))
        for experiment, rows in by_experiment.items()
    }


def _row_at(rows: List[Mapping[str, Any]], label: str) -> Any:
    """A cell's row at an aligned position of one experiment (None if absent)."""
    if label == "seed":
        return rows[0]
    if label == "end":
        return rows[-1]
    iteration = int(label.removeprefix("round "))
    return next((row for row in rows[1:] if row.get("iteration") == iteration), None)


def _aligned_labels(cells: List[List[Mapping[str, Any]]]) -> List[str]:
    """The positions of one experiment every cell has: its seed step, each
    candidate round (by ``iteration``) that every cell recorded, and its end
    (each cell's last step). A position that is every cell's last step is
    shown once, as the end."""
    common = set.intersection(
        *({row["iteration"] for row in rows[1:] if row.get("iteration") is not None}
          for rows in cells)
    )
    labels = ["seed"] + [f"round {i}" for i in sorted(common)]
    labels = [
        label for label in labels
        if not all(_row_at(rows, label) is rows[-1] for rows in cells)
    ]
    return labels + ["end"]


def _fitted_seed_value(
    gt_run: Mapping[str, Any], experiment: int, final_experiment: int, field: str
) -> "tuple[Any, str]":
    """A cell's fitted-seed baseline value at the end of ``experiment``, or
    ``(None, reason)``.

    ``fitted_baseline_by_experiment`` holds one entry per experiment, each fit
    on that experiment's cumulative data. A result scored before it existed
    has only ``fitted_baseline`` (the final data), which is the final
    experiment's value; its earlier experiments have none.
    """
    by_experiment = gt_run.get("fitted_baseline_by_experiment")
    if by_experiment is not None:
        entry = next((e for e in by_experiment if int(e["experiment"]) == experiment), None)
        if entry is None:
            return None, f"no fitted-seed baseline recorded for experiment {experiment}"
    elif experiment == final_experiment:
        entry = gt_run.get("fitted_baseline") or {}
    else:
        return None, (
            "no fitted-seed baseline before the final experiment (scored before "
            "per-experiment baselines existed; re-score it with "
            "reevaluate_trajectories)"
        )
    if field not in entry:
        return None, f"no {field!r} in its fitted-seed baseline (scored before it existed; re-score it)"
    value = entry[field]
    if not _is_finite(value):
        return None, (
            entry.get("elpd_best_reason")
            or f"fitted-seed baseline undefined at experiment {experiment}"
        )
    return value, ""


def aggregate_holdout_trajectories(
    results: Iterable[Mapping[str, Any]],
    *,
    metric: str = "pearson_r",
    error: str = "sem",
    labels: "Sequence[str] | None" = None,
) -> Mapping[str, Any]:
    """Pool several single-cell holdout results into mean ± spread per aligned
    position, with the baselines over the same cells.

    ``results`` is one decoded ``holdout.json`` per cell (each with a
    ``gt_runs`` list); ``labels`` names each result's cell (``run<r>/<gt>``) in
    the exclusion list (default: its ``run_root``). Ground truths are pooled by
    name.

    **Positions.** Cells are aligned by experiment, not by their running
    ``global_step``: an abandoned candidate round writes no step, so step k of
    one cell used to be averaged with a different round of another. Within
    each experiment the positions are its seed step, each round every cell
    recorded (by ``iteration``), and its end — every cell's last step. Each
    point carries ``x`` (consecutive over the panel), ``experiment`` and
    ``label`` (``seed``, ``round <i>``, ``end``).

    **Same cells.** At each position a cell counts only if the loop's value
    (best model and model average) and every baseline's are defined there;
    otherwise it is left out of all of them and listed in ``excluded`` with
    the reason. The baselines, per position (``baseline_series``): the
    fitted-seed baseline of that experiment (``elpd_best_*`` of
    ``fitted_baseline_by_experiment``, fit on the same cumulative data as the
    loop's steps in that experiment; a result scored before it existed joins
    only at its final experiment), and for Pearson r the best default-params
    seed (it has no RMSE). ``baselines`` keeps the headline: each baseline at
    the end of the final experiment, over the cells of the loop's final point
    (it was a flat final-data value over whichever cells defined it).
    Undefined values (``None``, ``NaN``, ``inf``) count as undefined, never as
    zero.
    """
    if metric not in _HOLDOUT_METRIC_SPECS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of "
            f"{sorted(_HOLDOUT_METRIC_SPECS)}"
        )
    if error not in _ERROR_KINDS:
        raise ValueError(
            f"Unknown error {error!r}; expected one of {list(_ERROR_KINDS)}"
        )
    spec = _HOLDOUT_METRIC_SPECS[metric]
    results = list(results)
    if labels is not None and len(labels) != len(results):
        raise ValueError(f"{len(labels)} labels for {len(results)} results.")

    # Pool every ground truth's cells, preserving first-seen order across files.
    runs_by_model: "defaultdict[str, List[tuple]]" = defaultdict(list)
    for index, result in enumerate(results):
        for gt_run in result["gt_runs"]:
            label = (
                labels[index] if labels is not None
                else str(gt_run.get("run_root") or f"result{index}")
            )
            runs_by_model[gt_run["gt_model"]].append((label, gt_run))

    series_keys = {"best": spec["best_key"], "bma": spec["bma_key"]}
    baseline_keys = ["fitted_baseline"] + (["baseline"] if metric == "pearson_r" else [])

    panels: List[Mapping[str, Any]] = []
    for name in sorted(runs_by_model):
        cells = [
            (label, gt_run, _experiment_rows(gt_run["trajectory"]))
            for label, gt_run in runs_by_model[name]
        ]
        experiments = sorted({e for _, _, rows in cells for e in rows})
        final_experiment = experiments[-1] if experiments else 0
        best: List[Mapping[str, Any]] = []
        bma: List[Mapping[str, Any]] = []
        baseline_series: dict = {key: [] for key in baseline_keys}
        positions: List[Mapping[str, Any]] = []
        excluded: List[Mapping[str, Any]] = []
        boundaries: List[int] = []
        x = 0
        for experiment in experiments:
            present = [(label, gt_run, rows[experiment]) for label, gt_run, rows in cells
                       if experiment in rows]
            for label, _, rows in cells:
                if experiment not in rows:
                    excluded.append({"cell": label, "experiment": experiment,
                                     "label": "all", "reason": f"no experiment {experiment}"})
            if not present:
                continue
            if experiment > experiments[0]:
                boundaries.append(x)
            for position in _aligned_labels([rows for _, _, rows in present]):
                values: dict = {key: [] for key in (*series_keys, *baseline_keys)}
                included: List[str] = []
                for label, gt_run, rows in present:
                    row = _row_at(rows, position)
                    value = {key: row.get(field) for key, field in series_keys.items()}
                    reason = next(
                        (f"loop {series_keys[key]} undefined" for key in series_keys
                         if not _is_finite(value[key])),
                        "",
                    )
                    if not reason:
                        value["fitted_baseline"], why = _fitted_seed_value(
                            gt_run, experiment, final_experiment, spec["fitted_baseline_field"]
                        )
                        if why:
                            reason = f"fitted-seed baseline: {why}"
                    if not reason and "baseline" in baseline_keys:
                        value["baseline"] = _best_seed_value(
                            gt_run.get("baseline") or {}, "baseline", spec
                        )
                        if not _is_finite(value["baseline"]):
                            reason = "default-params baseline undefined"
                    if reason:
                        excluded.append({"cell": label, "experiment": experiment,
                                         "label": position, "reason": reason})
                        continue
                    included.append(label)
                    for key in values:
                        values[key].append(value[key])
                positions.append({"x": x, "experiment": experiment, "label": position,
                                  "cells": included})
                if included:
                    where = {"x": x, "experiment": experiment, "label": position}
                    best.append({**where, **_summarize(values["best"], error)})
                    bma.append({**where, **_summarize(values["bma"], error)})
                    for key in baseline_keys:
                        baseline_series[key].append({**where, **_summarize(values[key], error)})
                x += 1

        # The headline: each baseline at the loop's final point (the end of
        # the final experiment), over the same cells.
        final = best[-1] if best and best[-1]["experiment"] == final_experiment else None
        headline = {"fitted_baseline": None, "baseline": None}
        if final is not None:
            for key in baseline_keys:
                headline[key] = next(p for p in baseline_series[key] if p["x"] == final["x"])
        panels.append(
            {
                "gt_model": name,
                "n_runs": len(cells),
                "positions": positions,
                "best": best,
                "bma": bma,
                "baseline_series": baseline_series,
                "baselines": headline,
                "experiment_boundaries": boundaries,
                "excluded": excluded,
            }
        )

    return {"metric": metric, "error": error, "gt_models": panels}


# One tidy row per pooled point.
# ``position``/``experiment``/``label`` replace the old ``global_step`` column:
# points are aligned by experiment now, and the baselines are per position.
AGGREGATE_TIDY_COLUMNS = [
    "gt_model", "metric", "error", "series", "position", "experiment", "label",
    "mean", "err", "n",
]


def aggregate_tidy_rows(aggregated: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Flatten a pooled aggregate into one tidy row per plotted point: the
    best-model series (``best``) and each baseline, at every position."""
    metric, error = aggregated["metric"], aggregated["error"]
    rows: List[Mapping[str, Any]] = []
    for panel in aggregated["gt_models"]:
        series = {"best": panel["best"], **panel["baseline_series"]}
        for name, points in series.items():
            for point in points:
                rows.append({
                    "gt_model": panel["gt_model"], "metric": metric, "error": error,
                    "series": name, "position": point["x"],
                    "experiment": point["experiment"], "label": point["label"],
                    "mean": point["mean"], "err": point["err"], "n": point["n"],
                })
    return rows


def aggregate_exclusion_lines(aggregated: Mapping[str, Any]) -> List[str]:
    """Markdown: per ground truth, the cells each position leaves out and why."""
    lines = [f"## Cells left out of points ({aggregated['metric']})", ""]
    for panel in aggregated["gt_models"]:
        lines.append(f"### {panel['gt_model']}: {panel['n_runs']} complete cell(s)")
        lines.append("")
        if not panel["excluded"]:
            lines.append("Every position covers every complete cell.")
        for entry in panel["excluded"]:
            lines.append(
                f"- `{entry['cell']}` experiment {entry['experiment']} "
                f"{entry['label']}: {entry['reason']}"
            )
        lines.append("")
    return lines


# Per-series labels and styling for the combined (plotnine) holdout figure.
_BEST_LABEL = "best model"
_DEFAULT_PARAMS_LABEL = "best seed (default params)"
# Default display label for the fitted-seed baseline. In holdout recovery the
# ground truth is itself a seed model, so the baseline is the best of the *other*
# seed models. Impossible recovery holds out no seed (the ground truth lies
# outside the seed family), so its caller passes "best seed model" instead.
DEFAULT_FITTED_BASELINE_LABEL = "ELPD-best other seed model"

# The x axis counts inner-loop scoring steps in the full pipeline. The
# no-inner-loop ablation has a single step per experiment, so its callers pass
# a different label (e.g. "experiment").
DEFAULT_TRAJECTORY_X_LABEL = "position in experiment (seed, shared rounds, end)"

# Horizontal inset (in step units) of an "exp. round N" label from the left edge
# of its experiment region, so the left-justified text clears the boundary line.
_ROUND_LABEL_X_PAD = 0.25

# ColorBrewer Dark2 (qualitative), keyed by role so the fitted baseline can be
# relabelled without losing its color/linetype.
_SERIES_COLOR_BY_ROLE = {"best": "#1B9E77", "fitted": "#D95F02", "default": "#7570B3"}
_SERIES_LINETYPE_BY_ROLE = {"best": "solid", "fitted": "dotted", "default": "dashdot"}


def _series_styling(fitted_baseline_label: str):
    """Per-figure series labels, draw order, and color/linetype maps.

    Returns ``(baseline_labels, order, colors, linetypes)`` where
    ``baseline_labels`` maps the aggregate's baseline keys
    (``fitted_baseline``/``baseline``) to display labels, and the color/linetype
    maps are keyed by display label for plotnine's ``scale_*_manual``.
    """
    role_label = {
        "best": _BEST_LABEL,
        "fitted": fitted_baseline_label,
        "default": _DEFAULT_PARAMS_LABEL,
    }
    order = [role_label["best"], role_label["fitted"], role_label["default"]]
    colors = {role_label[r]: _SERIES_COLOR_BY_ROLE[r] for r in role_label}
    linetypes = {role_label[r]: _SERIES_LINETYPE_BY_ROLE[r] for r in role_label}
    baseline_labels = {
        "fitted_baseline": fitted_baseline_label,
        "baseline": _DEFAULT_PARAMS_LABEL,
    }
    return baseline_labels, order, colors, linetypes


# Half the width, in position units, of a baseline's bar at one position.
_BASELINE_HALF_WIDTH = 0.45


def holdout_combined_frames(
    aggregated: Mapping[str, Any],
    *,
    fitted_baseline_label: str = DEFAULT_FITTED_BASELINE_LABEL,
) -> Mapping[str, Any]:
    """Flatten a pooled aggregate into tidy data frames for plotnine.

    Returns a mapping with the run aggregate's ``metric`` and ``error`` plus
    four ``pandas`` data frames, all faceted by a ``facet`` column (the
    ground-truth name with underscores spaced out, an ordered categorical so
    panels stay in ground-truth order):

    * ``trajectory`` — one row per aligned position of the best-model series
      (``position`` is its x; ``experiment`` and ``label`` say which), with
      ``mean`` and ``ymin``/``ymax`` = mean ± spread for the error bars.
    * ``baselines`` — one row per baseline per position, over the same cells
      as the trajectory's point there, with the same ``mean``/``ymin``/``ymax``
      plus ``xmin``/``xmax``, a short bar around the position (the fitted-seed
      baseline changes with each experiment's data).
    * ``boundaries`` — one row per outer-experiment boundary, with ``boundary``
      (the first position of a new experiment) and ``x`` = ``boundary - 0.5``
      (where the marker line is drawn).
    * ``rounds`` — the "exp. round N" labels, on the leftmost panel only: one
      row per experiment region (the spans between the boundary lines) of that
      panel, with ``round`` (1-based), a two-line ``label`` (``"exp.\\nround
      N"``), and ``x``, the left-justified text position just inside the region's
      left edge.
    """
    import pandas as pd

    baseline_labels, series_order, _, _ = _series_styling(fitted_baseline_label)
    panels = aggregated["gt_models"]
    facet_order = [p["gt_model"].replace("_", " ") for p in panels]
    all_positions = [pt["x"] for p in panels for pt in p["best"]]
    xmin = (min(all_positions) - 0.5) if all_positions else -0.5

    traj_rows: List[Mapping[str, Any]] = []
    baseline_rows: List[Mapping[str, Any]] = []
    boundary_rows: List[Mapping[str, Any]] = []
    round_rows: List[Mapping[str, Any]] = []
    for panel in panels:
        gt = panel["gt_model"]
        facet = gt.replace("_", " ")
        for point in panel["best"]:
            traj_rows.append(
                {
                    "gt_model": gt,
                    "facet": facet,
                    "series": _BEST_LABEL,
                    "position": point["x"],
                    "experiment": point["experiment"],
                    "label": point["label"],
                    "mean": point["mean"],
                    "err": point["err"],
                    "ymin": point["mean"] - point["err"],
                    "ymax": point["mean"] + point["err"],
                    "n": point["n"],
                }
            )
        for run_key, label in baseline_labels.items():
            for point in panel["baseline_series"].get(run_key, []):
                baseline_rows.append(
                    {
                        "gt_model": gt,
                        "facet": facet,
                        "series": label,
                        "position": point["x"],
                        "experiment": point["experiment"],
                        "mean": point["mean"],
                        "err": point["err"],
                        "ymin": point["mean"] - point["err"],
                        "ymax": point["mean"] + point["err"],
                        "xmin": point["x"] - _BASELINE_HALF_WIDTH,
                        "xmax": point["x"] + _BASELINE_HALF_WIDTH,
                        "n": point["n"],
                    }
                )
        for boundary in panel["experiment_boundaries"]:
            boundary_rows.append(
                {
                    "gt_model": gt,
                    "facet": facet,
                    "boundary": boundary,
                    "x": boundary - 0.5,
                }
            )

    # "exp. round N" labels go on the leftmost panel only (one shared key for the
    # whole figure, not repeated in every facet). The boundary lines (at
    # boundary - 0.5) cut the axis into one region per experiment; label each
    # region just inside its left edge.
    if panels:
        first = panels[0]
        region_lefts = [xmin] + [b - 0.5 for b in first["experiment_boundaries"]]
        for index, left in enumerate(region_lefts):
            round_number = index + 1
            round_rows.append(
                {
                    "gt_model": first["gt_model"],
                    "facet": first["gt_model"].replace("_", " "),
                    "round": round_number,
                    "label": f"exp.\nround {round_number}",
                    "x": left + _ROUND_LABEL_X_PAD,
                }
            )

    # Restrict the series legend to series that actually appear (e.g. the
    # default-param baseline has no RMSE, so it must not show up in the RMSE
    # legend), keeping the canonical order.
    present = {row["series"] for row in (*traj_rows, *baseline_rows)}
    series_categories = [s for s in series_order if s in present]

    def _framed(rows: List[Mapping[str, Any]], columns: List[str]):
        df = pd.DataFrame(rows, columns=columns)
        df["facet"] = pd.Categorical(df["facet"], categories=facet_order, ordered=True)
        if "series" in df.columns:
            df["series"] = pd.Categorical(
                df["series"], categories=series_categories, ordered=True
            )
        return df

    return {
        "metric": aggregated["metric"],
        "error": aggregated["error"],
        "trajectory": _framed(
            traj_rows,
            ["gt_model", "facet", "series", "position", "experiment", "label",
             "mean", "err", "ymin", "ymax", "n"],
        ),
        "baselines": _framed(
            baseline_rows,
            ["gt_model", "facet", "series", "position", "experiment", "mean", "err",
             "ymin", "ymax", "xmin", "xmax", "n"],
        ),
        "boundaries": _framed(boundary_rows, ["gt_model", "facet", "boundary", "x"]),
        "rounds": _framed(
            round_rows, ["gt_model", "facet", "round", "label", "x"]
        ),
    }


def _round_label_placement(spec: Mapping[str, Any], trajectory, baselines):
    """Where to seat the "exp. round N" labels and the y-limit that gives them room.

    Returns ``(label_y, ylim)``. The labels always sit in a band above the data:

    * A fixed metric axis (Pearson r, near the +1 ceiling) leaves no headroom, so
      the ceiling is lifted above the window's top and ``ylim`` is the raised
      window. The labels go in the new band.
    * RMSE autoscales up from 0, so the labels go a margin above the highest
      plotted point (an error-bar or baseline-band top); the mapped y then lifts
      the axis on its own and ``ylim`` is ``None``.
    """
    if spec["ylim"] is not None:
        lo, hi = spec["ylim"]
        span = hi - lo
        return hi + 0.10 * span, (lo, hi + 0.16 * span)
    tops = [
        value
        for value in (*trajectory["ymax"], *baselines["ymax"])
        if _is_finite(value)
    ]
    return (max(tops) if tops else 1.0) * 1.3, None


def holdout_trajectories_ggplot(
    aggregated: Mapping[str, Any],
    *,
    fitted_baseline_label: str = DEFAULT_FITTED_BASELINE_LABEL,
    x_label: str = DEFAULT_TRAJECTORY_X_LABEL,
    strip_text_size: float = 17,
):
    """Build the pooled holdout-recovery figure as a plotnine ``ggplot``.

    One facet per held-out model: the best-model recovery trajectory is a mean
    line with per-position error bars (positions aligned by experiment, see
    ``aggregate_holdout_trajectories``), and at every position each baseline
    (averaged over the same cells) gets a short bar with a shaded ± spread
    band. Returning the unsaved ``ggplot`` lets the caller tweak formatting
    before rendering — e.g.::

        from src.subjective_randomness.reporting import holdout_trajectories_ggplot
        from plotnine import theme, element_text
        p = holdout_trajectories_ggplot(aggregated)
        (p + theme(figure_size=(16, 4))).save("holdout.png", dpi=300)

    Colors, line types, and series order come from :func:`_series_styling`;
    ``fitted_baseline_label`` renames the fitted-seed baseline series (holdout
    recovery uses the default "best other seed model"; impossible recovery, which
    holds out no seed, passes "best seed model"). ``strip_text_size`` sets the
    per-panel heading font in points; callers with long facet names (impossible
    recovery) pass a smaller value so adjacent headings do not overlap.
    """
    from plotnine import (
        aes,
        coord_cartesian,
        element_blank,
        element_text,
        expand_limits,
        facet_wrap,
        geom_errorbar,
        geom_hline,
        geom_line,
        geom_point,
        geom_rect,
        geom_segment,
        geom_text,
        geom_vline,
        ggplot,
        labs,
        scale_color_manual,
        scale_fill_manual,
        scale_linetype_manual,
        scale_x_continuous,
        theme,
        theme_minimal,
    )

    spec = _HOLDOUT_METRIC_SPECS[aggregated["metric"]]
    _, _, series_colors, series_linetypes = _series_styling(fitted_baseline_label)
    frames = holdout_combined_frames(
        aggregated, fitted_baseline_label=fitted_baseline_label
    )
    trajectory, baselines, boundaries, rounds = (
        frames["trajectory"],
        frames["baselines"],
        frames["boundaries"],
        frames["rounds"],
    )
    n_panels = max(len(aggregated["gt_models"]), 1)
    # Integer x ticks at the aligned positions (no 2.5/7.5 fractions).
    x_breaks = sorted(
        {pt["x"] for p in aggregated["gt_models"] for pt in p["best"]}
    )

    plot = (
        ggplot()
        # Baselines, per position (the fitted seeds follow each experiment's
        # data): a ± spread band behind a short bar at the mean.
        + geom_rect(
            baselines,
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="series"),
            alpha=0.15,
            show_legend=False,
        )
        + geom_segment(
            baselines,
            aes(x="xmin", xend="xmax", y="mean", yend="mean", color="series",
                linetype="series"),
            size=1.0,
        )
        # Best-model recovery trajectory: line + error bars + points.
        + geom_line(
            trajectory,
            aes(x="position", y="mean", color="series", linetype="series"),
            size=0.8,
        )
        + geom_errorbar(
            trajectory,
            aes(x="position", ymin="ymin", ymax="ymax", color="series"),
            width=0.3,
            size=0.6,
        )
        + geom_point(
            trajectory, aes(x="position", y="mean", color="series"), size=2.2
        )
        + facet_wrap("facet", nrow=1)
        + scale_x_continuous(breaks=x_breaks)
        + scale_color_manual(values=series_colors, name="")
        + scale_fill_manual(values=series_colors, name="")
        + scale_linetype_manual(values=series_linetypes, name="")
        + labs(x=x_label, y=spec["combined_ylabel"])
        + theme_minimal()
        # Compact layout (tight panels), but larger, legible text throughout.
        + theme(
            figure_size=(3.2 * n_panels, 3.0),
            legend_position="bottom",
            legend_title=element_blank(),
            legend_box_spacing=0.0,
            axis_title=element_text(size=17),
            axis_text=element_text(size=14),
            strip_text=element_text(size=strip_text_size),
            legend_text=element_text(size=15),
            panel_spacing=0.02,
        )
    )

    # Outer-experiment boundaries (skip the geom when none — it errors on empty data).
    if len(boundaries):
        plot = plot + geom_vline(
            boundaries,
            aes(xintercept="x"),
            linetype="dotted",
            color="grey",
            size=1.5,
        )
    # "exp. round N" labels for the regions between the boundaries, seated in a
    # band above the data. The y is carried as a mapped column, not a constant
    # param, so it trains the scale and lifts the autoscaled RMSE axis; the fixed
    # Pearson-r axis instead gets its ceiling raised below (see label_ylim).
    label_y, label_ylim = _round_label_placement(spec, trajectory, baselines)
    if len(rounds):
        plot = plot + geom_text(
            rounds.assign(y=label_y),
            aes(x="x", y="y", label="label"),
            ha="left",
            va="top",
            size=12,
            color="#444444",
            lineheight=0.9,
        )
    if spec["zero_line"]:
        plot = plot + geom_hline(yintercept=0.0, color="grey", size=0.3)
    if spec["ylim"] is not None:
        # Use the label-aware window (raised ceiling) when labels are drawn.
        plot = plot + coord_cartesian(ylim=label_ylim if len(rounds) else spec["ylim"])
    else:
        plot = plot + expand_limits(y=0.0)  # RMSE: keep the floor at 0
    return plot


def plot_holdout_trajectories_combined(
    aggregated: Mapping[str, Any],
    out_path: Path,
    *,
    fitted_baseline_label: str = DEFAULT_FITTED_BASELINE_LABEL,
    x_label: str = DEFAULT_TRAJECTORY_X_LABEL,
    strip_text_size: float = 17,
) -> None:
    """Render :func:`holdout_trajectories_ggplot` to ``out_path``.

    A thin save wrapper kept for the CLI and back-compatibility; for manual
    formatting, build the figure with :func:`holdout_trajectories_ggplot` and
    save it yourself. ``fitted_baseline_label`` renames the fitted-seed baseline
    series, ``x_label`` renames the x axis, and ``strip_text_size`` sets the
    per-panel heading font (shrunk by callers with long facet names, e.g.
    impossible recovery) (see :func:`holdout_trajectories_ggplot`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" trims margins to the content so the compact panels keep
    # their size while the title and axis labels never clip at the figure edge.
    holdout_trajectories_ggplot(
        aggregated,
        fitted_baseline_label=fitted_baseline_label,
        x_label=x_label,
        strip_text_size=strip_text_size,
    ).save(out_path, dpi=150, verbose=False, bbox_inches="tight")


def plot_model_recovery(confusion: Mapping[str, Any], out_path: Path) -> None:
    """Write the generating x recovered posterior confusion heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = confusion["seed_models"]
    by_gen = {e["generating_model"]: e["posteriors"] for e in confusion["generating"]}
    gens = [m for m in models if m in by_gen]
    matrix = np.array([[by_gen[g].get(r, 0.0) for r in models] for g in gens])

    fig, ax = plt.subplots(figsize=(1.4 * len(models) + 2, 1.4 * len(gens) + 2))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
    ax.set_yticks(range(len(gens)), gens)
    ax.set_xlabel("recovered model")
    ax.set_ylabel("generating model")
    ax.set_title("Model recovery — posterior confusion")
    for i in range(len(gens)):
        for j in range(len(models)):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="posterior probability")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
