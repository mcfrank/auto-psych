"""Figures and pooled summaries for holdout-recovery results."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean
from statistics import stdev as _stdev
from typing import Any, Iterable, List, Mapping, Sequence

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
            by_experiment = [
                {
                    **gt_run["fitted_baseline"],
                    "experiment": max(row["experiment"] for row in trajectory),
                }
            ]
        drawn = False
        for entry in by_experiment:
            value = entry.get(spec["fitted_baseline_field"])
            steps = [
                row["global_step"]
                for row in trajectory
                if row["experiment"] == entry["experiment"]
            ]
            if value is None or not steps:
                continue
            plotted_values.append(value)
            ax.hlines(
                value,
                min(steps) - 0.4,
                max(steps) + 0.4,
                colors=SEED_FIT_COLOR,
                linestyles=":",
                linewidth=1.6,
                label=None
                if drawn
                else "ELPD-best seed model (fit to the experiment's data)",
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
    training data, as the loop chooses its winner (``elpd_best_*``), not the
    seed with the best held-out value, which only an oracle could choose. A
    result scored before ``elpd_best_*`` existed
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


def _experiment_rows(
    trajectory: Iterable[Mapping[str, Any]],
) -> "dict[int, List[Mapping[str, Any]]]":
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
        *(
            {row["iteration"] for row in rows[1:] if row.get("iteration") is not None}
            for rows in cells
        )
    )
    labels = ["seed"] + [f"round {i}" for i in sorted(common)]
    labels = [
        label
        for label in labels
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
        entry = next(
            (e for e in by_experiment if int(e["experiment"]) == experiment), None
        )
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
        return (
            None,
            f"no {field!r} in its fitted-seed baseline (scored before it existed; re-score it)",
        )
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
    one cell can be a different round of another. Within
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
                labels[index]
                if labels is not None
                else str(gt_run.get("run_root") or f"result{index}")
            )
            runs_by_model[gt_run["gt_model"]].append((label, gt_run))

    series_keys = {"best": spec["best_key"], "bma": spec["bma_key"]}
    baseline_keys = ["fitted_baseline"] + (
        ["baseline"] if metric == "pearson_r" else []
    )

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
            present = [
                (label, gt_run, rows[experiment])
                for label, gt_run, rows in cells
                if experiment in rows
            ]
            for label, _, rows in cells:
                if experiment not in rows:
                    excluded.append(
                        {
                            "cell": label,
                            "experiment": experiment,
                            "label": "all",
                            "reason": f"no experiment {experiment}",
                        }
                    )
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
                        (
                            f"loop {series_keys[key]} undefined"
                            for key in series_keys
                            if not _is_finite(value[key])
                        ),
                        "",
                    )
                    if not reason:
                        value["fitted_baseline"], why = _fitted_seed_value(
                            gt_run,
                            experiment,
                            final_experiment,
                            spec["fitted_baseline_field"],
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
                        excluded.append(
                            {
                                "cell": label,
                                "experiment": experiment,
                                "label": position,
                                "reason": reason,
                            }
                        )
                        continue
                    included.append(label)
                    for key in values:
                        values[key].append(value[key])
                positions.append(
                    {
                        "x": x,
                        "experiment": experiment,
                        "label": position,
                        "cells": included,
                    }
                )
                if included:
                    where = {"x": x, "experiment": experiment, "label": position}
                    best.append({**where, **_summarize(values["best"], error)})
                    bma.append({**where, **_summarize(values["bma"], error)})
                    for key in baseline_keys:
                        baseline_series[key].append(
                            {**where, **_summarize(values[key], error)}
                        )
                x += 1

        # The headline: each baseline at the loop's final point (the end of
        # the final experiment), over the same cells.
        final = (
            best[-1] if best and best[-1]["experiment"] == final_experiment else None
        )
        headline = {"fitted_baseline": None, "baseline": None}
        if final is not None:
            for key in baseline_keys:
                headline[key] = next(
                    p for p in baseline_series[key] if p["x"] == final["x"]
                )
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
