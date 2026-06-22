"""Human-readable text blocks and figures for recovery results.

One implementation shared by the `analyze_recovery.py` CLI (which prints the
text and writes one figure) and the pipeline runner (which aggregates the same
text into a key-results file alongside the figures):

* `parameter_recovery_text` / `plot_parameter_recovery` — per-parameter
  recovery quality for a `pymc_recover.py` report. Sampled-truth reports get a
  ground-truth vs. recovered correlation scatter per parameter; fixed-truth
  reports get the estimate spread around the single true value.

* `model_recovery_text` / `plot_model_recovery` — per-generating-model
  recovery and the posterior confusion heatmap for a closed-ended
  `model_recovery.py` result.

* `selection_comparison_parameter_text` / `selection_comparison_model_text` /
  `plot_selection_comparison_parameters` / `plot_selection_comparison_models`
  — side-by-side EIG-optimized vs. random stimulus sets for the
  `adaptive_recovery.compare_*` reports.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean
from statistics import stdev as _stdev
from typing import Any, Iterable, List, Mapping

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


def selection_comparison_parameter_text(report: Mapping[str, Any]) -> str:
    """EIG-optimized vs. random stimulus set, per parameter, one text table."""
    arms = report["arms"]
    lines = [
        f"Stimulus-selection comparison — model: {report['model']} "
        "(EIG-optimized vs. random stimulus set; grid-posterior recovery)"
    ]
    lines.append(
        f"  repeats: {report['n_repeats']} | stimuli/set: {report['n_stimuli']} | "
        f"participants/stimulus: {report['n_participants']}"
    )
    lines.append(
        "  mean EIG of the chosen set (bits): "
        + " vs. ".join(
            f"{name} {_fmt(arm['mean_stimulus_eig'])}" for name, arm in arms.items()
        )
    )
    header = ["parameter", "r_eig", "r_random", "rmse_eig", "rmse_random"]
    lines.append("  " + "  ".join(f"{h:>14}" for h in header))
    for param in arms["eig"]["summary"]:
        eig = arms["eig"]["summary"][param]
        random_ = arms["random"]["summary"][param]
        row = [
            param,
            eig["pearson_r"],
            random_["pearson_r"],
            eig["rmse"],
            random_["rmse"],
        ]
        lines.append("  " + "  ".join(f"{_fmt(v):>14}" for v in row))
    return "\n".join(lines)


def selection_comparison_model_text(report: Mapping[str, Any]) -> str:
    """EIG-optimized vs. random stimulus set for model recovery, as text."""
    arms = report["arms"]
    lines = [
        "Model-recovery stimulus-selection comparison "
        "(EIG-optimized vs. random stimulus set; grid-posterior recovery)"
    ]
    lines.append(
        f"  {report['n_repeats']} repeats x {len(report['model_names'])} generating "
        f"models | stimuli/set: {report['n_stimuli']} | "
        f"participants/stimulus: {report['n_participants']}"
    )
    lines.append(
        "  accuracy: "
        + ", ".join(f"{name} {arm['accuracy']:.2f}" for name, arm in arms.items())
        + " | mean posterior on true model: "
        + ", ".join(
            f"{name} {arm['mean_true_posterior']:.3f}" for name, arm in arms.items()
        )
    )
    header = ["generating_model", "P(true)_eig", "P(true)_random"]
    lines.append("  " + "  ".join(f"{h:>20}" for h in header))
    for gen in report["model_names"]:
        row = [
            gen,
            f"{arms['eig']['confusion'][gen][gen]:.3f}",
            f"{arms['random']['confusion'][gen][gen]:.3f}",
        ]
        lines.append("  " + "  ".join(f"{v:>20}" for v in row))
    return "\n".join(lines)


def plot_selection_comparison_parameters(
    report: Mapping[str, Any], out_path: Path
) -> None:
    """Truth vs. recovered scatter per parameter, one row per selection rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = report["arms"]
    params = list(next(iter(arms.values()))["summary"])
    arm_colors = {"eig": "#4878CF", "random": "#D65F5F"}
    fig, axes = plt.subplots(
        len(arms),
        len(params),
        figsize=(3.4 * len(params), 3.4 * len(arms)),
        squeeze=False,
    )
    for row, (arm_name, arm) in enumerate(arms.items()):
        for col, param in enumerate(params):
            ax = axes[row][col]
            trues = [r["true_params"][param] for r in arm["runs"]]
            ests = [r["posterior_mean"][param] for r in arm["runs"]]
            lo = min(trues + ests)
            hi = max(trues + ests)
            pad = 0.05 * ((hi - lo) or 1.0)
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color="#999999",
                linestyle="--",
                linewidth=1,
            )
            ax.scatter(
                trues, ests, alpha=0.6, color=arm_colors.get(arm_name, "#4878CF")
            )
            r = arm["summary"][param]["pearson_r"]
            ax.set_title(
                f"{arm_name}: {param}" + ("" if r is None else f" (r = {r:.2f})")
            )
            ax.set_xlabel("true value")
            ax.set_ylabel("recovered estimate")
    fig.suptitle(
        f"Stimulus-selection comparison — {report['model']} "
        "(EIG-optimized vs. random set)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_selection_comparison_models(report: Mapping[str, Any], out_path: Path) -> None:
    """Mean-posterior confusion heatmap per selection rule, side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = report["model_names"]
    arms = report["arms"]
    panel = 1.4 * len(models) + 2
    fig, axes = plt.subplots(
        1, len(arms), figsize=(panel * len(arms), panel), squeeze=False
    )
    for ax, (arm_name, arm) in zip(axes[0], arms.items()):
        matrix = np.array([[arm["confusion"][g][m] for m in models] for g in models])
        im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
        ax.set_yticks(range(len(models)), models)
        ax.set_xlabel("recovered model")
        ax.set_ylabel("generating model")
        ax.set_title(f"{arm_name} (accuracy {arm['accuracy']:.2f})")
        for i in range(len(models)):
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
        fig.colorbar(im, ax=ax, label="mean posterior")
    fig.suptitle("Model recovery — EIG-optimized vs. random stimulus set")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    mark outer-experiment boundaries; flat lines mark the seed-model baselines.
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
        # Two flat seed-model baselines (constant across steps): the other seed
        # models with default params, and those seed models fit on all the data.
        for run_key, baseline_color, baseline_style, baseline_label in (
            ("fitted_baseline", SEED_FIT_COLOR, ":", "seed models (fit to all data)"),
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
    """The best (not mean) recovery among the non-held-out seed models, or None.

    Each held-out result stores every sibling seed model's recovery under
    ``per_model``. The fit-to-all-data baseline stores a ``{metric: value}``
    dict per model; the default-params baseline stores only a bare Pearson r
    per model (so it has no RMSE). "Best" is the max for a higher-is-better
    metric (Pearson r) and the min for RMSE.
    """
    per_model = baseline.get("per_model")
    if not per_model:
        return None
    metric_key = spec["best_key"]
    if run_key == "fitted_baseline":
        values = [entry.get(metric_key) for entry in per_model.values()]
    elif metric_key == "pearson_r":  # default-params baseline: bare Pearson r
        values = list(per_model.values())
    else:
        return None  # default-params baseline has no RMSE
    values = [v for v in values if _is_finite(v)]
    if not values:
        return None
    return max(values) if spec["higher_is_better"] else min(values)


def aggregate_holdout_trajectories(
    results: Iterable[Mapping[str, Any]],
    *,
    metric: str = "pearson_r",
    error: str = "sem",
) -> Mapping[str, Any]:
    """Pool several single-run holdout results into per-step mean ± spread.

    ``results`` is one decoded ``holdout.json`` per run (each a mapping with a
    ``gt_runs`` list). Ground truths are pooled by name across runs; for every
    held-out model this returns, at each inner-loop ``global_step`` that any run
    reached, the mean and ``error`` spread of the best-model and Bayesian
    model-average ``metric`` across the runs that defined a value there. Steps
    whose value is undefined in a run (``None``, ``NaN``, or ``inf`` — e.g. a
    constant prediction, common for impossible ground truths, makes correlation
    undefined) are dropped from that step's sample rather than counted as zero.
    The two flat seed-model baselines are the *best* sibling
    seed per run (not the average), pooled the same way; outer-experiment
    boundaries (where any run starts a new experiment) are collected so the
    figure can mark them.
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

    # Pool every ground truth's runs, preserving first-seen order across files.
    runs_by_model: "defaultdict[str, List[Mapping[str, Any]]]" = defaultdict(list)
    order: List[str] = []
    for result in results:
        for gt_run in result["gt_runs"]:
            name = gt_run["gt_model"]
            if name not in runs_by_model:
                order.append(name)
            runs_by_model[name].append(gt_run)

    panels: List[Mapping[str, Any]] = []
    for name in sorted(order):
        gt_runs = runs_by_model[name]

        def _series(key: str) -> List[Mapping[str, Any]]:
            by_step: "defaultdict[int, List[float]]" = defaultdict(list)
            for gt_run in gt_runs:
                for row in gt_run["trajectory"]:
                    value = row.get(key)
                    if _is_finite(value):
                        by_step[row["global_step"]].append(value)
            return [
                {"global_step": step, **_summarize(by_step[step], error)}
                for step in sorted(by_step)
            ]

        baselines: dict[str, Any] = {}
        for run_key in ("fitted_baseline", "baseline"):
            values = [
                _best_seed_value(gt_run.get(run_key) or {}, run_key, spec)
                for gt_run in gt_runs
            ]
            values = [v for v in values if _is_finite(v)]
            baselines[run_key] = _summarize(values, error) if values else None

        boundaries = sorted(
            {
                row["global_step"]
                for gt_run in gt_runs
                for row in gt_run["trajectory"]
                if row["step"] == 0 and row["experiment"] > 1
            }
        )

        panels.append(
            {
                "gt_model": name,
                "n_runs": len(gt_runs),
                "best": _series(spec["best_key"]),
                "bma": _series(spec["bma_key"]),
                "baselines": baselines,
                "experiment_boundaries": boundaries,
            }
        )

    return {"metric": metric, "error": error, "gt_models": panels}


# Per-series labels and styling for the combined (plotnine) holdout figure.
_BEST_LABEL = "best model"
_DEFAULT_PARAMS_LABEL = "best seed (default params)"
# Default display label for the fitted-seed baseline. In holdout recovery the
# ground truth is itself a seed model, so the baseline is the best of the *other*
# seed models. Impossible recovery holds out no seed (the ground truth lies
# outside the seed family), so its caller passes "best seed model" instead.
DEFAULT_FITTED_BASELINE_LABEL = "best other seed model"

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


def holdout_combined_frames(
    aggregated: Mapping[str, Any],
    *,
    fitted_baseline_label: str = DEFAULT_FITTED_BASELINE_LABEL,
) -> Mapping[str, Any]:
    """Flatten a pooled aggregate into tidy data frames for plotnine.

    Returns a mapping with the run aggregate's ``metric`` and ``error`` plus
    three ``pandas`` data frames, all faceted by a ``facet`` column (the
    ground-truth name with underscores spaced out, an ordered categorical so
    panels stay in ground-truth order):

    * ``trajectory`` — one row per inner-loop step of the best-model series,
      with ``mean`` and ``ymin``/``ymax`` = mean ± spread for the error bars.
    * ``baselines`` — one row per defined flat seed-model baseline, with the
      same ``mean``/``ymin``/``ymax`` plus ``xmin``/``xmax`` spanning the step
      range so the spread can be drawn as a horizontal band.
    * ``boundaries`` — one row per outer-experiment boundary, with ``boundary``
      (the step a new experiment begins) and ``x`` = ``boundary - 0.5`` (where
      the marker line is drawn).
    """
    import pandas as pd

    baseline_labels, series_order, _, _ = _series_styling(fitted_baseline_label)
    panels = aggregated["gt_models"]
    facet_order = [p["gt_model"].replace("_", " ") for p in panels]
    all_steps = [pt["global_step"] for p in panels for pt in p["best"]]
    xmin = (min(all_steps) - 0.5) if all_steps else -0.5
    xmax = (max(all_steps) + 0.5) if all_steps else 0.5

    traj_rows: List[Mapping[str, Any]] = []
    baseline_rows: List[Mapping[str, Any]] = []
    boundary_rows: List[Mapping[str, Any]] = []
    for panel in panels:
        gt = panel["gt_model"]
        facet = gt.replace("_", " ")
        for point in panel["best"]:
            traj_rows.append(
                {
                    "gt_model": gt,
                    "facet": facet,
                    "series": _BEST_LABEL,
                    "global_step": point["global_step"],
                    "mean": point["mean"],
                    "err": point["err"],
                    "ymin": point["mean"] - point["err"],
                    "ymax": point["mean"] + point["err"],
                    "n": point["n"],
                }
            )
        for run_key, label in baseline_labels.items():
            stats = panel["baselines"].get(run_key)
            if stats is not None:
                baseline_rows.append(
                    {
                        "gt_model": gt,
                        "facet": facet,
                        "series": label,
                        "mean": stats["mean"],
                        "err": stats["err"],
                        "ymin": stats["mean"] - stats["err"],
                        "ymax": stats["mean"] + stats["err"],
                        "xmin": xmin,
                        "xmax": xmax,
                        "n": stats["n"],
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
            [
                "gt_model",
                "facet",
                "series",
                "global_step",
                "mean",
                "err",
                "ymin",
                "ymax",
                "n",
            ],
        ),
        "baselines": _framed(
            baseline_rows,
            [
                "gt_model",
                "facet",
                "series",
                "mean",
                "err",
                "ymin",
                "ymax",
                "xmin",
                "xmax",
                "n",
            ],
        ),
        "boundaries": _framed(boundary_rows, ["gt_model", "facet", "boundary", "x"]),
    }


def holdout_trajectories_ggplot(
    aggregated: Mapping[str, Any],
    *,
    fitted_baseline_label: str = DEFAULT_FITTED_BASELINE_LABEL,
):
    """Build the pooled holdout-recovery figure as a plotnine ``ggplot``.

    One facet per held-out model: the best-model recovery trajectory is a mean
    line with per-step error bars, and the flat best-seed baselines (the best
    sibling seed model, averaged across runs) get a line with a shaded ± spread
    band. Returning the unsaved ``ggplot`` lets the caller tweak formatting
    before rendering — e.g.::

        from src.subjective_randomness.reporting import holdout_trajectories_ggplot
        from plotnine import theme, element_text
        p = holdout_trajectories_ggplot(aggregated)
        (p + theme(figure_size=(16, 4))).save("holdout.png", dpi=300)

    Colors, line types, and series order come from :func:`_series_styling`;
    ``fitted_baseline_label`` renames the fitted-seed baseline series (holdout
    recovery uses the default "best other seed model"; impossible recovery, which
    holds out no seed, passes "best seed model").
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
    trajectory, baselines, boundaries = (
        frames["trajectory"],
        frames["baselines"],
        frames["boundaries"],
    )
    n_panels = max(len(aggregated["gt_models"]), 1)
    # Integer x ticks at the actual inner-loop steps (no 2.5/7.5 fractions).
    x_breaks = sorted(
        {pt["global_step"] for p in aggregated["gt_models"] for pt in p["best"]}
    )

    plot = (
        ggplot()
        # Seed-model baseline ± spread bands, behind everything (no legend key).
        + geom_rect(
            baselines,
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="series"),
            alpha=0.15,
            show_legend=False,
        )
        # Seed-model baseline mean lines.
        + geom_hline(
            baselines,
            aes(yintercept="mean", color="series", linetype="series"),
            size=1.0,
        )
        # Best-model recovery trajectory: line + error bars + points.
        + geom_line(
            trajectory,
            aes(x="global_step", y="mean", color="series", linetype="series"),
            size=0.8,
        )
        + geom_errorbar(
            trajectory,
            aes(x="global_step", ymin="ymin", ymax="ymax", color="series"),
            width=0.3,
            size=0.6,
        )
        + geom_point(
            trajectory, aes(x="global_step", y="mean", color="series"), size=2.2
        )
        + facet_wrap("facet", nrow=1)
        + scale_x_continuous(breaks=x_breaks)
        + scale_color_manual(values=series_colors, name="")
        + scale_fill_manual(values=series_colors, name="")
        + scale_linetype_manual(values=series_linetypes, name="")
        + labs(x="inner-loop scoring step", y=spec["combined_ylabel"])
        + theme_minimal()
        # Compact layout (tight panels), but larger, legible text throughout.
        + theme(
            figure_size=(3.2 * n_panels, 3.0),
            legend_position="bottom",
            legend_title=element_blank(),
            legend_box_spacing=0.0,
            axis_title=element_text(size=13),
            axis_text=element_text(size=11),
            strip_text=element_text(size=13),
            legend_text=element_text(size=12),
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
    if spec["zero_line"]:
        plot = plot + geom_hline(yintercept=0.0, color="grey", size=0.3)
    if spec["ylim"] is not None:
        plot = plot + coord_cartesian(ylim=spec["ylim"])
    else:
        plot = plot + expand_limits(y=0.0)  # RMSE: keep the floor at 0
    return plot


def plot_holdout_trajectories_combined(
    aggregated: Mapping[str, Any],
    out_path: Path,
    *,
    fitted_baseline_label: str = DEFAULT_FITTED_BASELINE_LABEL,
) -> None:
    """Render :func:`holdout_trajectories_ggplot` to ``out_path``.

    A thin save wrapper kept for the CLI and back-compatibility; for manual
    formatting, build the figure with :func:`holdout_trajectories_ggplot` and
    save it yourself. ``fitted_baseline_label`` renames the fitted-seed baseline
    series (see :func:`holdout_trajectories_ggplot`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" trims margins to the content so the compact panels keep
    # their size while the title and axis labels never clip at the figure edge.
    holdout_trajectories_ggplot(
        aggregated, fitted_baseline_label=fitted_baseline_label
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
