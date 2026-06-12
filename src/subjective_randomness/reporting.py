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

from pathlib import Path
from typing import Any, Mapping

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


def plot_selection_comparison_models(
    report: Mapping[str, Any], out_path: Path
) -> None:
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
        matrix = np.array(
            [[arm["confusion"][g][m] for m in models] for g in models]
        )
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
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
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
        row["parameter"]: row["pearson_r"]
        for row in parameter_recovery_summary(report)
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
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="posterior probability")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
