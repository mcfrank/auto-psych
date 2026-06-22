"""CLI: compare seed-model fit against the best agent-proposed model (human runs).

Each live outer-loop experiment leaves one ``model_posterior.json`` under::

    <runs_root>/run<r>/subjective_randomness/experiment<e>/model_loop/

That file records every model's PSIS-LOO fit (``elpd_loo``) and an
``arviz.compare`` table (``comparison``) whose ``elpd_diff`` and ``dse`` are
measured against the overall best (rank-0) model. This script draws a forest
array — one facet per (run, experiment) — that compares the four hand-written
**seed models** against the single **best agent-proposed model**.

ELPD-LOO is a sum over trials, so its raw value is not comparable across
experiments (which differ in trial count). We therefore plot each model's
ELPD-LOO *relative to the best agent model*: the best agent sits at 0 and each
seed sits at ``elpd_loo - elpd_best_agent`` (<= 0) with a +-``dse`` horizontal
error bar (the LOO standard error of that difference). A seed whose bar clears 0
by more than ~2 dse is reliably worse-fitting than the agent's best model.

The script fails loudly when:
  * no ``model_posterior.json`` is found under ``--runs-root``;
  * any of the four seed models is missing from a cell's ``elpd_loo``;
  * the best agent model is not the overall best (rank 0) in a cell, because then
    the ``dse`` error bars are no longer measured relative to it.

Usage:
    # Default: pool data/results/human_experiment, write the figure + CSV there
    uv run python scripts/analysis/plot_human_fit_comparison.py

    # Explicit source and output directory
    uv run python scripts/analysis/plot_human_fit_comparison.py \\
        --runs-root data/results/human_experiment --out-dir /tmp/figs
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402

DEFAULT_RUNS_ROOT = Path("data/results/human_experiment")

# The four hand-written cognitive theories that seed every experiment. Defined in
# src/pipelines/outer_loop/projects/subjective_randomness/seed_models/models_manifest.yaml;
# every other model in a posterior (iterN_candidateK, inner_loop_model, and any
# carried-forward named model) was proposed by an agent.
SEED_MODELS: Tuple[str, ...] = (
    "prototype_similarity",
    "encoding_compressibility",
    "bayesian_diagnosticity",
    "window_typicality",
)

_BEST_AGENT_LABEL = "best agent model"
_SEED_KIND = "seed model"

# Tolerance for the elpd_loo-vs-compare-table consistency check. The top-level
# elpd_loo is rounded to ~4 decimals, so a difference of two such values can drift
# by ~1e-4; 1e-3 accepts that while still catching whole-unit inconsistencies.
_ELPD_CONSISTENCY_TOL = 1e-3

# y-axis category order, top to bottom: the best agent model on top, then seeds.
MODEL_LABEL_ORDER = (_BEST_AGENT_LABEL,) + SEED_MODELS

# ColorBrewer Dark2, matching the holdout figures: green = the agent's best model,
# purple = the seed baselines.
KIND_COLORS = {_BEST_AGENT_LABEL: "#1B9E77", _SEED_KIND: "#7570B3"}

# Tidy CSV columns (one row per plotted point).
TIDY_COLUMNS = [
    "run",
    "experiment",
    "model",
    "model_label",
    "kind",
    "elpd_loo",
    "elpd_rel",
    "dse",
    "xmin",
    "xmax",
]


@dataclass
class Args:
    """Compare seed-model fit to the best agent model across human-experiment runs."""

    runs_root: Path = DEFAULT_RUNS_ROOT
    """Directory holding ``run<r>/subjective_randomness/experiment<e>/model_loop``."""
    out_dir: Optional[Path] = None
    """Where to write the figure and CSV (default: ``--runs-root``)."""
    experiment: Optional[str] = None
    """Restrict to one experiment (e.g. ``experiment3`` for each run's final
    state); default shows every (run, experiment) cell. Filtered views are
    written to their own ``..._<experiment>`` filenames."""


# A letter immediately followed by a digit, e.g. the "n1" in "run1".
_LETTER_DIGIT = re.compile(r"([A-Za-z])(\d)")


def prettify_label(label: str) -> str:
    """Display label: underscores -> spaces and ``run1`` -> ``run 1``."""
    return _LETTER_DIGIT.sub(r"\1 \2", label.replace("_", " "))


def find_posterior_files(runs_root: Path) -> List[Path]:
    """All per-(run, experiment) ``model_posterior.json`` files under ``runs_root``."""
    return sorted(
        runs_root.glob(
            "run*/subjective_randomness/experiment*/model_loop/model_posterior.json"
        )
    )


def cell_label(path: Path) -> Tuple[str, str]:
    """``(run, experiment)`` parsed from a ``model_posterior.json`` path."""
    parts = path.parts
    anchor = parts.index("subjective_randomness")
    return parts[anchor - 1], parts[anchor + 1]


def load_cell(path: Path) -> Mapping[str, Any]:
    """Read one experiment's posterior into the cell shape the row builder wants."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    run, experiment = cell_label(path)
    return {
        "run": run,
        "experiment": experiment,
        "elpd_loo": payload["elpd_loo"],
        "comparison": payload["comparison"],
    }


def _rank_zero_model(comparison: Mapping[str, Any]) -> str:
    """The overall best (rank-0) model name in an ``arviz.compare`` table."""
    for name, row in comparison.items():
        if row["rank"] == 0:
            return name
    raise ValueError("comparison table has no rank-0 model")


def seed_vs_best_agent_rows(
    cell: Mapping[str, Any], seed_models: Iterable[str] = SEED_MODELS
) -> List[Mapping[str, Any]]:
    """Plotted rows for one cell: each seed model plus the best agent model.

    The best agent model is the highest-ELPD model that is not a seed. It must be
    the overall best (rank 0) so that every other model's ``dse`` — which
    ``arviz.compare`` reports relative to rank 0 — is exactly the standard error
    of its ELPD difference from the agent's best model. We refuse otherwise.
    """
    seed_models = tuple(seed_models)
    elpd = cell["elpd_loo"]
    comparison = cell["comparison"]
    run, experiment = cell["run"], cell["experiment"]

    missing = [m for m in seed_models if m not in elpd]
    if missing:
        raise ValueError(
            f"{run}/{experiment}: seed model(s) {missing} missing from elpd_loo "
            f"(have {sorted(elpd)})"
        )

    agents = [m for m in elpd if m not in seed_models]
    if not agents:
        raise ValueError(f"{run}/{experiment}: no agent-proposed models in elpd_loo")

    best_agent = max(agents, key=lambda m: elpd[m])
    rank0 = _rank_zero_model(comparison)
    if best_agent != rank0:
        raise ValueError(
            f"{run}/{experiment}: best agent model {best_agent!r} is not the overall "
            f"best (rank 0 is {rank0!r}); dse error bars are not measured relative to "
            f"the best agent, so the comparison would be wrong."
        )

    rows: List[Mapping[str, Any]] = []
    for model in (best_agent, *seed_models):
        comp = comparison[model]
        elpd_rel = -float(comp["elpd_diff"])  # 0 for the best agent, <= 0 for seeds
        # Cross-check the compare table against the raw ELPD values; fail if they
        # disagree by more than rounding. The stored top-level elpd_loo is rounded
        # to ~4 decimals while comparison.elpd_diff is full precision, so a
        # difference of two rounded values can drift by ~1e-4; only a whole-unit
        # mismatch signals a genuinely inconsistent posterior file.
        from_elpd = elpd[model] - elpd[best_agent]
        if abs(from_elpd - elpd_rel) > _ELPD_CONSISTENCY_TOL:
            raise ValueError(
                f"{run}/{experiment}: {model} elpd_diff disagrees with elpd_loo "
                f"({elpd_rel} vs {from_elpd})"
            )
        dse = float(comp["dse"])
        is_best_agent = model == best_agent
        rows.append(
            {
                "run": run,
                "experiment": experiment,
                "model": model,
                "model_label": _BEST_AGENT_LABEL if is_best_agent else model,
                "kind": _BEST_AGENT_LABEL if is_best_agent else _SEED_KIND,
                "elpd_loo": float(elpd[model]),
                "elpd_rel": elpd_rel,
                "dse": dse,
                "xmin": elpd_rel - dse,
                "xmax": elpd_rel + dse,
            }
        )
    return rows


def select_cells(
    cells: Iterable[Mapping[str, Any]], experiment: Optional[str]
) -> List[Mapping[str, Any]]:
    """Keep only one experiment's cells, or all of them when ``experiment`` is None."""
    cells = list(cells)
    if experiment is None:
        return cells
    chosen = [c for c in cells if c["experiment"] == experiment]
    if not chosen:
        available = sorted({c["experiment"] for c in cells})
        raise ValueError(
            f"no cells for experiment {experiment!r}; available: {available}"
        )
    return chosen


def human_fit_rows(
    cells: Iterable[Mapping[str, Any]], seed_models: Iterable[str] = SEED_MODELS
) -> List[Mapping[str, Any]]:
    """Concatenate :func:`seed_vs_best_agent_rows` over every cell."""
    rows: List[Mapping[str, Any]] = []
    for cell in cells:
        rows.extend(seed_vs_best_agent_rows(cell, seed_models))
    return rows


def human_fit_forest_frame(rows: Iterable[Mapping[str, Any]]):
    """A ``pandas`` frame with ordered categoricals for the forest facets.

    ``model_label`` is ordered so the best agent model lands on top of each facet
    (plotnine draws the last category at the top); ``run`` and ``experiment`` are
    ordered so the facet grid stays in run/experiment order.
    """
    import pandas as pd

    df = pd.DataFrame(list(rows), columns=TIDY_COLUMNS)
    if df.empty:
        raise ValueError("no rows to plot")

    # Prettify the labels used for display (underscores -> spaces, run1 -> run 1);
    # the raw machine-readable values stay in the tidy CSV, which is built from rows.
    df["model_label"] = df["model_label"].map(prettify_label)
    df["run"] = df["run"].map(prettify_label)
    df["experiment"] = df["experiment"].map(prettify_label)

    # Last category == top of the y-axis, so reverse the top-to-bottom order.
    y_categories = [prettify_label(m) for m in reversed(MODEL_LABEL_ORDER)]
    df["model_label"] = pd.Categorical(
        df["model_label"], categories=y_categories, ordered=True
    )
    df["kind"] = pd.Categorical(
        df["kind"], categories=[_BEST_AGENT_LABEL, _SEED_KIND], ordered=True
    )
    df["run"] = pd.Categorical(
        df["run"], categories=sorted(df["run"].unique()), ordered=True
    )
    df["experiment"] = pd.Categorical(
        df["experiment"], categories=sorted(df["experiment"].unique()), ordered=True
    )
    return df


def _present(frame, column):
    """The categories of ``column`` that actually occur in ``frame``, in order."""
    return [c for c in frame[column].cat.categories if (frame[column] == c).any()]


def human_fit_forest_ggplot(frame):
    """Build the seed-vs-best-agent forest array as a plotnine ``ggplot``.

    Each facet's best agent model is the reference at 0 (dotted vertical line) and
    every model is a point at its ELPD-LOO relative to the best agent, with a
    +-``dse`` horizontal error bar. Faceting adapts to what is present:

    * many runs and experiments -> a run-by-experiment grid, x free per column
      (ELPD scale grows with an experiment's trial count);
    * a single experiment (e.g. each run's final state) -> one row of run panels
      sharing an x-axis, so the runs can be compared directly;
    * a single run -> one row of experiment panels, x free per panel.
    """
    from plotnine import (
        aes,
        element_text,
        facet_grid,
        facet_wrap,
        geom_errorbarh,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        scale_color_manual,
        theme,
        theme_minimal,
    )

    runs = _present(frame, "run")
    experiments = _present(frame, "experiment")

    if len(experiments) == 1:
        # One panel per run, shared x so gaps are comparable across runs.
        facet = facet_wrap("run", nrow=1)
        figure_size = (3.6 * len(runs), 3.6)
    elif len(runs) == 1:
        facet = facet_wrap("experiment", nrow=1, scales="free_x")
        figure_size = (3.6 * len(experiments), 3.6)
    else:
        facet = facet_grid("run ~ experiment", scales="free_x")
        figure_size = (3.6 * len(experiments), 2.6 * len(runs))

    return (
        ggplot(frame, aes(x="elpd_rel", y="model_label", color="kind"))
        # The best agent model is the reference at 0.
        + geom_vline(xintercept=0.0, linetype="dotted", color="grey", size=0.5)
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.3, size=0.6)
        + geom_point(size=2.4)
        + facet
        + scale_color_manual(values=KIND_COLORS, name="")
        + labs(
            x="ELPD-LOO relative to best agent model (± dse)",
            y="",
        )
        + theme_minimal()
        + theme(
            figure_size=figure_size,
            # The y-axis already labels every model, so the color legend is
            # redundant — drop it and let the rows speak for themselves.
            legend_position="none",
            axis_title=element_text(size=17),
            axis_text=element_text(size=15),
            strip_text=element_text(size=17),
            panel_spacing=0.04,
        )
    )


def plot_human_fit_forest(frame, out_path: Path) -> None:
    """Render :func:`human_fit_forest_ggplot` to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    human_fit_forest_ggplot(frame).save(
        out_path, dpi=150, verbose=False, bbox_inches="tight"
    )


def main(args: Args) -> None:
    runs_root = resolve_path(args.runs_root)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else runs_root

    files = find_posterior_files(runs_root)
    if not files:
        raise FileNotFoundError(
            f"No run files matched {runs_root}/run*/subjective_randomness/"
            f"experiment*/model_loop/model_posterior.json — nothing to plot."
        )
    cells = select_cells((load_cell(path) for path in files), args.experiment)
    scope = args.experiment or "all experiments"
    print(f"Comparing seed vs best-agent fit across {len(cells)} cell(s) ({scope})")

    rows = human_fit_rows(cells)
    frame = human_fit_forest_frame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "human_fit_comparison" + (f"_{args.experiment}" if args.experiment else "")

    figure_path = out_dir / f"{stem}.pdf"
    plot_human_fit_forest(frame, figure_path)
    print(f"Wrote figure to {figure_path}")

    csv_path = out_dir / f"{stem}.csv"
    write_tidy_csv(rows, csv_path, columns=TIDY_COLUMNS)
    print(f"Wrote tidy CSV to {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
