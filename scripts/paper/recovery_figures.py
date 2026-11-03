"""Model-recovery figures for the number game and the reference games.

- Number game (``fig_recovery_number_game``): one panel per held-out ground
  truth of ``data/results/number_game_recovery-oct26/run1/`` (one repeat). The
  RMSE to the ground truth of the loop's best model at every scoring step,
  against the ELPD-best other seed fitted to all of the cell's data (the
  subjective-randomness recovery figure's baseline). The
  ``bigelow_piantadosi_independent`` cell has no result (its scoring job ran
  out of memory) and is not drawn.
- Reference games (``fig_recovery_rsa``): the four recovery cells of Sherlock
  run 2 (inner loop only, on the published displays). Each step's best
  model's pool RMSE to the ground truth, against the starting model that was
  best at step 0, all refitted by ``rsa_recovery_trajectory.py``
  (``data/paper/rsa_recovery_trajectory.json``).

Usage:
    uv run python -m scripts.paper.recovery_figures
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro
from plotnine import (
    aes,
    annotate,
    element_blank,
    element_text,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_segment,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    theme,
    theme_minimal,
)
from pyprojroot import here

BEST_COLOR = "#1B9E77"  # ColorBrewer Dark2, as the other paper figures
SEED_COLOR = "#D95F02"
NG_GT_NAMES = {
    "tenenbaum_rules_similarity": "Ground truth: Tenenbaum (2000)",
    "bigelow_piantadosi_interval": "Ground truth: Bigelow & Piantadosi, interval variant",
}
RSA_GT_NAMES = {"literal": "literal listener", "salience": "salience listener"}
STEPS_PER_EXPERIMENT = 6


@dataclass
class Args:
    ng_root: Path = here() / "data/results/number_game_recovery-oct26/run1"
    """The number-game recovery sweep's cells (``<ground truth>/holdout.{json,csv}``)."""
    rsa_trajectory: Path = here() / "data/paper/rsa_recovery_trajectory.json"
    """Each recovery cell's step-best models' distances (``rsa_recovery_trajectory.py``)."""
    out_dir: Path = here() / "data/paper/figures"


def number_game_frame(root: Path) -> pd.DataFrame:
    """One row per (cell, scoring step) for the best model, plus the baseline per cell."""
    rows = []
    for gt in NG_GT_NAMES:
        holdout = json.loads((root / gt / "holdout.json").read_text())["gt_runs"][0]
        per_model = holdout["fitted_baseline"]["per_model"]
        seed = max(per_model, key=lambda m: per_model[m]["elpd_loo"])
        trajectory = pd.read_csv(root / gt / "holdout.csv")
        for step, rmse in zip(trajectory["global_step"], trajectory["rmse"]):
            rows.append(dict(gt=NG_GT_NAMES[gt], step=step, rmse=rmse, series="best model"))
        for step in trajectory["global_step"]:
            rows.append(dict(gt=NG_GT_NAMES[gt], step=step, rmse=per_model[seed]["rmse"],
                             series="best other seed model"))
    out = pd.DataFrame(rows)
    out["gt"] = pd.Categorical(out["gt"], categories=list(NG_GT_NAMES.values()))
    return out


def number_game_plot(df: pd.DataFrame) -> ggplot:
    n_steps = int(df["step"].max()) + 1
    boundaries = [k * STEPS_PER_EXPERIMENT - 0.5 for k in range(1, n_steps // STEPS_PER_EXPERIMENT)]
    best = df[df["series"] == "best model"]
    seed = df[df["series"] == "best other seed model"]
    plot = (
        ggplot(df, aes("step", "rmse", color="series", linetype="series"))
        + geom_vline(xintercept=boundaries, linetype="dotted", color="grey", size=0.8)
        + geom_line(data=seed, size=0.8)
        + geom_line(data=best, size=0.8)
        + geom_point(data=best, size=1.6)
        + facet_wrap("gt", nrow=1)
        + scale_color_manual(values={"best model": BEST_COLOR, "best other seed model": SEED_COLOR})
        + scale_linetype_manual(values={"best model": "solid", "best other seed model": "dotted"})
        + scale_x_continuous(breaks=list(range(n_steps)))
        + scale_y_continuous(limits=(0, None))
        + labs(x="inner loop iteration", y="RMSE vs. ground truth", color="", linetype="")
        + theme_minimal()
        + theme(legend_position="bottom", panel_grid_minor=element_blank(),
                strip_text=element_text(size=10), figure_size=(9, 3.2))
    )
    for k in range(n_steps // STEPS_PER_EXPERIMENT):
        plot += annotate("text", x=k * STEPS_PER_EXPERIMENT - 0.3, y=df["rmse"].max() * 1.05,
                         label=f"exp. {k + 1}", ha="left", size=8, color="#555555")
    return plot


def rsa_frame(path: Path) -> pd.DataFrame:
    """One row per (cell, scoring step) for the best model and the best seed.

    The cells run the inner loop on one fixed dataset, so a model's fit, and
    its pool RMSE to the ground truth, is the same at every step. Step 0 is
    the seeds, and its best is the baseline; the last step (the end-of-loop
    prune, which never changes the best model) is not drawn.
    """
    rows = []
    for cell, record in sorted(json.loads(path.read_text())["cells"].items()):
        _, gt, rep = cell.split("_")
        trajectory = record["trajectory"][:-1]
        for step in trajectory:
            common = dict(gt=f"Ground truth: {RSA_GT_NAMES[gt]}", repeat=f"repeat {rep.removeprefix('rep')}",
                          step=step["step"])
            rows.append(dict(common, series="best model", rmse=step["rmse_to_gt"]))
            rows.append(dict(common, series="best other seed model", rmse=trajectory[0]["rmse_to_gt"]))
    return pd.DataFrame(rows)


def rsa_plot(df: pd.DataFrame) -> ggplot:
    best = df[df["series"] == "best model"]
    # The repeats share their data and step-0 best seed, so one line per panel (their mean).
    seed = (df[df["series"] == "best other seed model"]
            .groupby(["gt", "series", "step"], as_index=False, observed=True)["rmse"].mean())
    return (
        ggplot(df, aes("step", "rmse", color="series", linetype="series"))
        + geom_line(data=seed, size=0.8)
        + geom_line(aes(group="repeat"), data=best, size=0.8)
        + geom_point(aes(shape="repeat"), data=best, size=2)
        + facet_wrap("gt", nrow=1)
        + scale_color_manual(values={"best model": BEST_COLOR, "best other seed model": SEED_COLOR})
        + scale_linetype_manual(values={"best model": "solid", "best other seed model": "dotted"})
        + scale_x_continuous(breaks=sorted(df["step"].unique()))
        + scale_y_continuous(limits=(0, None))
        + labs(x="inner loop iteration", y="RMSE vs. ground truth", color="", linetype="", shape="")
        + theme_minimal()
        + theme(legend_position="bottom", panel_grid_minor=element_blank(),
                strip_text=element_text(size=10), figure_size=(7, 3.2))
    )


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in [
        ("fig_recovery_number_game", number_game_plot(number_game_frame(args.ng_root))),
        ("fig_recovery_rsa", rsa_plot(rsa_frame(args.rsa_trajectory))),
    ]:
        for ext in ("pdf", "png"):
            path = args.out_dir / f"{name}.{ext}"
            plot.save(path, dpi=200, verbose=False)
            print(f"wrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
