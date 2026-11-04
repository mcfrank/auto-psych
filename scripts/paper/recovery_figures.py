"""Model-recovery figures: one combining every domain, and the number game's
and the reference games' own.

- All domains (``fig_recovery``): one row per domain, one panel per held-out
  ground truth, each domain's runs as the mean ± SEM of the best model's RMSE
  to the ground truth at every scoring step, against the best other seed's
  (dotted, its SEM shaded). Subjective randomness: the final recovery sweep's
  summary over its repeats (``data/results/final_recovery/holdout_combined.csv``;
  4-5 runs per ground truth). Number game and reference games: their three
  and two repeats below.

- Number game (``fig_recovery_number_game``): one panel per held-out ground
  truth of the November sweep (``data/results/number_game_recovery-nov26/``,
  the 2016 paper's seeds; three repeats, one line each). The RMSE to the ground
  truth of the loop's best model at every scoring step, against the ELPD-best
  other seed fitted to all of the cell's data (the subjective-randomness
  recovery figure's baseline; each repeat has its own data, so its own).
- Reference games (``fig_recovery_rsa``): the four recovery cells of Sherlock
  run 2 (inner loop only, on the published displays). Each step's best
  model's pool RMSE to the ground truth, against the starting model that was
  best at step 0, all refitted by ``rsa_recovery_trajectory.py``
  (``data/paper/rsa_recovery_trajectory.json``).
- Alien ground truths (``fig_recovery_alien``): subjective randomness's four
  implausible rules (``data/results/final_recovery/impossible/run1/``, one
  repeat; the loop starts from all four seeds). The RMSE to the ground truth
  of the loop's best model at every scoring step, against the ELPD-best seed
  among those with reliable PSIS-LOO fitted to all of the cell's data (the
  cell's recorded ``elpd_best_rmse``, as in the subjective-randomness summary).

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
    geom_blank,
    geom_linerange,
    geom_ribbon,
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
    "bigelow_piantadosi_mixture": "Ground truth: Bigelow & Piantadosi, mixture",
    "bigelow_piantadosi_independent": "Ground truth: Bigelow & Piantadosi, independent",
}
RSA_GT_NAMES = {"literal": "literal listener", "salience": "salience listener"}
SR_GT_NAMES = {  # as in the paper (paper/methods_changes.md)
    "falk_konold_dp": "encoding difficulty",
    "finite_experience_occurrence": "finite experience",
    "local_representativeness": "local representativeness",
    "motif_stack": "regularity detection",
}
ALIEN_GT_NAMES = {
    "fewer_heads_more_random": "fewer heads more random",
    "longer_runs_more_random": "longer runs more random",
    "more_heads_more_random": "more heads more random",
    "more_imbalance_more_random": "more imbalance more random",
}
COMBINED_NG_NAMES = {gt: name.removeprefix("Ground truth: ") for gt, name in NG_GT_NAMES.items()}
SERIES = {"best": "best model", "fitted_baseline": "best other seed model"}
STEPS_PER_EXPERIMENT = 6


@dataclass
class Args:
    sr_csv: Path = here() / "data/results/final_recovery/holdout_combined.csv"
    """The subjective-randomness final recovery's summary over its repeats."""
    ng_root: Path = here() / "data/results/number_game_recovery-nov26"
    """The number-game recovery sweep (``run<r>/<ground truth>/holdout.{json,csv}``)."""
    rsa_trajectory: Path = here() / "data/paper/rsa_recovery_trajectory.json"
    """Each recovery cell's step-best models' distances (``rsa_recovery_trajectory.py``)."""
    alien_root: Path = here() / "data/results/final_recovery/impossible"
    """The alien recovery cells (``run1/<ground truth>/holdout.{json,csv}``)."""
    out_dir: Path = here() / "data/paper/figures"


def number_game_frame(root: Path) -> pd.DataFrame:
    """One row per (cell, scoring step) for the best model, plus each cell's baseline."""
    rows = []
    for run in sorted(root.glob("run*")):
        repeat = f"repeat {run.name.removeprefix('run')}"
        for gt in NG_GT_NAMES:
            holdout = json.loads((run / gt / "holdout.json").read_text())["gt_runs"][0]
            per_model = holdout["fitted_baseline"]["per_model"]
            seed = max(per_model, key=lambda m: per_model[m]["elpd_loo"])
            trajectory = pd.read_csv(run / gt / "holdout.csv")
            for step, rmse in zip(trajectory["global_step"], trajectory["rmse"]):
                rows.append(dict(gt=NG_GT_NAMES[gt], repeat=repeat, step=step, rmse=rmse, series="best model"))
            for step in trajectory["global_step"]:
                rows.append(dict(gt=NG_GT_NAMES[gt], repeat=repeat, step=step, rmse=per_model[seed]["rmse"],
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
        + geom_line(aes(group="repeat"), data=seed, size=0.8)
        + geom_line(aes(group="repeat"), data=best, size=0.8)
        + geom_point(aes(shape="repeat"), data=best, size=1.6)
        + facet_wrap("gt", nrow=1)
        + scale_color_manual(values={"best model": BEST_COLOR, "best other seed model": SEED_COLOR})
        + scale_linetype_manual(values={"best model": "solid", "best other seed model": "dotted"})
        + scale_x_continuous(breaks=list(range(n_steps)))
        + scale_y_continuous(limits=(0, None))
        + labs(x="inner loop iteration", y="RMSE vs. ground truth", color="", linetype="", shape="")
        + theme_minimal()
        + theme(legend_position="bottom", panel_grid_minor=element_blank(),
                strip_text=element_text(size=10), figure_size=(12, 3.4))
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


def alien_frame(root: Path) -> pd.DataFrame:
    """One row per (alien cell, scoring step) for the best model and the best seed."""
    rows = []
    for gt, name in ALIEN_GT_NAMES.items():
        cell = root / "run1" / gt
        baseline = json.loads((cell / "holdout.json").read_text())["gt_runs"][0]["fitted_baseline"]
        if baseline["elpd_best_rmse"] is None:
            raise RuntimeError(f"{cell}: no ELPD-best seed ({baseline['elpd_best_reason']})")
        trajectory = pd.read_csv(cell / "holdout.csv")
        for step, rmse in zip(trajectory["global_step"], trajectory["rmse"]):
            rows.append(dict(gt=name, step=step, rmse=rmse, series="best model"))
            rows.append(dict(gt=name, step=step, rmse=baseline["elpd_best_rmse"], series="best seed model"))
    out = pd.DataFrame(rows)
    out["gt"] = pd.Categorical(out["gt"], categories=list(ALIEN_GT_NAMES.values()))
    return out


def alien_plot(df: pd.DataFrame) -> ggplot:
    best = df[df["series"] == "best model"]
    seed = df[df["series"] == "best seed model"]
    return (
        ggplot(df, aes("step", "rmse", color="series", linetype="series"))
        + geom_vline(xintercept=[STEPS_PER_EXPERIMENT - 0.5, 2 * STEPS_PER_EXPERIMENT - 0.5],
                     linetype="dotted", color="grey", size=0.6)
        + geom_line(data=seed, size=0.8)
        + geom_line(data=best, size=0.8)
        + geom_point(data=best, size=1.4)
        + facet_wrap("gt", nrow=1)
        + scale_color_manual(values={"best model": BEST_COLOR, "best seed model": SEED_COLOR},
                             breaks=["best model", "best seed model"])
        + scale_linetype_manual(values={"best model": "solid", "best seed model": "dotted"},
                                breaks=["best model", "best seed model"])
        + scale_y_continuous(limits=(0, None))
        + labs(x="inner loop iteration", y="RMSE vs. ground truth", color="", linetype="")
        + theme_minimal()
        + theme(legend_position="bottom", panel_grid_minor=element_blank(),
                strip_text=element_text(size=9), figure_size=(10, 3))
    )


DOMAINS = ("Subjective randomness", "Number game", "Reference games")


def _panel(domain: str, gt: str) -> str:
    return f"{domain}\n{gt}"


def _over_repeats(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """The mean and SEM over a per-repeat frame's repeats at each (ground truth, series, step)."""
    out = (df.groupby(["gt", "series", "step"], observed=True)["rmse"]
           .agg(mean="mean", err=lambda r: r.std(ddof=1) / len(r) ** 0.5, n="size").reset_index())
    out["panel"] = [_panel(domain, str(g).removeprefix("Ground truth: ")) for g in out["gt"]]
    return out.drop(columns="gt")


def combined_frame(sr_csv: Path, ng_root: Path, rsa_trajectory: Path) -> pd.DataFrame:
    """One row per (panel, series, step): ``mean`` and ``err`` (SEM) of the
    RMSE to the ground truth over the domain's ``n`` runs."""
    sr = pd.read_csv(sr_csv)
    sr = sr[(sr["metric"] == "rmse") & sr["series"].isin(list(SERIES))]
    sr = pd.DataFrame(dict(panel=[_panel(DOMAINS[0], SR_GT_NAMES[g]) for g in sr["gt_model"]],
                           series=sr["series"].map(SERIES).values, step=sr["position"].values,
                           mean=sr["mean"].values, err=sr["err"].values, n=sr["n"].values))
    out = pd.concat([sr, _over_repeats(number_game_frame(ng_root), DOMAINS[1]),
                     _over_repeats(rsa_frame(rsa_trajectory), DOMAINS[2])], ignore_index=True)
    order = ([_panel(DOMAINS[0], v) for v in SR_GT_NAMES.values()]
             + [_panel(DOMAINS[1], v) for v in COMBINED_NG_NAMES.values()]
             + [_panel(DOMAINS[2], v) for v in RSA_GT_NAMES.values()])
    out["panel"] = pd.Categorical(out["panel"], categories=order)
    return out


def _domain_row(df: pd.DataFrame, domain: str, last: bool) -> ggplot:
    """One domain's panels, titled by the domain, each by its ground truth."""
    df = df[df["panel"].astype(str).str.startswith(domain + "\n")].copy()
    df["gt"] = pd.Categorical(df["panel"].astype(str).str.split("\n").str[1],
                              categories=[str(p).split("\n")[1] for p in df["panel"].cat.categories
                                          if str(p).startswith(domain + "\n")])
    best = df[df["series"] == "best model"]
    seed = df[df["series"] == "best other seed model"]
    zero = pd.DataFrame(dict(gt=pd.Categorical(df["gt"].cat.categories, categories=df["gt"].cat.categories),
                             step=0, mean=0.0))
    plot = (
        ggplot(df, aes("step", "mean", color="series", linetype="series"))
        + geom_blank(aes("step", "mean"), data=zero, inherit_aes=False)
        + geom_ribbon(aes(ymin="mean - err", ymax="mean + err"), data=seed.dropna(subset=["err"]),
                      fill=SEED_COLOR, alpha=0.15, color="none", show_legend=False)
        + geom_line(data=seed, size=0.8)
        + geom_linerange(aes(ymin="mean - err", ymax="mean + err"), data=best.dropna(subset=["err"]), size=0.5,
                         show_legend=False)
        + geom_line(data=best, size=0.8)
        + geom_point(data=best, size=1.4)
        + facet_wrap("gt", nrow=1, scales="free")
        + scale_color_manual(values={"best model": BEST_COLOR, "best other seed model": SEED_COLOR},
                             breaks=list(SERIES.values()))
        + scale_linetype_manual(values={"best model": "solid", "best other seed model": "dotted"},
                                breaks=list(SERIES.values()))
        + labs(title=domain, x="inner loop iteration" if last else "", y="RMSE vs. ground truth",
               color="", linetype="")
        + theme_minimal()
        + theme(legend_position="bottom" if last else "none", panel_grid_minor=element_blank(),
                strip_text=element_text(size=9), plot_title=element_text(size=11, ha="left"))
    )
    if domain != "Reference games":  # three experiments of six scoring steps
        plot += geom_vline(xintercept=[STEPS_PER_EXPERIMENT - 0.5, 2 * STEPS_PER_EXPERIMENT - 0.5],
                           linetype="dotted", color="grey", size=0.6)
    return plot


def combined_plot(df: pd.DataFrame):
    rows = [_domain_row(df, domain, last=domain == DOMAINS[-1]) for domain in DOMAINS]
    return (rows[0] / rows[1] / rows[2]) + theme(figure_size=(10, 8.5))


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in [
        ("fig_recovery", combined_plot(combined_frame(args.sr_csv, args.ng_root, args.rsa_trajectory))),
        ("fig_recovery_number_game", number_game_plot(number_game_frame(args.ng_root))),
        ("fig_recovery_rsa", rsa_plot(rsa_frame(args.rsa_trajectory))),
        ("fig_recovery_alien", alien_plot(alien_frame(args.alien_root))),
    ]:
        for ext in ("pdf", "png"):
            path = args.out_dir / f"{name}.{ext}"
            plot.save(path, dpi=200, verbose=False)
            print(f"wrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
