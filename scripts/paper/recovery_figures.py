"""Model-recovery figures: one combining every domain, and the number game's
and the reference games' own.

- All domains (``fig_recovery``): one row per domain, one panel per held-out
  ground truth, each domain's runs as the mean ± SEM of the best model's RMSE
  to the ground truth at every scoring step (each run a thin translucent line
  behind it), against the best other seed's (grey, dashed, its SEM shaded). Subjective randomness: the final recovery sweep's
  five repeats (``data/results/final_recovery/run<r>/<ground truth>/``), every
  cell required; the best model at each position aligned by experiment
  (``aggregate_holdout_trajectories``), against the ELPD-best other seed fitted
  to all of the cell's data (``fitted_baseline.elpd_best_rmse``), as in the
  number game. Number game and reference games: their three and two repeats
  below.

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

Drawn in Figures 2-4's house style (matplotlib, ``scripts/paper/figures/style.py``): blue for the loop's
best model, seed grey for the best seed, Avenir, open axes, PNAS's full width.

Usage:
    uv run python -m scripts.paper.recovery_figures
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pyprojroot import here

from scripts.paper.figures.fig3 import BEST, SEED, WIDTH_IN
from scripts.paper.figures.icons import STIMULI
from scripts.paper.figures.style import DPI, INK, INK_2, TITLE_FONT, setup_style
from src.subjective_randomness.reporting import aggregate_holdout_trajectories
from src.subjective_randomness.sweep_cells import survey_sweep

# Figures 2-4's house style: blue (``BEST``) for the loop's best model, seed grey (``SEED``) for the best seed,
# dashed as Figure 2's best-seed line; the experiments' boundaries in a neutral lighter than the seed grey
SEED_LS, BOUNDARY = (0, (3, 2)), "#dcdcdc"
MARKERS = {"repeat 1": "o", "repeat 2": "^", "repeat 3": "s"}  # a repeat's points, where repeats are drawn apart
XLABEL, YLABEL = "inner loop iteration", "RMSE vs. ground truth"

SEED_COLOR = "#D95F02"
NG_GT_NAMES = {
    "tenenbaum_rules_similarity": "Ground truth: Tenenbaum (1999)",
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
SR_REPEATS = 5


@dataclass
class Args:
    sr_root: Path = here() / "data/results/final_recovery"
    """The subjective-randomness final recovery (``run<r>/<ground truth>/holdout.json``)."""
    ng_root: Path = here() / "data/results/number_game_recovery-nov26"
    """The number-game recovery sweep (``run<r>/<ground truth>/holdout.{json,csv}``)."""
    rsa_trajectory: Path = here() / "data/paper/rsa_recovery_trajectory.json"
    """Each recovery cell's step-best models' distances (``rsa_recovery_trajectory.py``)."""
    alien_root: Path = here() / "data/results/final_recovery/impossible"
    """The alien recovery cells (``run1/<ground truth>/holdout.{json,csv}``)."""
    out_dir: Path = here() / "data/paper/figures"


def subjective_randomness_frame(root: Path) -> pd.DataFrame:
    """One row per (ground truth, series, position): ``mean``, ``err`` (SEM)
    and ``n`` of the RMSE to the ground truth over every repeat's cell."""
    survey = survey_sweep(root, n_repeats=SR_REPEATS, gt_models=list(SR_GT_NAMES))
    unfinished = {**survey.partial, **survey.missing}
    if unfinished:
        raise RuntimeError(f"{root}: unfinished recovery cells: {unfinished}")
    labels = sorted(survey.complete)
    results = [json.loads((survey.complete[label] / "holdout.json").read_text()) for label in labels]
    aggregated = aggregate_holdout_trajectories(results, metric="rmse", error="sem", labels=labels)
    baseline = pd.Series([result["gt_runs"][0]["fitted_baseline"]["elpd_best_rmse"] for result in results])
    gts = pd.Series([label.split("/")[1] for label in labels])
    rows = []
    for panel in aggregated["gt_models"]:
        if panel["excluded"]:
            raise RuntimeError(f"{panel['gt_model']}: cells left out of points: {panel['excluded']}")
        seed = baseline[gts == panel["gt_model"]]
        for point in panel["best"]:
            common = dict(gt=panel["gt_model"], step=point["x"])
            rows.append(dict(common, series="best model", mean=point["mean"], err=point["err"], n=point["n"]))
            rows.append(dict(common, series="best other seed model", mean=seed.mean(),
                             err=seed.std(ddof=1) / len(seed) ** 0.5, n=len(seed)))
    return pd.DataFrame(rows)


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


def _style_panel(ax, title: str) -> None:
    """A panel as in Figures 2-4 (open axes, no grid: ``setup_style``), its title in the panel-title size."""
    ax.set_title(title, fontsize=7, color=INK, pad=3)


def _boundaries(ax, n_steps: int) -> None:
    """Dotted lines between the experiments (six scoring steps each)."""
    for k in range(1, n_steps // STEPS_PER_EXPERIMENT):
        ax.axvline(k * STEPS_PER_EXPERIMENT - 0.5, color=BOUNDARY, lw=0.7, ls=(0, (1, 1.5)), zorder=0)


def _key(fig, handles, ncol: int) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.2, columnspacing=1.6)


def _series_key(seed_label: str, marker: str = "o") -> list:
    return [Line2D([], [], color=BEST, lw=1.2, marker=marker, ms=2.5, label="best model"),
            Line2D([], [], color=SEED, lw=1.2, ls=SEED_LS, label=seed_label)]


def _repeat_key(repeats) -> list:
    return [Line2D([], [], color=INK_2, lw=0, marker=MARKERS[r], ms=3, label=r) for r in repeats]


def _save(fig: Figure, out_dir: Path, name: str) -> None:
    """Uncropped, as Figure 4: the canvas is the saved size, PNAS's full width."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI)
        print(f"wrote {path}")
    plt.close(fig)


def number_game_plot(df: pd.DataFrame) -> Figure:
    n_steps = int(df["step"].max()) + 1
    gts = list(df["gt"].cat.categories)
    fig, axes = plt.subplots(1, len(gts), figsize=(WIDTH_IN, 2.3), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.3, wspace=0.08)
    top = df["rmse"].max() * 1.05
    for ax, gt in zip(axes, gts):
        _boundaries(ax, n_steps)
        cell = df[df["gt"] == gt]
        for repeat, d in cell.groupby("repeat"):
            seed = d[d["series"] == "best other seed model"]
            ax.plot(seed["step"], seed["rmse"], color=SEED, lw=1.0, ls=SEED_LS, gid="seed", zorder=1)
        for repeat, d in cell.groupby("repeat"):
            best = d[d["series"] == "best model"]
            ax.plot(best["step"], best["rmse"], color=BEST, lw=1.0, marker=MARKERS[repeat], ms=2.5,
                    gid="best", zorder=2)
        for k in range(n_steps // STEPS_PER_EXPERIMENT):
            ax.text(k * STEPS_PER_EXPERIMENT - 0.3, top, f"exp. {k + 1}", ha="left", va="bottom",
                    fontsize=5.5, color=INK_2)
        ax.set_xticks(range(0, n_steps, 3))
        ax.set_xticks(range(n_steps), minor=True)
        ax.set_xlabel(XLABEL)
        _style_panel(ax, gt.replace(": ", ":\n"))  # on one line, the longest is wider than its panel
    axes[0].set_ylim(0, top * 1.1)  # shared: room for the experiments' labels over every panel's data
    axes[0].set_ylabel(YLABEL)
    _key(fig, _series_key("best other seed model") + _repeat_key(sorted(df["repeat"].unique())), ncol=5)
    return fig


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


def rsa_plot(df: pd.DataFrame) -> Figure:
    gts = sorted(df["gt"].unique())
    fig, axes = plt.subplots(1, len(gts), figsize=(WIDTH_IN, 2.3), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.3, wspace=0.08)
    for ax, gt in zip(axes, gts):
        cell = df[df["gt"] == gt]
        # The repeats share their data and step-0 best seed, so one line per panel (their mean).
        seed = cell[cell["series"] == "best other seed model"].groupby("step", as_index=False)["rmse"].mean()
        ax.plot(seed["step"], seed["rmse"], color=SEED, lw=1.0, ls=SEED_LS, gid="seed", zorder=1)
        for repeat, d in cell[cell["series"] == "best model"].groupby("repeat"):
            ax.plot(d["step"], d["rmse"], color=BEST, lw=1.0, marker=MARKERS[repeat], ms=3, gid="best", zorder=2)
        ax.set_xticks(sorted(df["step"].unique()))
        ax.set_xlabel(XLABEL)
        _style_panel(ax, gt)
    axes[0].set_ylabel(YLABEL)
    axes[0].set_ylim(bottom=0)  # shared: set once every panel's data is in
    _key(fig, _series_key("best other seed model") + _repeat_key(sorted(df["repeat"].unique())), ncol=4)
    return fig


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


def alien_plot(df: pd.DataFrame) -> Figure:
    gts = list(df["gt"].cat.categories)
    fig, axes = plt.subplots(1, len(gts), figsize=(WIDTH_IN, 2.1), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.31, wspace=0.08)
    for ax, gt in zip(axes, gts):
        cell = df[df["gt"] == gt]
        _boundaries(ax, int(df["step"].max()) + 1)
        seed, best = cell[cell["series"] == "best seed model"], cell[cell["series"] == "best model"]
        ax.plot(seed["step"], seed["rmse"], color=SEED, lw=1.0, ls=SEED_LS, gid="seed", zorder=1)
        ax.plot(best["step"], best["rmse"], color=BEST, lw=1.0, marker="o", ms=2.2, gid="best", zorder=2)
        ax.set_xlabel(XLABEL)
        _style_panel(ax, gt)
    axes[0].set_ylabel(YLABEL)
    axes[0].set_ylim(bottom=0)  # shared: set once every panel's data is in
    _key(fig, _series_key("best seed model"), ncol=2)
    return fig


DOMAINS = ("Subjective randomness", "Number sequence generalization", "Reference games")


def _panel(domain: str, gt: str) -> str:
    return f"{domain}\n{gt}"


def _over_repeats(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """The mean and SEM over a per-repeat frame's repeats at each (ground truth, series, step)."""
    out = (df.groupby(["gt", "series", "step"], observed=True)["rmse"]
           .agg(mean="mean", err=lambda r: r.std(ddof=1) / len(r) ** 0.5, n="size").reset_index())
    out["panel"] = [_panel(domain, str(g).removeprefix("Ground truth: ")) for g in out["gt"]]
    return out.drop(columns="gt")


def _panel_order() -> list:
    return ([_panel(DOMAINS[0], v) for v in SR_GT_NAMES.values()]
            + [_panel(DOMAINS[1], v) for v in COMBINED_NG_NAMES.values()]
            + [_panel(DOMAINS[2], v) for v in RSA_GT_NAMES.values()])


def combined_runs_frame(sr_root: Path, ng_root: Path, rsa_trajectory: Path) -> pd.DataFrame:
    """One row per (panel, run, step): that run's best model's RMSE to the ground truth."""
    sr = [dict(panel=_panel(DOMAINS[0], name), run=run.name, step=step, rmse=rmse)
          for run in sorted(sr_root.glob("run*")) for gt, name in SR_GT_NAMES.items()
          for step, rmse in pd.read_csv(run / gt / "holdout.csv")[["global_step", "rmse"]].itertuples(index=False)]
    parts = [pd.DataFrame(sr)]
    for domain, df in [(DOMAINS[1], number_game_frame(ng_root)), (DOMAINS[2], rsa_frame(rsa_trajectory))]:
        best = df[df["series"] == "best model"]
        parts.append(pd.DataFrame(dict(
            panel=[_panel(domain, str(g).removeprefix("Ground truth: ")) for g in best["gt"]],
            run=best["repeat"].values, step=best["step"].values, rmse=best["rmse"].values)))
    out = pd.concat(parts, ignore_index=True)
    out["panel"] = pd.Categorical(out["panel"], categories=_panel_order())
    return out


def combined_frame(sr_root: Path, ng_root: Path, rsa_trajectory: Path) -> pd.DataFrame:
    """One row per (panel, series, step): ``mean`` and ``err`` (SEM) of the
    RMSE to the ground truth over the domain's ``n`` runs."""
    sr = subjective_randomness_frame(sr_root)
    sr = sr.assign(panel=[_panel(DOMAINS[0], SR_GT_NAMES[g]) for g in sr["gt"]]).drop(columns="gt")
    out = pd.concat([sr, _over_repeats(number_game_frame(ng_root), DOMAINS[1]),
                     _over_repeats(rsa_frame(rsa_trajectory), DOMAINS[2])], ignore_index=True)
    out["panel"] = pd.Categorical(out["panel"], categories=_panel_order())
    return out


def _in_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """A combined frame's rows of one domain, with its ground truth as ``gt``."""
    df = df[df["panel"].astype(str).str.startswith(domain + "\n")].copy()
    df["gt"] = pd.Categorical(df["panel"].astype(str).str.split("\n").str[1],
                              categories=[str(p).split("\n")[1] for p in df["panel"].cat.categories
                                          if str(p).startswith(domain + "\n")])
    return df


DOMAIN_ICONS = dict(zip(DOMAINS, ("subjective_randomness", "number_game", "rsa")))  # icons.STIMULI's keys
GRID_COLUMNS = 12  # each row's panels span an equal share of these (4, 3 and 2 panels)


def _domain_row(axes, df: pd.DataFrame, runs: pd.DataFrame, domain: str) -> None:
    """One domain's panels, each titled by its ground truth: every run thin behind the best model's mean
    ± SEM, against the best other seed's mean (its SEM shaded)."""
    df, runs = _in_domain(df, domain), _in_domain(runs, domain)
    for ax, gt in zip(axes, df["gt"].cat.categories):
        cell, mine = df[df["gt"] == gt], runs[runs["gt"] == gt]
        if domain != "Reference games":  # three experiments of six scoring steps
            _boundaries(ax, int(cell["step"].max()) + 1)
        for _, run in mine.groupby("run"):
            ax.plot(run["step"], run["rmse"], color=BEST, lw=0.5, alpha=0.35, gid="run", zorder=1)
        seed, best = cell[cell["series"] == "best other seed model"], cell[cell["series"] == "best model"]
        ax.fill_between(seed["step"], seed["mean"] - seed["err"], seed["mean"] + seed["err"], color=SEED,
                        alpha=0.3, lw=0, zorder=1)
        ax.plot(seed["step"], seed["mean"], color=SEED, lw=1.0, ls=SEED_LS, gid="seed", zorder=2)
        ax.vlines(best["step"], best["mean"] - best["err"], best["mean"] + best["err"], color=BEST, lw=0.6,
                  zorder=3)
        ax.plot(best["step"], best["mean"], color=BEST, lw=1.2, marker="o", ms=2.2, gid="best", zorder=3)
        if domain == "Reference games":
            ax.set_xticks(sorted(cell["step"].unique()))
        _style_panel(ax, str(gt))
        ax.set_ylim(bottom=0)  # each panel its own scale
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))  # short labels, clear of the panel to the left
    axes[0].set_ylabel(YLABEL)


def _domain_title(fig: Figure, ax, domain: str) -> None:
    """The domain's stimulus icon at the row's left edge and its name beside it, above the panel titles."""
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    y = to_fig.transform((0, ax.title.get_window_extent().y1))[1] + 0.022
    x0 = ax.get_position().x0
    t = fig.text(x0, y, domain, fontdict=TITLE_FONT, va="center", ha="left")
    box = t.get_window_extent()
    h = 1.9 * (to_fig.transform((0, box.y1))[1] - to_fig.transform((0, box.y0))[1])
    draw, aspect = STIMULI[DOMAIN_ICONS[domain]]
    w = h * aspect * fig.get_figheight() / fig.get_figwidth()
    draw(fig.add_axes((x0, y - h / 2, w, h)))
    t.set_x(x0 + w + 0.01)


def combined_plot(df: pd.DataFrame, runs: pd.DataFrame) -> Figure:
    fig = plt.figure(figsize=(WIDTH_IN, 6.4))
    grid = fig.add_gridspec(len(DOMAINS), GRID_COLUMNS, left=0.075, right=0.99, top=0.93, bottom=0.11,
                            hspace=0.72, wspace=1.5)
    rows = []
    for k, domain in enumerate(DOMAINS):
        n = _in_domain(df, domain)["gt"].cat.categories.size
        span = GRID_COLUMNS // n
        axes = [fig.add_subplot(grid[k, j * span:(j + 1) * span]) for j in range(n)]
        _domain_row(axes, df, runs, domain)
        rows.append(axes)
    for ax in rows[-1]:
        ax.set_xlabel(XLABEL)
    for axes, domain in zip(rows, DOMAINS):
        _domain_title(fig, axes[0], domain)
    _key(fig, _series_key("best other seed model"), ncol=2)
    return fig


def main(args: Args) -> None:
    with plt.rc_context():  # Figures 2-4's fonts and axes; the PDF's fonts embedded on saving
        setup_style()
        for name, draw in [
            ("fig_recovery", lambda: combined_plot(combined_frame(args.sr_root, args.ng_root, args.rsa_trajectory),
                                                   combined_runs_frame(args.sr_root, args.ng_root,
                                                                       args.rsa_trajectory))),
            ("fig_recovery_number_game", lambda: number_game_plot(number_game_frame(args.ng_root))),
            ("fig_recovery_rsa", lambda: rsa_plot(rsa_frame(args.rsa_trajectory))),
            ("fig_recovery_alien", lambda: alien_plot(alien_frame(args.alien_root))),
        ]:
            _save(draw(), args.out_dir, name)


if __name__ == "__main__":
    main(tyro.cli(Args))
