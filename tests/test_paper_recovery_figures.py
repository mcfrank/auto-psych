"""The model-recovery figures (``scripts/paper/recovery_figures.py``): the
combined one, every domain's recovery cells as the mean ± SEM over its runs,
and the number game's own, one line per repeat."""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from scripts.paper import recovery_figures as rf
from scripts.paper.figures import fig3, style


@pytest.fixture(scope="module")
def frame():
    args = rf.Args()
    return rf.combined_frame(args.sr_root, args.ng_root, args.rsa_trajectory)


def _row(frame, panel, series, step):
    rows = frame[(frame["panel"] == panel) & (frame["series"] == series) & (frame["step"] == step)]
    assert len(rows) == 1, (panel, series, step, len(rows))
    return rows.iloc[0]


def test_one_panel_per_ground_truth_in_domain_order(frame):
    assert list(frame["panel"].cat.categories) == [
        "Subjective randomness\nencoding difficulty", "Subjective randomness\nfinite experience",
        "Subjective randomness\nlocal representativeness", "Subjective randomness\nregularity detection",
        "Number game\nTenenbaum (1999)", "Number game\nBigelow & Piantadosi, independent",
        "Reference games\nliteral listener", "Reference games\nsalience listener"]


def test_subjective_randomness_pools_every_cell_of_the_final_recovery(frame):
    sr = frame[frame["panel"].astype(str).str.startswith("Subjective randomness")]
    assert len(sr) == 4 * 2 * 18 and (sr["n"] == 5).all()


def test_subjective_randomness_best_model_is_the_mean_and_sem_over_its_five_runs(frame):
    root = rf.Args().sr_root
    seed = pd.Series([pd.read_csv(root / f"run{r}" / "falk_konold_dp" / "holdout.csv")["rmse"].iloc[0]
                      for r in range(1, 6)])
    row = _row(frame, "Subjective randomness\nencoding difficulty", "best model", 0)
    assert (row["mean"], row["err"]) == pytest.approx((seed.mean(), seed.std(ddof=1) / 5 ** 0.5))


def test_subjective_randomness_baseline_is_the_elpd_best_seed_fitted_to_all_data(frame):
    root = rf.Args().sr_root
    values = pd.Series([json.loads((root / f"run{r}" / "motif_stack" / "holdout.json").read_text())
                        ["gt_runs"][0]["fitted_baseline"]["elpd_best_rmse"] for r in range(1, 6)])
    seed = frame[(frame["panel"] == "Subjective randomness\nregularity detection")
                 & (frame["series"] == "best other seed model")]
    assert len(seed) == 18
    assert seed["mean"].to_list() == pytest.approx([values.mean()] * 18)
    assert seed["err"].to_list() == pytest.approx([values.std(ddof=1) / 5 ** 0.5] * 18)


def test_subjective_randomness_refuses_a_sweep_with_an_unfinished_cell(tmp_path):
    for r in range(1, 6):
        for gt in rf.SR_GT_NAMES:
            (tmp_path / f"run{r}" / gt).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="run1/falk_konold_dp"):
        rf.subjective_randomness_frame(tmp_path)


def test_the_number_game_is_the_mean_and_sem_of_the_november_sweeps_three_repeats(frame):
    root = rf.Args().ng_root
    final = pd.Series([pd.read_csv(root / f"run{r}" / "tenenbaum_rules_similarity" / "holdout.csv")["rmse"].iloc[-1]
                       for r in (1, 2, 3)])
    row = _row(frame, "Number game\nTenenbaum (1999)", "best model", 17)
    assert row["n"] == 3
    assert (row["mean"], row["err"]) == pytest.approx((final.mean(), final.std(ddof=1) / 3 ** 0.5))


def test_reference_games_average_both_repeats_at_every_refitted_step(frame):
    rsa = frame[frame["panel"].astype(str).str.startswith("Reference games")]
    assert (rsa["n"] == 2).all() and not rsa["mean"].isna().any()


def test_the_number_games_own_figure_has_every_repeat_of_every_ground_truth():
    df = rf.number_game_frame(rf.Args().ng_root)
    best = df[df["series"] == "best model"]
    assert best.groupby("gt", observed=True)["repeat"].nunique().to_dict() == {
        "Ground truth: Tenenbaum (1999)": 3, "Ground truth: Bigelow & Piantadosi, independent": 3}
    assert not math.isnan(best["rmse"].iloc[0])


def test_the_alien_figure_is_one_panel_per_alien_rule_over_every_scoring_step():
    df = rf.alien_frame(rf.Args().alien_root)
    assert list(df["gt"].cat.categories) == [
        "fewer heads more random", "longer runs more random",
        "more heads more random", "more imbalance more random"]
    best = df[df["series"] == "best model"]
    assert best.groupby("gt", observed=True)["step"].apply(list).to_dict() == {
        gt: list(range(18)) for gt in df["gt"].cat.categories}
    cell = rf.Args().alien_root / "run1" / "fewer_heads_more_random" / "holdout.csv"
    final = best[best["gt"] == "fewer heads more random"]["rmse"].iloc[-1]
    assert final == pytest.approx(pd.read_csv(cell)["rmse"].iloc[-1])


def test_the_alien_baseline_is_the_recorded_elpd_best_seed_among_trusted_fits():
    df = rf.alien_frame(rf.Args().alien_root)
    seed = df[(df["series"] == "best seed model") & (df["gt"] == "longer runs more random")]
    # motif_stack has the higher ELPD there, but its PSIS-LOO is not trusted.
    assert len(seed) == 18 and seed["rmse"].nunique() == 1
    assert seed["rmse"].iloc[0] == pytest.approx(0.2118, abs=1e-4)


def test_the_alien_figure_draws(styled):
    rf.alien_plot(rf.alien_frame(rf.Args().alien_root))


@pytest.fixture(scope="module")
def runs():
    args = rf.Args()
    return rf.combined_runs_frame(args.sr_root, args.ng_root, args.rsa_trajectory)


def test_the_combined_figure_has_every_run_of_every_panel(runs):
    n_runs = runs.groupby("panel", observed=True)["run"].nunique()
    assert n_runs.to_dict() == {
        "Subjective randomness\nencoding difficulty": 5, "Subjective randomness\nfinite experience": 5,
        "Subjective randomness\nlocal representativeness": 5, "Subjective randomness\nregularity detection": 5,
        "Number game\nTenenbaum (1999)": 3,
        "Number game\nBigelow & Piantadosi, independent": 3,
        "Reference games\nliteral listener": 2, "Reference games\nsalience listener": 2}


def test_the_runs_average_to_the_plotted_best_model_mean(frame, runs):
    means = runs.groupby(["panel", "step"], observed=True)["rmse"].mean().rename("runs_mean").reset_index()
    best = frame[frame["series"] == "best model"].merge(means, on=["panel", "step"], how="outer", indicator=True)
    assert (best["_merge"] == "both").all()
    assert best["runs_mean"].to_list() == pytest.approx(best["mean"].to_list())


def test_the_combined_figure_draws(frame, runs, styled):
    rf.combined_plot(frame, runs)


# --- the main figures' house style (scripts/paper/figures/style.py), the data as before -------------------


@pytest.fixture
def styled():
    with plt.rc_context():
        style.setup_style()
        yield
    plt.close("all")


@pytest.fixture
def figures(frame, runs, styled):
    args = rf.Args()
    ng, rsa, alien = rf.number_game_frame(args.ng_root), rf.rsa_frame(args.rsa_trajectory), rf.alien_frame(args.alien_root)
    return {"combined": (rf.combined_plot(frame, runs), frame),
            "number_game": (rf.number_game_plot(ng), ng),
            "rsa": (rf.rsa_plot(rsa), rsa),
            "alien": (rf.alien_plot(alien), alien)}


def _data_axes(fig):
    """The panels (not the domains' icons, which have no frame)."""
    return [ax for ax in fig.axes if ax.axison]


def test_the_recovery_figures_are_matplotlib_in_the_paper_style(figures):
    for name, (fig, _) in figures.items():
        assert isinstance(fig, Figure), name
        assert fig.get_figwidth() == fig3.WIDTH_IN, name  # PNAS's full width, as Figures 2-4
        assert fig._suptitle is None, name  # no figure titles: the caption describes
        for ax in _data_axes(fig):
            assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible(), name
            assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()), name
            assert ax.title.get_fontfamily()[0] == style.TITLE_FONT["family"], name


def test_the_recovery_figures_use_the_main_figures_palette(figures):
    """Blue for the loop's best (discovered) model, seed grey for the best seed, as in Figures 2-4."""
    for name, (fig, _) in figures.items():
        colours = {gid: set() for gid in ("best", "run", "seed")}
        for ax in _data_axes(fig):
            for line in ax.get_lines():
                if line.get_gid() in colours:
                    colours[line.get_gid()].add(to_hex(line.get_color()))
        assert colours["best"] == {to_hex(fig3.BEST)}, name
        assert colours["seed"] == {to_hex(fig3.SEED)}, name
        assert colours["run"] <= {to_hex(fig3.BEST)}, name


def _plotted(fig, gid):
    """Each panel's (title, on one line; x; y) of its lines with this gid."""
    return [(ax.get_title().replace("\n", " "), list(line.get_xdata()), list(line.get_ydata()))
            for ax in _data_axes(fig) for line in ax.get_lines() if line.get_gid() == gid]


def test_the_combined_figure_plots_the_frames_values(figures, runs):
    fig, frame = figures["combined"]
    for gid, series in (("best", "best model"), ("seed", "best other seed model")):
        lines = _plotted(fig, gid)
        assert len(lines) == 8
        for title, x, y in lines:
            want = frame[(frame["series"] == series) & (frame["panel"].astype(str).str.endswith("\n" + title))]
            assert x == want["step"].to_list() and y == pytest.approx(want["mean"].to_list()), (gid, title)
    assert len(_plotted(fig, "run")) == len(runs.groupby(["panel", "run"], observed=True))
    for title, x, y in _plotted(fig, "run"):
        mine = runs[runs["panel"].astype(str).str.endswith("\n" + title)]
        assert any(x == r["step"].to_list() and y == pytest.approx(r["rmse"].to_list())
                   for _, r in mine.groupby("run")), title


def test_the_per_repeat_figures_plot_the_frames_values(figures):
    for name in ("number_game", "rsa", "alien"):
        fig, df = figures[name]
        best = df[df["series"] == "best model"]
        groups = best.groupby(["gt", "repeat"] if "repeat" in df else ["gt"], observed=True)
        lines = _plotted(fig, "best")
        assert len(lines) == len(groups), name
        for title, x, y in lines:
            assert any(str(key[0]) == title and x == g["step"].to_list() and y == pytest.approx(g["rmse"].to_list())
                       for key, g in groups), (name, title)
        seed = df[df["series"] != "best model"]
        for title, x, y in _plotted(fig, "seed"):
            values = set(seed[seed["gt"].astype(str) == title]["rmse"].round(12))
            if name == "rsa":  # one line per panel, the repeats' mean (they share the step-0 seed)
                values = {round(seed[seed["gt"] == title]["rmse"].mean(), 12)}
            assert set(pd.Series(y).round(12)) <= values and len(set(y)) == 1, (name, title)


def test_no_plotted_value_is_cut_off_by_its_panels_y_range(figures):
    """Shared y axes are limited once every panel is drawn, so no panel's data runs off the top."""
    for name, (fig, _) in figures.items():
        for ax in _data_axes(fig):
            lo, hi = ax.get_ylim()
            assert lo == 0, name
            for line in ax.get_lines():
                if line.get_gid() in ("best", "seed", "run"):
                    assert max(line.get_ydata()) <= hi, (name, ax.get_title())
