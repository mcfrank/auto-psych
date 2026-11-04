"""The model-recovery figures (``scripts/paper/recovery_figures.py``): the
combined one, every domain's recovery cells as the mean ± SEM over its runs,
and the number game's own, one line per repeat."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.paper import recovery_figures as rf


@pytest.fixture(scope="module")
def frame():
    args = rf.Args()
    return rf.combined_frame(args.sr_csv, args.ng_root, args.rsa_trajectory)


def _row(frame, panel, series, step):
    rows = frame[(frame["panel"] == panel) & (frame["series"] == series) & (frame["step"] == step)]
    assert len(rows) == 1, (panel, series, step, len(rows))
    return rows.iloc[0]


def test_one_panel_per_ground_truth_in_domain_order(frame):
    assert list(frame["panel"].cat.categories) == [
        "Subjective randomness\nencoding difficulty", "Subjective randomness\nfinite experience",
        "Subjective randomness\nlocal representativeness", "Subjective randomness\nregularity detection",
        "Number game\nTenenbaum (2000)", "Number game\nBigelow & Piantadosi, mixture",
        "Number game\nBigelow & Piantadosi, independent",
        "Reference games\nliteral listener", "Reference games\nsalience listener"]


def test_subjective_randomness_is_its_runs_summary(frame):
    row = _row(frame, "Subjective randomness\nencoding difficulty", "best model", 0)
    assert (row["mean"], row["err"], row["n"]) == pytest.approx((0.077171316582329, 0.02358623826221914, 4))


def test_the_number_game_is_the_mean_and_sem_of_the_november_sweeps_three_repeats(frame):
    root = rf.Args().ng_root
    final = pd.Series([pd.read_csv(root / f"run{r}" / "tenenbaum_rules_similarity" / "holdout.csv")["rmse"].iloc[-1]
                       for r in (1, 2, 3)])
    row = _row(frame, "Number game\nTenenbaum (2000)", "best model", 17)
    assert row["n"] == 3
    assert (row["mean"], row["err"]) == pytest.approx((final.mean(), final.std(ddof=1) / 3 ** 0.5))


def test_reference_games_average_both_repeats_at_every_refitted_step(frame):
    rsa = frame[frame["panel"].astype(str).str.startswith("Reference games")]
    assert (rsa["n"] == 2).all() and not rsa["mean"].isna().any()


def test_the_number_games_own_figure_has_every_repeat_of_every_ground_truth():
    df = rf.number_game_frame(rf.Args().ng_root)
    best = df[df["series"] == "best model"]
    assert best.groupby("gt", observed=True)["repeat"].nunique().to_dict() == {
        "Ground truth: Tenenbaum (2000)": 3, "Ground truth: Bigelow & Piantadosi, mixture": 3,
        "Ground truth: Bigelow & Piantadosi, independent": 3}
    assert not math.isnan(best["rmse"].iloc[0])
