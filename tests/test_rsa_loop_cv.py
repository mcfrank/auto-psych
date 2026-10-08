"""Grouped cross-validation folds and comparison (no fitting)."""

import numpy as np
import pandas as pd
import pytest

from src.rsa.loop.cv import CVResult, assign_folds, compare_cv
from src.rsa.split import unit_keys


def frame():
    rows = []
    for src, n_units in (("a", 6), ("b", 4)):
        for u in range(n_units):
            for _ in range(5 + u):
                rows.append(dict(source=src, experiment="E1", condition=f"c{u}", participant_id=f"{src}{u}",
                                 objects="[[0,1],[1,1]]", query="utterance", utterance=1,
                                 dv="forced_choice", included=True))
    return pd.DataFrame(rows)


def test_folds_hold_whole_units_and_every_source():
    f = frame()
    fold = assign_folds(f, 3, seed=1)
    units = unit_keys(f)
    # Whole units: every trial of a unit is in one fold.
    assert (pd.Series(fold).groupby(units.to_numpy()).nunique() == 1).all()
    # Every fold has trials from both sources.
    for k in range(3):
        assert set(f.source[fold == k]) == {"a", "b"}
    assert (assign_folds(f, 3, seed=1) == fold).all()


def test_a_source_with_fewer_units_than_folds_raises():
    with pytest.raises(ValueError, match="fewer than the 5 folds"):
        assign_folds(frame(), 5)


def test_compare_cv_ranks_by_elpd_with_a_clustered_se():
    units = np.repeat(np.arange(4), 3)
    good = CVResult(pointwise=np.full(12, -0.5), converged=True)
    bad = CVResult(pointwise=np.array([-0.5] * 6 + [-1.5] * 6), converged=True)
    out = compare_cv({"good": good, "bad": bad}, units)
    assert out["good"]["cv_diff"] == 0 and out["bad"]["cv_diff"] == pytest.approx(6.0)
    assert out["bad"]["cv_dse"] > 0 and out["good"]["cv_dse"] == 0


def test_compare_cv_reports_each_source_behind_its_own_best():
    units = np.repeat(np.arange(4), 3)
    sources = np.array(["a"] * 6 + ["b"] * 6)
    # x leads on a, y leads on b.
    x = CVResult(pointwise=np.array([-0.2] * 6 + [-1.0] * 6), converged=True)
    y = CVResult(pointwise=np.array([-0.8] * 3 + [-0.6] * 3 + [-0.5] * 6), converged=True)
    out = compare_cv({"x": x, "y": y}, units, sources)
    assert out["x"]["cv_behind_by_source"] == {"a": 0.0, "b": pytest.approx(3.0)}
    assert out["y"]["cv_behind_by_source"] == {"a": pytest.approx(3.0), "b": 0.0}
    assert out["y"]["cv_dse_by_source"]["a"] > 0 and out["x"]["cv_dse_by_source"]["a"] == 0
