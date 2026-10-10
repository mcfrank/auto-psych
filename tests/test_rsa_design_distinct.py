"""One model per hypothesis the live displays can tell apart."""

import numpy as np

from src.rsa.design.distinct import SAME_ON_POOL_RMSE, keep_distinct


def test_twins_are_merged_into_the_better_model_and_the_cap_counts_distinct_ones():
    base = np.full((10, 4), 0.25)
    preds = {"best": base, "twin": base + 0.0005, "other": base + 0.05, "third": base - 0.05, "fourth": base + 0.1}
    out = keep_distinct(["best", "twin", "other", "third", "fourth"], preds, cap=3)
    assert out["kept"] == ["best", "other", "third"]
    assert [m["name"] for m in out["merged"]] == ["twin"] and out["merged"][0]["same_as"] == "best"
    assert [m["name"] for m in out["over_cap"]] == ["fourth"]


def test_the_threshold_keeps_rsa_l1_and_rsa_l2_apart_but_merges_twins():
    # Calibrated on human-data fits (2026-10-10): twins 0.0002, rsa_l1 vs rsa_l2 0.0070.
    assert 0.0002 < SAME_ON_POOL_RMSE < 0.0070
