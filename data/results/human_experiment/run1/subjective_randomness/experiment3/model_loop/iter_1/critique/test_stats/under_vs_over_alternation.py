# name: under_vs_over_alternation
# description: Mean choice rate of sequence A when A is under-alternating (<0.4) and B is over-alternating (>0.6) with similar imbalance.
import numpy as np


def test_statistic(df):
    mask = (
        (df["p_alts_a"] < 0.4)
        & (df["p_alts_b"] > 0.6)
        & (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.2)
    )
    return float(df.loc[mask, "chose_left"].mean())
