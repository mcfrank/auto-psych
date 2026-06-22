# name: heads_vs_tails_asymmetry
# description: Mean choice rate of sequence A when A is head-heavy and B is tail-heavy, but equally imbalanced, testing for asymmetry.
import numpy as np


def test_statistic(df):
    mask = (
        (df["p_a"] > 0.6)
        & (df["p_b"] < 0.4)
        & (np.abs((df["p_a"] - 0.5) + (df["p_b"] - 0.5)) < 0.1)
    )
    return float(df.loc[mask, "chose_left"].mean())
