# name: extreme_alternation_penalty
# description: Mean choice rate of sequence A when it has an extreme alternation rate (0 or 1) and sequence B is moderate (0.2 to 0.8).
import numpy as np


def test_statistic(df):
    mask = (
        ((df["p_alts_a"] == 0) | (df["p_alts_a"] == 1))
        & (df["p_alts_b"] > 0.2)
        & (df["p_alts_b"] < 0.8)
    )
    return float(df.loc[mask, "chose_left"].mean())
