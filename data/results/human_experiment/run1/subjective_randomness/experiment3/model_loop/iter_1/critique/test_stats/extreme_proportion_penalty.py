# name: extreme_proportion_penalty
# description: Mean choice rate of sequence A when it has extreme proportion imbalance (>0.6) and sequence B is balanced (<0.2).
import numpy as np


def test_statistic(df):
    mask = (df["imbalance_a"] > 0.6) & (df["imbalance_b"] < 0.2)
    return float(df.loc[mask, "chose_left"].mean())
