# name: short_vs_long_preference
# description: Mean choice rate of sequence A when A is very short (n <= 4) and B is long (n >= 7), capturing baseline typicality length effects.
import numpy as np


def test_statistic(df):
    mask = (df["n_a"] <= 4) & (df["n_b"] >= 7)
    return float(df.loc[mask, "chose_left"].mean())
