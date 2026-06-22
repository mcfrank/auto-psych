import numpy as np
import pandas as pd


# name: heads_vs_tails_bias
# description: The average choice rate for sequence A when A has a majority of heads (p_a > 0.5) and B has a majority of tails (p_b < 0.5), restricting to trials with similar absolute imbalance.
def test_statistic(df):
    mask = (
        (df["p_a"] > 0.5)
        & (df["p_b"] < 0.5)
        & (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.05)
    )
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
