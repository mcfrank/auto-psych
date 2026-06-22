# name: length_penalty_scaling
# description: Mean choice rate of the shorter sequence when both sequences are highly imbalanced (>0.4), testing if the penalty accumulates linearly.
import numpy as np


def test_statistic(df):
    mask = (
        (df["imbalance_a"] > 0.4) & (df["imbalance_b"] > 0.4) & (df["n_a"] != df["n_b"])
    )
    shorter_chosen = np.where(
        df.loc[mask, "n_a"] < df.loc[mask, "n_b"],
        df.loc[mask, "chose_left"],
        1 - df.loc[mask, "chose_left"],
    )
    return float(shorter_chosen.mean())
