# name: imbalanced_length_aversion
# description: Proportion of trials where the shorter sequence is chosen, conditioned on both sequences being highly imbalanced (imbalance >= 0.5) and having different lengths.
import numpy as np


def test_statistic(df):
    mask = (
        (df["imbalance_a"] >= 0.5)
        & (df["imbalance_b"] >= 0.5)
        & (df["n_a"] != df["n_b"])
    )
    sub = df[mask]
    if len(sub) == 0:
        return np.nan

    chose_shorter = np.where(
        sub["n_a"] < sub["n_b"], sub["chose_left"], sub["chose_right"]
    )
    return float(np.mean(chose_shorter))
