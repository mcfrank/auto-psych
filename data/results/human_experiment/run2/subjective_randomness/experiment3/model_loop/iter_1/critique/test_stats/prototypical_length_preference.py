# name: prototypical_length_preference
# description: Proportion of trials where the longer sequence is chosen, conditioned on both sequences being prototypical (imbalance < 0.25 and 0.4 <= p_alts <= 0.6) and having different lengths.
import numpy as np


def test_statistic(df):
    mask = (
        (df["imbalance_a"] < 0.25)
        & (df["imbalance_b"] < 0.25)
        & (df["p_alts_a"] >= 0.4)
        & (df["p_alts_a"] <= 0.6)
        & (df["p_alts_b"] >= 0.4)
        & (df["p_alts_b"] <= 0.6)
        & (df["n_a"] != df["n_b"])
    )
    sub = df[mask]
    if len(sub) == 0:
        return np.nan

    chose_longer = np.where(
        sub["n_a"] > sub["n_b"], sub["chose_left"], sub["chose_right"]
    )
    return float(np.mean(chose_longer))
