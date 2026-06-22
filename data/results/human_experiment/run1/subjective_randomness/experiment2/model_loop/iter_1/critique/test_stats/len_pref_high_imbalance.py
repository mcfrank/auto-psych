# name: len_pref_high_imbalance
# description: Rate of choosing the longer sequence when both sequences are highly imbalanced (imbalance > 0.6) and lengths are unequal.
import numpy as np


def test_statistic(df):
    sub = df[
        (df["imbalance_a"] > 0.6) & (df["imbalance_b"] > 0.6) & (df["n_a"] != df["n_b"])
    ]
    if len(sub) == 0:
        return np.nan
    is_longer_chosen = np.where(
        sub["n_a"] > sub["n_b"], sub["chose_left"], 1 - sub["chose_left"]
    )
    return float(np.mean(is_longer_chosen))
