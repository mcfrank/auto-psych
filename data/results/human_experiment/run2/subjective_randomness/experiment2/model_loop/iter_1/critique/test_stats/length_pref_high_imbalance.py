# name: length_pref_high_imbalance
# description: The overall proportion of trials where the longer sequence was chosen, restricted to trials where both sequences are highly imbalanced (>=0.4) and unequal lengths.
import numpy as np


def test_statistic(df):
    mask = (
        (df["n_a"] != df["n_b"])
        & (df["imbalance_a"] >= 0.4)
        & (df["imbalance_b"] >= 0.4)
    )
    sub_df = df[mask]
    if len(sub_df) == 0:
        return np.nan
    chose_longer = np.where(
        sub_df["n_a"] > sub_df["n_b"], sub_df["chose_left"], sub_df["chose_right"]
    )
    return float(np.mean(chose_longer))
