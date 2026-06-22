# name: length_pref_high_alternation
# description: The overall proportion of trials where the longer sequence was chosen, restricted to trials where both sequences have high alternation rates (>=0.6) and unequal lengths.
import numpy as np


def test_statistic(df):
    mask = (df["n_a"] != df["n_b"]) & (df["p_alts_a"] >= 0.6) & (df["p_alts_b"] >= 0.6)
    sub_df = df[mask]
    if len(sub_df) == 0:
        return np.nan
    chose_longer = np.where(
        sub_df["n_a"] > sub_df["n_b"], sub_df["chose_left"], sub_df["chose_right"]
    )
    return float(np.mean(chose_longer))
