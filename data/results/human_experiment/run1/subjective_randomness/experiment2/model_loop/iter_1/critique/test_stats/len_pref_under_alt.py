# name: len_pref_under_alt
# description: Rate of choosing the longer sequence when both sequences have very low alternation rates (p_alts < 0.3) and unequal lengths.
import numpy as np


def test_statistic(df):
    sub = df[(df["p_alts_a"] < 0.3) & (df["p_alts_b"] < 0.3) & (df["n_a"] != df["n_b"])]
    if len(sub) == 0:
        return np.nan
    is_longer_chosen = np.where(
        sub["n_a"] > sub["n_b"], sub["chose_left"], 1 - sub["chose_left"]
    )
    return float(np.mean(is_longer_chosen))
