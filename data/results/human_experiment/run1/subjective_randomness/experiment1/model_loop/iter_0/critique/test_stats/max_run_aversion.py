# name: max_run_aversion
# description: Rate of choosing the sequence with the shorter normalized max run, when alternation rate and imbalance are similar.
import numpy as np


def test_statistic(df):
    similar_alts = np.abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.1
    similar_imb = np.abs(df["imbalance_a"] - df["imbalance_b"]) <= 0.1
    diff_max_run = df["max_run_norm_a"] != df["max_run_norm_b"]
    mask = similar_alts & similar_imb & diff_max_run
    if not mask.any():
        return 0.5

    smaller_run_left = df["max_run_norm_a"] < df["max_run_norm_b"]
    smaller_run_right = df["max_run_norm_b"] < df["max_run_norm_a"]
    chose_smaller_run = (
        df.loc[mask & smaller_run_left, "chose_left"].sum()
        + df.loc[mask & smaller_run_right, "chose_right"].sum()
    )
    return float(chose_smaller_run / mask.sum())
