# name: max_run_aversion
# description: Proportion of trials where the sequence with the shorter normalized maximum run is chosen, conditioned on similar alternation rates (|p_alts_a - p_alts_b| <= 0.1).
import numpy as np


def test_statistic(df):
    mask = (np.abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.1) & (
        df["max_run_norm_a"] != df["max_run_norm_b"]
    )
    sub = df[mask]
    if len(sub) == 0:
        return np.nan

    chose_smaller_run = np.where(
        sub["max_run_norm_a"] < sub["max_run_norm_b"],
        sub["chose_left"],
        sub["chose_right"],
    )
    return float(np.mean(chose_smaller_run))
