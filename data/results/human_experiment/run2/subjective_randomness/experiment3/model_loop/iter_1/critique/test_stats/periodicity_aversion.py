# name: periodicity_aversion
# description: Proportion of trials where the less periodic sequence is chosen, conditioned on a substantial difference in periodicity (> 0.2) and similar alternation rates (|p_alts_a - p_alts_b| <= 0.1).
import numpy as np


def test_statistic(df):
    mask = (np.abs(df["periodicity_a"] - df["periodicity_b"]) > 0.2) & (
        np.abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.1
    )
    sub = df[mask]
    if len(sub) == 0:
        return np.nan

    chose_less_periodic = np.where(
        sub["periodicity_a"] < sub["periodicity_b"],
        sub["chose_left"],
        sub["chose_right"],
    )
    return float(np.mean(chose_less_periodic))
