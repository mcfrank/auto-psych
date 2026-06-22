# name: under_alternation_aversion
# description: Proportion of trials where a moderately alternating sequence (0.4 <= p_alts <= 0.6) is chosen over a severely under-alternating sequence (p_alts <= 0.2), conditioned on similar imbalance.
import numpy as np


def test_statistic(df):
    # A is moderate, B is under-alternating
    mask1 = (
        (df["p_alts_a"] >= 0.4)
        & (df["p_alts_a"] <= 0.6)
        & (df["p_alts_b"] <= 0.2)
        & (np.abs(df["imbalance_a"] - df["imbalance_b"]) <= 0.25)
    )
    # B is moderate, A is under-alternating
    mask2 = (
        (df["p_alts_b"] >= 0.4)
        & (df["p_alts_b"] <= 0.6)
        & (df["p_alts_a"] <= 0.2)
        & (np.abs(df["imbalance_a"] - df["imbalance_b"]) <= 0.25)
    )

    sub1 = df[mask1]
    sub2 = df[mask2]

    total = len(sub1) + len(sub2)
    if total == 0:
        return np.nan

    choices_mod = np.sum(sub1["chose_left"]) + np.sum(sub2["chose_right"])
    return float(choices_mod / total)
