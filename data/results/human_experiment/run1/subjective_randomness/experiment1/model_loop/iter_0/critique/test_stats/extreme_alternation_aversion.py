# name: extreme_alternation_aversion
# description: Rate of choosing the perfectly alternating sequence (p_alts > 0.9) against a sequence with moderate alternation (0.4 < p_alts < 0.6).
import numpy as np


def test_statistic(df):
    a_extreme = df["p_alts_a"] > 0.9
    b_mid = (df["p_alts_b"] > 0.4) & (df["p_alts_b"] < 0.6)
    b_extreme = df["p_alts_b"] > 0.9
    a_mid = (df["p_alts_a"] > 0.4) & (df["p_alts_a"] < 0.6)

    a_vs_b = a_extreme & b_mid
    b_vs_a = b_extreme & a_mid
    mask = a_vs_b | b_vs_a
    if not mask.any():
        return 0.5

    chose_extreme = (
        df.loc[a_vs_b, "chose_left"].sum() + df.loc[b_vs_a, "chose_right"].sum()
    )
    return float(chose_extreme / mask.sum())
