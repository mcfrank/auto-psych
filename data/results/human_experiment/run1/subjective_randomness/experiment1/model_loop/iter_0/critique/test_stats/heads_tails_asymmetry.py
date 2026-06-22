# name: heads_tails_asymmetry
# description: Rate of choosing the sequence with more heads, when imbalance is identical but they skew in opposite directions.
import numpy as np


def test_statistic(df):
    same_imb = np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.01
    opp_sides = (df["p_a"] - 0.5) * (df["p_b"] - 0.5) < 0
    mask = same_imb & opp_sides
    if not mask.any():
        return 0.5

    more_heads_left = df["p_a"] > df["p_b"]
    more_heads_right = df["p_b"] > df["p_a"]
    chose_more_heads = (
        df.loc[mask & more_heads_left, "chose_left"].sum()
        + df.loc[mask & more_heads_right, "chose_right"].sum()
    )
    return float(chose_more_heads / mask.sum())
