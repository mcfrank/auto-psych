# name: length_dependent_imbalance
# description: Rate of choosing the shorter sequence when both have high and similar imbalance, testing if long imbalanced sequences are penalized more.
import numpy as np


def test_statistic(df):
    high_imb = (df["imbalance_a"] > 0.2) & (df["imbalance_b"] > 0.2)
    similar_imb = np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.05
    diff_length = df["n_a"] != df["n_b"]
    mask = high_imb & similar_imb & diff_length
    if not mask.any():
        return 0.5

    shorter_left = df["n_a"] < df["n_b"]
    shorter_right = df["n_b"] < df["n_a"]
    chose_shorter = (
        df.loc[mask & shorter_left, "chose_left"].sum()
        + df.loc[mask & shorter_right, "chose_right"].sum()
    )
    return float(chose_shorter / mask.sum())
