# name: len_effect_imbalanced_vs_balanced
# description: Correlation between n_a and chose_left when Sequence A is highly imbalanced (imbalance > 0.6) and Sequence B is balanced (imbalance < 0.2).
import numpy as np


def test_statistic(df):
    sub = df[(df["imbalance_a"] > 0.6) & (df["imbalance_b"] < 0.2)]
    if len(sub) < 2 or np.var(sub["n_a"]) < 1e-9 or np.var(sub["chose_left"]) < 1e-9:
        return np.nan
    return float(np.corrcoef(sub["n_a"], sub["chose_left"])[0, 1])
