# name: imbalance_sensitivity_diff_by_length
# description: Difference in the slope of chose_left on (imbalance_b - imbalance_a) between long sequences (n >= 7) and short sequences (n <= 5).

import numpy as np


def test_statistic(df):
    long_mask = (df["n_a"] >= 7) & (df["n_b"] >= 7)
    short_mask = (df["n_a"] <= 5) & (df["n_b"] <= 5)

    def get_slope(mask):
        if mask.sum() < 2:
            return 0.0
        x = (df["imbalance_b"] - df["imbalance_a"])[mask]
        y = df["chose_left"][mask]
        if np.var(x) < 1e-8:
            return 0.0
        return float(np.cov(x, y)[0, 1] / np.var(x))

    return get_slope(long_mask) - get_slope(short_mask)
