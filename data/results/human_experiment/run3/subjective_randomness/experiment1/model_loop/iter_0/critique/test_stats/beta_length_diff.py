# name: beta_length_diff
# description: Slope of chose_left on the difference in sequence lengths (n_a - n_b).

import numpy as np


def test_statistic(df):
    x = df["n_a"] - df["n_b"]
    y = df["chose_left"]
    if np.var(x) < 1e-8:
        return 0.0
    return float(np.cov(x, y)[0, 1] / np.var(x))
