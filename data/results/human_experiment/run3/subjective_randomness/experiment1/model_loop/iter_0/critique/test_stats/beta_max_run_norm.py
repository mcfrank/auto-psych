# name: beta_max_run_norm
# description: Linear regression slope of chose_left on the difference in normalized max run length (max_run_norm_b - max_run_norm_a).

import numpy as np


def test_statistic(df):
    x = df["max_run_norm_b"] - df["max_run_norm_a"]
    y = df["chose_left"]
    if np.var(x) < 1e-8:
        return 0.0
    return float(np.cov(x, y)[0, 1] / np.var(x))
