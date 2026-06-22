# name: beta_periodicity
# description: Linear regression slope of chose_left on the difference in periodicity (periodicity_b - periodicity_a).

import numpy as np


def test_statistic(df):
    x = df["periodicity_b"] - df["periodicity_a"]
    y = df["chose_left"]
    if np.var(x) < 1e-8:
        return 0.0
    return float(np.cov(x, y)[0, 1] / np.var(x))
