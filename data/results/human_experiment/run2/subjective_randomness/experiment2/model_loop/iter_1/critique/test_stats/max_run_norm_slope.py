# name: max_run_norm_slope
# description: The linear slope of the difference in normalized maximum run length predicting the choice of the left sequence.
import numpy as np


def test_statistic(df):
    x = df["max_run_norm_a"] - df["max_run_norm_b"]
    y = df["chose_left"]
    cov = np.cov(x, y)[0, 1]
    var = np.var(x, ddof=1)
    return float(cov / var) if var > 1e-8 else np.nan
