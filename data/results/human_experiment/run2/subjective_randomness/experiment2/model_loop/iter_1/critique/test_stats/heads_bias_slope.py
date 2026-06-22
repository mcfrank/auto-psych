# name: heads_bias_slope
# description: The linear slope of the difference in the proportion of heads predicting the choice, testing if participants break symmetry between heads and tails.
import numpy as np


def test_statistic(df):
    x = df["p_a"] - df["p_b"]
    y = df["chose_left"]
    cov = np.cov(x, y)[0, 1]
    var = np.var(x, ddof=1)
    return float(cov / var) if var > 1e-8 else np.nan
