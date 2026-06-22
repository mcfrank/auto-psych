# name: length_imbalance_interaction_slope
# description: The slope of the product of length and imbalance (n * imbalance) predicting the choice, testing if the model misestimates the interaction between length and imbalance.
import numpy as np


def test_statistic(df):
    x = (df["n_a"] * df["imbalance_a"]) - (df["n_b"] * df["imbalance_b"])
    y = df["chose_left"]
    cov = np.cov(x, y)[0, 1]
    var = np.var(x, ddof=1)
    return float(cov / var) if var > 1e-8 else np.nan
