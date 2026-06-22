# name: periodicity_penalty
# description: Correlation between difference in periodicity (A - B) and choice probability, for sequences of equal length.
import numpy as np


def test_statistic(df):
    sub = df[df["n_a"] == df["n_b"]]
    if len(sub) < 2:
        return np.nan
    per_diff = sub["periodicity_a"] - sub["periodicity_b"]
    if np.var(per_diff) < 1e-9 or np.var(sub["chose_left"]) < 1e-9:
        return np.nan
    return float(np.corrcoef(per_diff, sub["chose_left"])[0, 1])
