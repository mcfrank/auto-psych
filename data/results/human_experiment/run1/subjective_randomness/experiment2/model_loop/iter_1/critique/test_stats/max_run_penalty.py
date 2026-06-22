# name: max_run_penalty
# description: Correlation between difference in normalized max run length (A - B) and choice probability, for sequences with matched alternation rates.
import numpy as np


def test_statistic(df):
    sub = df[(df["p_alts_a"] - df["p_alts_b"]).abs() < 0.1]
    if len(sub) < 2:
        return np.nan
    max_run_diff = sub["max_run_norm_a"] - sub["max_run_norm_b"]
    if np.var(max_run_diff) < 1e-9 or np.var(sub["chose_left"]) < 1e-9:
        return np.nan
    return float(np.corrcoef(max_run_diff, sub["chose_left"])[0, 1])
