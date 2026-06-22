# name: corr_max_run_diff
# description: The Pearson correlation between the difference in max run length (max_run_a - max_run_b) and the choice of A.
def test_statistic(df):
    import numpy as np

    return float(np.corrcoef(df["max_run_a"] - df["max_run_b"], df["chose_left"])[0, 1])
