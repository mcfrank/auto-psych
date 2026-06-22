# name: corr_periodicity_diff
# description: The Pearson correlation between the difference in periodicity (periodicity_a - periodicity_b) and the choice of A.
def test_statistic(df):
    import numpy as np

    return float(
        np.corrcoef(df["periodicity_a"] - df["periodicity_b"], df["chose_left"])[0, 1]
    )
