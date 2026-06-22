# name: max_run_preference
# description: Correlation between the preference for sequence A and the difference in maximum run lengths (B - A), for equal length sequences.
def test_statistic(df):
    import numpy as np

    mask = (df["n_a"] == df["n_b"]) & (df["max_run_norm_a"] != df["max_run_norm_b"])
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "max_run_norm_b"] - df.loc[mask, "max_run_norm_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )
