# name: periodicity_aversion
# description: Correlation between the preference for sequence A and the difference in periodicity (B - A), for equal length sequences.
def test_statistic(df):
    import numpy as np

    mask = (df["n_a"] == df["n_b"]) & (df["periodicity_a"] != df["periodicity_b"])
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "periodicity_b"] - df.loc[mask, "periodicity_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )
