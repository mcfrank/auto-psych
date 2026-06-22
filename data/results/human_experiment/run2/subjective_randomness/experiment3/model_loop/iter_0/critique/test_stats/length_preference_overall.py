# name: length_preference_overall
# description: Correlation between preference for A and length difference (A - B), capturing the main effect of length.
def test_statistic(df):
    import numpy as np

    mask = df["n_a"] != df["n_b"]
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
        )[0, 1]
    )
