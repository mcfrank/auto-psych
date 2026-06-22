# name: length_imbalance_interaction
# description: Correlation between preference for A and length difference (A - B) when both sequences are highly imbalanced (imbalance > 0.25).
def test_statistic(df):
    import numpy as np

    mask = (
        (df["n_a"] != df["n_b"])
        & (df["imbalance_a"] > 0.25)
        & (df["imbalance_b"] > 0.25)
    )
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
        )[0, 1]
    )
