# name: length_preference_equal_imbalance
# description: The choice rate for option A when it is longer than B (n_a > n_b) but they have the same absolute imbalance proportion.
def test_statistic(df):
    mask = (
        (df["n_a"] > df["n_b"])
        & (df["imbalance_a"] == df["imbalance_b"])
        & (df["imbalance_a"] > 0.1)
    )
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, "chose_left"].mean())
