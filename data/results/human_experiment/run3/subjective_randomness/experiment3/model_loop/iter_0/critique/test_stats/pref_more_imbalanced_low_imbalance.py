# name: pref_more_imbalanced_low_imbalance
# description: Rate of choosing the more imbalanced sequence when both sequences have low imbalance (<= 0.3).
def test_statistic(df):
    subset_idx = (
        (df["imbalance_a"] <= 0.3)
        & (df["imbalance_b"] <= 0.3)
        & (df["imbalance_a"] != df["imbalance_b"])
    )
    subset = df[subset_idx]
    if len(subset) == 0:
        return 0.0

    more_imb_a = subset["imbalance_a"] > subset["imbalance_b"]
    more_imb_b = subset["imbalance_b"] > subset["imbalance_a"]

    return (
        subset.loc[more_imb_a, "chose_left"].sum()
        + subset.loc[more_imb_b, "chose_right"].sum()
    ) / len(subset)
