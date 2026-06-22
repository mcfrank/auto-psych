# name: pref_more_imbalanced
# description: Overall rate of choosing the sequence with a greater H/T imbalance proportion.
def test_statistic(df):
    more_imb_a = df["imbalance_a"] > df["imbalance_b"]
    more_imb_b = df["imbalance_b"] > df["imbalance_a"]
    denom = more_imb_a.sum() + more_imb_b.sum()
    if denom == 0:
        return 0.0
    return (
        df.loc[more_imb_a, "chose_left"].sum() + df.loc[more_imb_b, "chose_right"].sum()
    ) / denom
