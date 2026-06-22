# name: pref_longer_sequence
# description: Overall rate of choosing the sequence with the greater length.
def test_statistic(df):
    longer_a = df["n_a"] > df["n_b"]
    longer_b = df["n_b"] > df["n_a"]
    denom = longer_a.sum() + longer_b.sum()
    if denom == 0:
        return 0.0
    return (
        df.loc[longer_a, "chose_left"].sum() + df.loc[longer_b, "chose_right"].sum()
    ) / denom
