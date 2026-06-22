# name: pref_more_alternations
# description: Overall rate of choosing the sequence with more overall alternations (alts).
def test_statistic(df):
    more_alts_a = df["alts_a"] > df["alts_b"]
    more_alts_b = df["alts_b"] > df["alts_a"]
    denom = more_alts_a.sum() + more_alts_b.sum()
    if denom == 0:
        return 0.0
    return (
        df.loc[more_alts_a, "chose_left"].sum()
        + df.loc[more_alts_b, "chose_right"].sum()
    ) / denom
