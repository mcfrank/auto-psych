# name: pref_longer_max_run
# description: Overall rate of choosing the sequence with a longer maximum run length.
def test_statistic(df):
    more_run_a = df["max_run_a"] > df["max_run_b"]
    more_run_b = df["max_run_b"] > df["max_run_a"]
    denom = more_run_a.sum() + more_run_b.sum()
    if denom == 0:
        return 0.0
    return (
        df.loc[more_run_a, "chose_left"].sum() + df.loc[more_run_b, "chose_right"].sum()
    ) / denom
