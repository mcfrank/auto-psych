# name: over_alternating_vs_under_alternating
# description: The choice rate for option A when it is highly alternating (p_alts > 0.7) and option B is highly clumpy (p_alts < 0.5).
def test_statistic(df):
    mask = (df["p_alts_a"] > 0.7) & (df["p_alts_b"] < 0.5)
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, "chose_left"].mean())
