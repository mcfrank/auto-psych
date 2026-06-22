# name: periodicity_preference_controlled
# description: The choice rate for option A when it is more periodic than B, but has a similar alternation proportion (|p_alts_a - p_alts_b| <= 0.15).
def test_statistic(df):
    mask = (df["periodicity_a"] > df["periodicity_b"]) & (
        abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.15
    )
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, "chose_left"].mean())
