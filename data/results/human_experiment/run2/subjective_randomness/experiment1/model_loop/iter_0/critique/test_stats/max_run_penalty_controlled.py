# name: max_run_penalty_controlled
# description: The choice rate for option A when it has a longer maximum run than B, but similar alternation proportion (|p_alts_a - p_alts_b| <= 0.15).
def test_statistic(df):
    mask = (df["max_run_a"] > df["max_run_b"]) & (
        abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.15
    )
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, "chose_left"].mean())
