# name: rep_motifs_preference_controlled
# description: The choice rate for option A when it has more repeating motifs than B, despite a similar alternation proportion (|p_alts_a - p_alts_b| <= 0.15).
def test_statistic(df):
    mask = (df["rep_motifs_a"] > df["rep_motifs_b"]) & (
        abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.15
    )
    if mask.sum() == 0:
        return 0.0
    return float(df.loc[mask, "chose_left"].mean())
