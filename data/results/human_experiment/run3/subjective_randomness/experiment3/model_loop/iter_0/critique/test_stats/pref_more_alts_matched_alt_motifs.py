# name: pref_more_alts_matched_alt_motifs
# description: Rate of choosing the sequence with more alternations, conditioned on identical number of alternating motifs.
def test_statistic(df):
    subset = df[df["alt_motifs_a"] == df["alt_motifs_b"]]
    more_alts_a = subset["alts_a"] > subset["alts_b"]
    more_alts_b = subset["alts_b"] > subset["alts_a"]
    denom = more_alts_a.sum() + more_alts_b.sum()
    if denom == 0:
        return 0.0
    return (
        subset.loc[more_alts_a, "chose_left"].sum()
        + subset.loc[more_alts_b, "chose_right"].sum()
    ) / denom
