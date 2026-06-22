# name: pref_higher_periodicity_matched_motifs
# description: Rate of choosing the sequence with higher periodicity, conditioned on identical number of repetition and alternating motifs.
def test_statistic(df):
    subset_idx = (
        (df["alt_motifs_a"] == df["alt_motifs_b"])
        & (df["rep_motifs_a"] == df["rep_motifs_b"])
        & (df["periodicity_a"] != df["periodicity_b"])
    )
    subset = df[subset_idx]
    if len(subset) == 0:
        return 0.0

    more_per_a = subset["periodicity_a"] > subset["periodicity_b"]
    more_per_b = subset["periodicity_b"] > subset["periodicity_a"]

    return (
        subset.loc[more_per_a, "chose_left"].sum()
        + subset.loc[more_per_b, "chose_right"].sum()
    ) / len(subset)
