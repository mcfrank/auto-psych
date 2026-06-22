# name: pref_longer_max_run_matched_rep_motifs
# description: Rate of choosing the sequence with a longer maximum run, conditioned on identical number of repetition motifs.
def test_statistic(df):
    subset = df[df["rep_motifs_a"] == df["rep_motifs_b"]]
    more_run_a = subset["max_run_a"] > subset["max_run_b"]
    more_run_b = subset["max_run_b"] > subset["max_run_a"]
    denom = more_run_a.sum() + more_run_b.sum()
    if denom == 0:
        return 0.0
    return (
        subset.loc[more_run_a, "chose_left"].sum()
        + subset.loc[more_run_b, "chose_right"].sum()
    ) / denom
