# name: rep_motif_slope
# description: Linear-regression slope of chose_left on the signed difference in normalized repetition-motif count ((rep_motifs_a/n_a) - (rep_motifs_b/n_b)); probes whether the model distinguishes run-structured sequences beyond the global alternation rate.
def test_statistic(df):
    na = df["n_a"].clip(lower=1)
    nb = df["n_b"].clip(lower=1)
    d = df["rep_motifs_a"] / na - df["rep_motifs_b"] / nb
    if d.nunique() < 2:
        return 0.0
    return float(np.polyfit(d, df["chose_left"], 1)[0])
