# name: fallback_corr_rep_motifs_b
# description: Pearson correlation between the response and the `rep_motifs_b` feature.
def test_statistic(df):
    x = df["chose_left"].astype(float)
    y = df["rep_motifs_b"].astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
