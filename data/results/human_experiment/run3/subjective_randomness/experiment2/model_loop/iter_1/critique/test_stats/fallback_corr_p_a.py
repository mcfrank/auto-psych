# name: fallback_corr_p_a
# description: Pearson correlation between the response and the `p_a` feature.
def test_statistic(df):
    x = df["chose_left"].astype(float)
    y = df["p_a"].astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
