# name: fallback_corr_max_run_norm_b
# description: Pearson correlation between the response and the `max_run_norm_b` feature.
def test_statistic(df):
    x = df["chose_left"].astype(float)
    y = df["max_run_norm_b"].astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
