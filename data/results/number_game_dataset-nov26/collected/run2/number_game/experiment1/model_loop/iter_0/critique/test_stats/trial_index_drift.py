# name: trial_index_drift
# description: Difference in yes-rate between late trials (trial_index above the median) and early trials; nonzero observed vs null near 0 means humans drift in yes-bias over the session, which the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy().astype(float)
    y = df["rating"].to_numpy().astype(float)
    late = ti > np.median(ti)
    if late.all() or (~late).all():
        return 0.0
    return float(y[late].mean() - y[~late].mean())
