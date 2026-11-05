# name: trial_index_slope
# description: OLS slope of yes-response on trial_index scaled per 100 trials; nonzero observed vs a ~0 null means the model (which has no time dependence) misses drift in yes-bias over the session (positive = more yes later).
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].values.astype(float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0] * 100.0)
