# name: trial_index_yes_slope
# description: OLS slope of yes (0/1) on trial_index scaled per 100 trials; observed more negative than null_mean means people grow more conservative over the session, a drift the stationary model cannot produce.
def test_statistic(df):
    x = df["trial_index"].astype(float).values / 100.0
    y = df["rating"].astype(float).values
    vx = np.var(x)
    return float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else 0.0
