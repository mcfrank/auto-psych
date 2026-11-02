# name: yes_rate_trial_slope
# description: OLS slope of rating (1=yes) on trial_index rescaled to [0,1]; observed more negative than null means people say yes less often as the session goes on, a drift the stationary model does not produce.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    x = x / max(x.max(), 1.0)
    xc = x - x.mean()
    den = (xc ** 2).sum()
    if den <= 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den)
