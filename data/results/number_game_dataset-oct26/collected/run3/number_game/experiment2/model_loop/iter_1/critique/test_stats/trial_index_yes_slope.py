# name: trial_index_yes_slope
# description: OLS slope of yes (0/1) on trial_index (per 100 trials); observed > model means people drift toward saying "yes" more as the session goes on, which the trial-invariant model cannot produce, observed < model means a drift toward "no".
def test_statistic(df):
    x = df["trial_index"].to_numpy(dtype=float) / 100.0
    y = df["rating"].to_numpy(dtype=float)
    xc = x - x.mean()
    v = (xc ** 2).sum()
    if v == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / v)
