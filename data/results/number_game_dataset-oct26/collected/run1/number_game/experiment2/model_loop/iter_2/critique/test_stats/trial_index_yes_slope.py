# name: trial_index_yes_slope
# description: OLS slope of rating on trial_index, in change of yes-probability per 100 trials; t_observed above null_mean (~0) means people drift toward saying yes more over the session, which the stationary model cannot produce (below = drift toward no).
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    xc = x - x.mean()
    den = (xc ** 2).sum()
    if den <= 0:
        return 0.0
    return float(100.0 * (xc * (y - y.mean())).sum() / den)
