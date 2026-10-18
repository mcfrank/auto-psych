# name: trial_index_yes_slope
# description: OLS slope of rating on trial_index, in change of yes-probability per 100 trials; t_observed above null_mean (~0) means people drift toward saying yes over the session, which the stationary model cannot produce (below = drift toward no).
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    if den <= 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den * 100.0)
