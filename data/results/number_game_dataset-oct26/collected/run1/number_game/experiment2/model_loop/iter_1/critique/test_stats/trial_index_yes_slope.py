# name: trial_index_yes_slope
# description: OLS slope of rating on trial_index scaled to change in yes-probability per 100 trials; t_observed differing from null_mean (~0) means participants drift toward more (positive) or fewer (negative) yes responses over the session, which the stationary model cannot produce.
def test_statistic(df):
    x = df["trial_index"].values.astype(float)
    y = df["rating"].values.astype(float)
    xc = x - x.mean()
    den = (xc * xc).sum()
    if den == 0:
        return 0.0
    return float(100.0 * (xc * (y - y.mean())).sum() / den)
