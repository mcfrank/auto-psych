# name: trial_index_slope
# description: OLS slope of yes (rating) on trial_index/100 across all trials; observed more negative than null means people grow more conservative over the session while the model (no trial dependence) predicts a flat rate.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy() / 100.0
    y = df["rating"].astype(float).to_numpy()
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / denom)
