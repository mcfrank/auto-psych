# name: trial_index_drift
# description: OLS slope of yes (rating) on trial_index, per 100 trials; the model has no time dependence, so observed slope far from null_mean means people drift toward more (positive) or fewer (negative) yeses over the session than the model predicts.
def test_statistic(df):
    x = df["trial_index"].astype(float).values / 100.0
    y = df["rating"].astype(float).values
    vx = np.var(x)
    if vx == 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
