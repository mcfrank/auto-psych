# name: late_minus_early_yes_rate
# description: Overall yes-rate on trials in the upper half of trial_index minus the lower half; nonzero observed vs a ~0 null means responding drifts over the session, which the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    med = np.median(ti); a = ti > med; b = ti <= med
    x = r[a].mean() if a.any() else 0.0; y = r[b].mean() if b.any() else 0.0
    return float(x - y)
