# name: trial_position_yes_drift
# description: Yes-rate on trials in the later half of the session (trial_index above the median) minus yes-rate on the earlier half; the model has no time dependence, so observed > null means people become more liberal as the session goes on (and observed < null more conservative), a drift the stationary model fails to produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy(dtype=float)
    r = df["rating"].to_numpy(dtype=float)
    m = np.median(ti)
    a, b = ti > m, ti <= m
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
