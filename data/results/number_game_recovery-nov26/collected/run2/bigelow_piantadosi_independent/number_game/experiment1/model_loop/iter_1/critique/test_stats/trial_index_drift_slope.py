# name: trial_index_drift_slope
# description: OLS slope of yes (0/1) on trial_index scaled to [0,1] by its max; nonzero observed vs null near 0 means people's yes rate drifts over the session (fatigue/learning), which the static model cannot produce.
def test_statistic(df):
    x = df["trial_index"].to_numpy(dtype=float)
    mx = x.max()
    if mx <= 0:
        return 0.0
    x = x / mx
    y = df["rating"].to_numpy(dtype=float)
    xc = x - x.mean()
    den = (xc ** 2).sum()
    if den == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den)
