# name: early_minus_late_yes_rate
# description: Yes rate in a session's first 100 trials minus yes rate after trial 300; observed > model means people endorse more early and become stricter over the session, a drift the stationary model cannot produce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    ti = df["trial_index"].astype(float).values
    return rate(ti < 100) - rate(ti >= 300)
