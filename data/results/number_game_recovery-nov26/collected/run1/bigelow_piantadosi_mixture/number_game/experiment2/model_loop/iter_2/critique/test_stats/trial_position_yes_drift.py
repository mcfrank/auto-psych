# name: trial_position_yes_drift
# description: Overall yes-rate in trials above the median trial_index minus below; observed different from null means response bias drifts over the session in a way the model does not capture.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    ti = df["trial_index"].astype(float).to_numpy()
    med = np.median(ti)
    return rate(ti > med) - rate(ti <= med)
