# name: trial_index_yes_slope
# description: OLS slope of yes response on trial_index (per 100 trials); model has no time dependence, so a nonzero observed slope vs null_mean ~0 indicates drift in yes-bias over the session.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    x = df["trial_index"].astype(float).to_numpy() / 100.0
    x = x - x.mean()
    v = (x * x).sum()
    return float((x * (r - r.mean())).sum() / v) if v > 0 else 0.0
