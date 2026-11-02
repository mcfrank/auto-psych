# name: in_range_setsize_slope
# description: Least-squares slope of yes on number of examples among non-example targets inside the set's [min-5, max+5] window; observed less negative (higher) than model means people tighten generalisation with more examples less than the size principle predicts.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    all_even = np.nansum(par, axis=1) == 0
    all_odd = np.nansum(par, axis=1) == n
    return ex, t, y, n, dist, lo, hi, all_even, all_odd


def _rate(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0


def test_statistic(df):
    ex, t, y, n, dist, lo, hi, ae, ao = _parse(df)
    m = (t >= lo - 5) & (t <= hi + 5) & (dist > 0)
    x = n[m].astype(float)
    yy = y[m]
    if m.sum() < 3 or np.var(x) == 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (yy - yy.mean())) / np.var(x))
