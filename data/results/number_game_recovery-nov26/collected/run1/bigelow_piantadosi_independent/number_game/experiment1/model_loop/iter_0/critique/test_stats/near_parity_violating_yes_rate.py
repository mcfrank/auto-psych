# name: near_parity_violating_yes_rate
# description: Yes rate for non-example targets within 5 of the nearest example whose parity contradicts a single-parity set of >=2 (odd target for all-even sets, even for all-odd); observed > model means people generalise by numerical proximity more than the crisp rules + size principle allow.
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
    m = (n >= 2) & (dist > 0) & (dist <= 5) & ((ae & (t % 2 == 1)) | (ao & (t % 2 == 0)))
    return _rate(y, m)
