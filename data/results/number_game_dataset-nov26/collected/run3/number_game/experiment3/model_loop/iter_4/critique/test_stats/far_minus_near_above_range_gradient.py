# name: far_minus_near_above_range_gradient
# description: Among targets above the set maximum, yes-rate for targets >30 above max minus yes-rate for targets 1-10 above max; observed > model means people extend upward generalization farther (flatter decay) than the model.
def _feat(df):
    s = df["number_set"].astype(str).str.split(",", expand=True)
    s = s.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    a = s.to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(a), axis=1)
    mn = np.nanmin(a, axis=1)
    mx = np.nanmax(a, axis=1)
    is_ex = np.any(a == t[:, None], axis=1)
    par = np.where(np.isnan(a), np.nan, np.mod(a, 2))
    allodd = np.nansum(par == 1, axis=1) == n
    alleven = np.nansum(par == 0, axis=1) == n
    return t, y, n, mn, mx, is_ex, allodd, alleven

def _m(y, mask):
    return float(np.mean(y[mask])) if np.any(mask) else 0.0

def test_statistic(df):
    t, y, n, mn, mx, ex, ao, ae = _feat(df)
    d = t - mx
    return _m(y, d > 30) - _m(y, (d >= 1) & (d <= 10))
