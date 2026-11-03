# name: heading_direction_extension
# description: For monotone (strictly ascending or descending as shown) sets of 3+ examples, yes rate for targets 1-15 beyond the end the sequence is heading toward minus beyond the end it came from; observed above null means people extrapolate in the direction of travel, which the order-blind model cannot produce.
def _prep(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    return X, n, t, y, mn, mx, dist

def _m(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0


def test_statistic(df):
    X, n, t, y, mn, mx, dist = _prep(df)
    d = np.diff(X, axis=1)
    valid = ~np.isnan(d)
    up = np.all((d > 0) | ~valid, axis=1) & (n >= 3)
    dn = np.all((d < 0) | ~valid, axis=1) & (n >= 3)
    a = t - mx
    b = mn - t
    above = (a >= 1) & (a <= 15)
    below = (b >= 1) & (b <= 15)
    ahead = (up & above) | (dn & below)
    behind = (up & below) | (dn & above)
    return _m(y, ahead) - _m(y, behind)
