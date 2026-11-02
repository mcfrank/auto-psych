# name: below_minus_above_near_edge_asymmetry
# description: Yes rate for targets 1-6 below the smallest example minus yes rate for targets 1-6 above the largest example (sets of 2+ examples); observed above null means people extend ranges downward more than upward, an asymmetry the symmetric margin prior misses.
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
    b = mn - t
    a = t - mx
    k = n >= 2
    return _m(y, k & (b >= 1) & (b <= 6)) - _m(y, k & (a >= 1) & (a <= 6))
