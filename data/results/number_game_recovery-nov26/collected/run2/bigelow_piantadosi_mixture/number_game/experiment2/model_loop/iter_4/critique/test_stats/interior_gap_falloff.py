# name: interior_gap_falloff
# description: Among non-example targets strictly inside the span of 2+ examples, yes rate when the target is 8+ from the nearest example minus when it is 1-3 away; observed below null means people treat big interior gaps as holes more than the range-based model predicts (similarity falloff inside the span).
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
    ins = (n >= 2) & (t > mn) & (t < mx) & (dist > 0)
    return _m(y, ins & (dist >= 8)) - _m(y, ins & (dist <= 3))
