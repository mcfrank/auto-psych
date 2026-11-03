# name: wide_span_interior_yes
# description: Yes rate for non-example targets inside the span of sets of 2+ examples spanning 50 or more; observed above null means people accept broad intervals for scattered examples more than the narrow-range-favouring model predicts, below null less.
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
    k = (n >= 2) & ((mx - mn) >= 50) & (t > mn) & (t < mx) & (dist > 0)
    return _m(y, k)
