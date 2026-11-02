# name: parity_match_outside_span
# description: For sets of 3+ examples that are all even or all odd, yes rate for targets outside the span with the same parity minus opposite parity; observed above null means people generalize the parity rule beyond the range more than the model predicts, below null less.
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
    par = np.where(np.isnan(X), np.nan, np.mod(X, 2))
    s = np.nansum(par, axis=1)
    alleven = (s == 0)
    allodd = (s == n)
    k = (n >= 3) & (alleven | allodd) & ((t < mn) | (t > mx))
    tp = np.mod(t, 2)
    match = (alleven & (tp == 0)) | (allodd & (tp == 1))
    return _m(y, k & match) - _m(y, k & ~match)
