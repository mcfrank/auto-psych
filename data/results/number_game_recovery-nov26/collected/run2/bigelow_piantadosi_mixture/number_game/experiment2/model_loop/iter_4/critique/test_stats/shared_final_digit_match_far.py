# name: shared_final_digit_match_far
# description: For sets of 2+ examples all ending in the same digit, yes rate for non-example targets ending in that digit and more than 10 from every example; observed above null means people generalize the ends-in rule beyond the cluster more than the model predicts, below null less.
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
    ld = np.mod(X, 10)
    first = ld[:, 0]
    same = np.all((ld == first[:, None]) | np.isnan(ld), axis=1) & (n >= 2)
    match = np.mod(t, 10) == first
    return _m(y, same & match & (dist > 10))
