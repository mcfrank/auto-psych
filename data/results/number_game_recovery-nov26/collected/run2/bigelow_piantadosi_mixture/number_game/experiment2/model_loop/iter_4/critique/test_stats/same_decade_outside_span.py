# name: same_decade_outside_span
# description: For targets 1-9 outside the span of the examples, yes rate when the target shares its tens digit with some example minus when it does not; observed above null means people use decade boundaries (tens-digit grouping) more than the model predicts.
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
    tens = np.floor(np.mod(X, 100) / 10)
    tt = np.floor(np.mod(t, 100) / 10)
    same = np.any(tens == tt[:, None], axis=1)
    out = ((t < mn) | (t > mx)) & (dist >= 1) & (dist <= 9)
    return _m(y, out & same) - _m(y, out & ~same)
