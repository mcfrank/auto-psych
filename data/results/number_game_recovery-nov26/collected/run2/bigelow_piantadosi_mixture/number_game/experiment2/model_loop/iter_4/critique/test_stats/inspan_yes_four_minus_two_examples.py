# name: inspan_yes_four_minus_two_examples
# description: Yes rate for non-example targets inside the example span, 4-example sets minus 2-example sets; observed above null means more examples raise interior acceptance more than the model predicts (model over-applies the size principle / strays for larger sets), below null the reverse.
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
    ins = (t > mn) & (t < mx) & (dist > 0)
    return _m(y, ins & (n == 4)) - _m(y, ins & (n == 2))
