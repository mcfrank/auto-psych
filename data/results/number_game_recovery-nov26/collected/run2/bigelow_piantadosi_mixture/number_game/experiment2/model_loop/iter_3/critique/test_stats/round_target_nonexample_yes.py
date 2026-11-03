# name: round_target_nonexample_yes
# description: Yes rate for targets that are multiples of 10 and not among the examples, on sets where not every example is a multiple of 10; observed below null means people reject round numbers more than the model predicts (round targets feel unrelated), above null means a round-number attraction.
def _m(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    allround = np.nansum(X % 10 == 0, axis=1) == n
    return _m(y, (t % 10 == 0) & (dist > 0) & ~allround)
