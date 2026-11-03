# name: tight_cluster_outside_yes
# description: For sets of 2+ examples spanning at most 10, yes rate for targets 1-5 outside the span minus yes rate for non-example targets inside it; observed above null means people blur the edges of tight clusters more (and/or fill them in less) than the size principle in the model predicts.
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
    tight = (n >= 2) & (mx - mn <= 10)
    inside = (t >= mn) & (t <= mx) & (dist > 0)
    near_out = ((t < mn) | (t > mx)) & (dist <= 5)
    return _m(y, tight & near_out) - _m(y, tight & inside)
