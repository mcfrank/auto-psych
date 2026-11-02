# name: single_example_gradient
# description: For one-example sets, yes rate for targets 1-5 away minus yes rate for targets 6-20 away; observed above null means people's similarity gradient around a lone example is steeper than the model's (it over-generalizes at mid distance), below null shallower.
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
    one = n == 1
    return _m(y, one & (dist >= 1) & (dist <= 5)) - _m(y, one & (dist >= 6) & (dist <= 20))
