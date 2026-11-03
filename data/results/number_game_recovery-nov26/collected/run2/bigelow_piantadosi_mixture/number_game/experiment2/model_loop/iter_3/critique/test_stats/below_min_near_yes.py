# name: below_min_near_yes
# description: Yes rate for targets 1-5 below the smallest example; observed above null means the model under-extends the range downward just past the examples' lower edge.
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
    b = mn - t
    return _m(y, (b >= 1) & (b <= 5))
