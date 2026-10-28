# name: descending_downward_generalization
# description: In sets that go down overall (last < first), yes-rate for targets 1-30 below the smallest example minus targets 1-30 above the largest; observed < null means people generalize downward less than upward even with no upward starting point (the model, lacking a start penalty and with momentum favouring the travel direction, over-produces downward yeses here).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), k - 1]
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    desc = last < first
    return rate(desc & (mn - t >= 1) & (mn - t <= 30)) - rate(desc & (t - mx >= 1) & (t - mx <= 30))
