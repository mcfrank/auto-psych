# name: inspan_below_start_overreach
# description: In sets that go up overall (last > first) where the first example is not the smallest, yes-rate on in-span non-example targets below the first example minus in-span non-example targets above it; observed > null means the model over-penalizes in-span numbers below the 'start' (people only reject numbers below the smallest example, not below the first one).
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
    inspan = (t > mn) & (t < mx) & ~is_ex & (last > first) & (first > mn)
    return rate(inspan & (t < first)) - rate(inspan & (t > first))
