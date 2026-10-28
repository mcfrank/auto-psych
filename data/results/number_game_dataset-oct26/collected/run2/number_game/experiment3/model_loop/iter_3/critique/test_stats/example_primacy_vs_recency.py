# name: example_primacy_vs_recency
# description: Among sets of 2+ examples, yes-rate when the target equals the first shown example minus when it equals the last shown example; observed > null means people endorse earlier examples as much as or more than later ones, i.e. the model's recency weighting over-favours the last example (and vice versa).
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
    m = k >= 2
    return rate(m & (t == first)) - rate(m & (t == last))
