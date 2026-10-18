# name: recency_order_effect
# description: For sets with 2+ examples and non-example targets, yes rate when the target is nearer the last-shown example than the first minus when nearer the first; observed differing from null (~0) means presentation order matters to humans (positive = recency) which the order-invariant model cannot produce.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    n = (~np.isnan(ex)).sum(axis=1).astype(int)
    first = ex[:, 0]
    last = ex[np.arange(len(t)), np.maximum(n - 1, 0)]
    isin = (ex == t[:, None]).any(axis=1)
    df_, dl = np.abs(t - first), np.abs(t - last)
    base = (n >= 2) & ~isin
    a, b = base & (dl < df_), base & (df_ < dl)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
