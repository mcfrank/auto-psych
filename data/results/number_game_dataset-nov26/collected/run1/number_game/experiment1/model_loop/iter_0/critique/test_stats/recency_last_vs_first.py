# name: recency_last_vs_first
# description: Among sets with >=2 examples, yes-rate for targets closer to the last-shown example minus yes-rate for targets closer to the first-shown example; the model is order-invariant, so observed above null_mean means a recency effect (last example weighs more) the model misses, below means primacy.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = parts.notna().sum(axis=1).values
    vals = parts.values
    first = vals[:, 0]
    idx = np.clip(n - 1, 0, None)
    last = vals[np.arange(len(vals)), idx]
    t = df["target"].astype(float).values
    y = df["rating"].astype(float).values
    m = n >= 2
    dl = np.abs(t - last); dfi = np.abs(t - first)
    a = m & (dl < dfi); b = m & (dfi < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
