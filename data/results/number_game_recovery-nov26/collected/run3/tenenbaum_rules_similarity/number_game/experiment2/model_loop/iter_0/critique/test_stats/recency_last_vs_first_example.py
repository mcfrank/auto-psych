# name: recency_last_vs_first_example
# description: Among multi-example sets and targets not in the set, yes rate when the target is closer to the last-shown example than to the first-shown example minus yes rate when it is closer to the first; observed > null (null ~0, model is order-invariant) means a recency effect in generalisation, observed < null a primacy effect.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    first = E[:, 0]
    last = E[np.arange(len(E)), np.maximum(n - 1, 0)]
    in_set = np.any(E == t[:, None], axis=1)
    ok = (n >= 2) & ~in_set
    df1 = np.abs(t - first); dl = np.abs(t - last)
    r = df["rating"].to_numpy().astype(float)
    a = ok & (dl < df1); b = ok & (df1 < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
