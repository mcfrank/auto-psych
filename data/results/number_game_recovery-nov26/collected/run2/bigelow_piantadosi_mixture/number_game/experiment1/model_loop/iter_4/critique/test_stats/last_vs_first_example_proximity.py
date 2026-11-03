# name: last_vs_first_example_proximity
# description: Among multi-example sets and non-example targets, yes rate when the target is numerically closer to the last-shown example minus when closer to the first-shown; the model is order-invariant, so a nonzero observed-vs-null gap means a recency (positive) or primacy (negative) effect.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    fin = np.isfinite(ex)
    n = fin.sum(axis=1)
    first = ex[:, 0]
    idx = np.clip(n - 1, 0, ex.shape[1] - 1)
    last = ex[np.arange(len(ex)), idx]
    isex = (ex == t[:, None]).any(axis=1)
    df_ = np.abs(t - first); dl = np.abs(t - last)
    base = (n >= 2) & ~isex & (first != last)
    a = base & (dl < df_); b = base & (df_ < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
