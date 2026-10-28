# name: recency_last_vs_first_example
# description: Among multi-example trials, yes-rate when the target is numerically closer to the last-shown example minus when closer to the first-shown; the model is order-invariant, so observed above null means a recency effect (below null a primacy effect).
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    n = np.sum(~np.isnan(ex), axis=1)
    last = ex[np.arange(len(t)), np.maximum(n - 1, 0)]
    dl = np.abs(last - t); dfirst = np.abs(ex[:, 0] - t)
    ml = (n >= 2) & (dl < dfirst); mf = (n >= 2) & (dfirst < dl)
    if not ml.any() or not mf.any(): return 0.0
    return float(r[ml].mean() - r[mf].mean())
