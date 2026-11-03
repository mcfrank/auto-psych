# name: last_vs_first_example_proximity
# description: For sets of 2+ examples, among non-example targets within 10 of some example, yes rate for targets strictly closer to the last-shown example than the first minus those strictly closer to the first; observed above null means a recency effect in generalisation (model is order-invariant), below means a primacy effect.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    first = E[:, 0]
    idx = np.clip(n - 1, 0, E.shape[1] - 1)
    last = E[np.arange(len(t)), idx]
    df_ = np.abs(t - first); dl = np.abs(t - last)
    dmin = np.nanmin(np.abs(E - t[:, None]), 1)
    m = (n >= 2) & ~isex & (dmin <= 10)
    return _rate(r, m & (dl < df_)) - _rate(r, m & (df_ < dl))
