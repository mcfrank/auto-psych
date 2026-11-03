# name: recency_last_vs_first_example
# description: In multi-example sets, among non-example targets within 10 of the first or last-shown example, yes rate when the target is closer to the last example minus closer to the first; observed above null means a recency effect the order-blind model cannot produce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    idx = np.clip(n - 1, 0, None).astype(int)
    first = ex[:, 0]
    last = ex[np.arange(len(t)), idx]
    df_ = np.abs(t - first); dl = np.abs(t - last)
    dmin = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    base = (n >= 2) & (dmin >= 1) & (np.minimum(df_, dl) <= 10) & (first != last)
    return rate(base & (dl < df_)) - rate(base & (df_ < dl))
