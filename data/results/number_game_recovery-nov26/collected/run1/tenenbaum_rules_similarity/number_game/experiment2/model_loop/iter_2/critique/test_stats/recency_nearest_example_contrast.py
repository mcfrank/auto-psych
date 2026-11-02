# name: recency_nearest_example_contrast
# description: For sets of >=2 examples and non-member targets, yes rate when the target is closer to the last-shown example than to the first-shown one minus the rate when it is closer to the first; observed above null means the most recent example pulls generalisation more than the order-blind model allows (below null: primacy).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    k = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(df)), np.maximum(k - 1, 0)]
    mem = np.any(ex == t[:, None], axis=1)
    dl = np.abs(t - last); dfst = np.abs(t - first)
    base = (k >= 2) & (~mem)
    a = base & (dl < dfst); b = base & (dfst < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
