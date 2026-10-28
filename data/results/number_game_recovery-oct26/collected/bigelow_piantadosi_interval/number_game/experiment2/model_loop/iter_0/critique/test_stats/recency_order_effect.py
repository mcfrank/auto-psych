# name: recency_order_effect
# description: Among non-example targets of sets with >= 2 examples, yes rate when the target is numerically closer to the last-shown example than to the first-shown one minus yes rate when closer to the first; observed > null means a recency effect (observed < null a primacy effect) that the order-blind model cannot produce (null ~ only stimulus-driven differences).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    tg = df["target"].astype(int).values.astype(float)
    k = (~np.isnan(ex)).sum(axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), np.maximum(k - 1, 0)]
    isex = (ex == tg[:, None]).any(axis=1)
    base = (k >= 2) & ~isex
    df_, dl = np.abs(tg - first), np.abs(tg - last)
    near_last = base & (dl < df_)
    near_first = base & (df_ < dl)
    r = df["rating"].values.astype(float)
    if near_last.sum() == 0 or near_first.sum() == 0:
        return 0.0
    return float(r[near_last].mean() - r[near_first].mean())
