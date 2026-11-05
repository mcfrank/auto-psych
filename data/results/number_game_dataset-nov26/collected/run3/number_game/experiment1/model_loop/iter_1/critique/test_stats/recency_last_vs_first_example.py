# name: recency_last_vs_first_example
# description: Among non-example targets in multi-example sets, yes rate when the target is strictly closer to the last-shown example than to the first-shown minus the reverse; the model is order-blind, so observed above null means a recency bias, below null a primacy bias.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    n = ex.notna().sum(axis=1).to_numpy()
    arr = ex.to_numpy()
    first = arr[:, 0]
    last = arr[np.arange(len(arr)), np.maximum(n - 1, 0)]
    tv = t.to_numpy()
    df_ = np.abs(tv - first); dl = np.abs(tv - last)
    base = (~inset).to_numpy() & (n >= 2)
    return m(pd.Series(base & (dl < df_), index=df.index)) - m(pd.Series(base & (df_ < dl), index=df.index))
