# name: recency_nearest_example
# description: For 2+ example sets and non-example targets, yes rate when the nearest example is the last-shown one minus yes rate when the nearest is the first-shown one; the model is order-invariant, so observed < null means people generalise less around the most recent example (primacy), observed > null means a recency advantage.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    n = ex.notna().sum(axis=1)
    inset = ex.eq(t, axis=0).any(axis=1)
    last = ex.ffill(axis=1).iloc[:, -1]
    first = ex.iloc[:, 0]
    dl = (t - last).abs()
    dfst = (t - first).abs()
    base = (n >= 2) & ~inset
    r = df["rating"].astype(float)
    a = r[base & (dl < dfst)]
    b = r[base & (dfst < dl)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
