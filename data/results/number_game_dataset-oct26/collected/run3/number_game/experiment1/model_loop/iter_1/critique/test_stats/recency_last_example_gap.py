# name: recency_last_example_gap
# description: Among sets of >=2 examples and non-member targets, yes-rate when the target is strictly closer to the last-shown example than to the first-shown one minus the reverse; the model is order-invariant, so observed above null means a recency (last-example) weighting the model lacks, below means primacy.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    t = df["target"].astype(int)
    arr = s.to_numpy(float)
    last = arr[np.arange(len(arr)), (n.to_numpy() - 1)]
    first = arr[:, 0]
    tv = t.to_numpy(float)
    dl, dfst = np.abs(tv - last), np.abs(tv - first)
    ok = (n.to_numpy() >= 2) & ~s.eq(t, axis=0).any(axis=1).to_numpy()
    r = df["rating"].to_numpy(float)
    a, b = ok & (dl < dfst), ok & (dfst < dl)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
