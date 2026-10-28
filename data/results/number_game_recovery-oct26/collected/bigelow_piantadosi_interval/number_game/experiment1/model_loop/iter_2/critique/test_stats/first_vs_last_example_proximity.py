# name: first_vs_last_example_proximity
# description: Among non-example targets of multi-example sets, yes rate when the target is strictly closer to the first-shown example than to the last-shown one minus yes rate when strictly closer to the last; observed differing from null means presentation order matters (positive = primacy, negative = recency), which the order-invariant model cannot produce.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = list(sets.unique())
    pad = np.full((len(uniq), 4), -1000, dtype=int)
    n_ex = np.zeros(len(uniq), dtype=int)
    for i, s in enumerate(uniq):
        a = [int(x) for x in s.split(",")][:4]
        pad[i, :len(a)] = a
        n_ex[i] = len(a)
    idx = sets.map({s: i for i, s in enumerate(uniq)}).to_numpy()
    ex = pad[idx]
    k = n_ex[idx]
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    is_ex = (ex == t[:, None]).any(axis=1)
    f = ex[:, 0]
    l = ex[np.arange(len(t)), np.maximum(k - 1, 0)]
    valid = (k >= 2) & ~is_ex
    dfirst = np.abs(t - f)
    dlast = np.abs(t - l)
    a = valid & (dfirst < dlast)
    b = valid & (dlast < dfirst)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
