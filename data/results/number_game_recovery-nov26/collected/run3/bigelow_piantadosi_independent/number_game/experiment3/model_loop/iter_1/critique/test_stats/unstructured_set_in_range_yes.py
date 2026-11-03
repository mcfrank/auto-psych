# name: unstructured_set_in_range_yes
# description: Among 3-4 example sets with no shared parity, final digit or times table (2..12), yes rate on non-example targets inside the set's [min,max] range minus yes rate on targets outside it; observed below null means people treat wide "random-looking" sets less as a contiguous range than the model's interval hypotheses predict, above null means more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    exi = np.where(valid, ex, 0).astype(int)
    shared = np.zeros(len(df), dtype=bool)
    for m in range(2, 13):
        shared |= np.where(valid, exi % m == 0, True).all(axis=1)
    shared |= np.where(valid, exi % 2 == 1, True).all(axis=1)
    last = np.where(valid, exi % 10, -1)
    shared |= np.where(valid, last == last[:, :1], True).all(axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    isex = (ex == t[:, None]).any(axis=1)
    sel = (n >= 3) & ~shared & ~isex
    inside = sel & (t >= lo) & (t <= hi)
    outside = sel & ((t < lo) | (t > hi))
    r = df["rating"].astype(float).to_numpy()
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
