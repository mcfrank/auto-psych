# name: edge_extension_contrast
# description: Among non-example targets outside the example range and 1-15 from the nearest example, yes rate when the target lies near the 1-100 boundary (<= 8 or >= 93) minus elsewhere; observed > null means people extend to the ends of the number line more than the model's bounded intervals produce, observed < null less.
def test_statistic(df):
    sets = df["number_set"].unique()
    arr = np.full((len(sets), 4), np.nan)
    for i, s in enumerate(sets):
        v = [int(x) for x in str(s).split(",")]
        arr[i, :len(v)] = v
    idx = pd.Index(sets).get_indexer(df["number_set"])
    ex = arr[idx]
    t = df["target"].astype(float).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    outside = (t < lo) | (t > hi)
    sel = outside & (d >= 1) & (d <= 15)
    edge = (t <= 8) | (t >= 93)
    r = df["rating"].astype(float).values
    a, b = sel & edge, sel & ~edge
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
