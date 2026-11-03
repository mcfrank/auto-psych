# name: magnitude_scaled_generalization
# description: Yes rate for non-example targets 4-12 from the nearest example when that example is > 50 minus when it is <= 50; observed > null means people generalise more broadly around large numbers (log/Weber-like similarity) than the model's linear intervals produce, observed < null the reverse.
def test_statistic(df):
    sets = df["number_set"].unique()
    arr = np.full((len(sets), 4), np.nan)
    for i, s in enumerate(sets):
        v = [int(x) for x in str(s).split(",")]
        arr[i, :len(v)] = v
    idx = pd.Index(sets).get_indexer(df["number_set"])
    ex = arr[idx]
    t = df["target"].astype(float).values[:, None]
    d = np.abs(ex - t)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    dmin = d[np.arange(len(d)), j]
    near = ex[np.arange(len(d)), j]
    r = df["rating"].astype(float).values
    sel = (dmin >= 4) & (dmin <= 12)
    hi, lo = sel & (near > 50), sel & (near <= 50)
    if hi.sum() == 0 or lo.sum() == 0:
        return 0.0
    return float(r[hi].mean() - r[lo].mean())
