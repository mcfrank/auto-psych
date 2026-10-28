# name: narrow_span_edge_extension
# description: For 2+ example sets spanning <= 15 (e.g. 98,83; 3,7; 83,77), yes-rate for targets 1-10 beyond the example range minus targets >25 beyond it; observed above null means people extend tight clusters to nearby numbers more sharply than the model's intervals predict.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sp = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    v = sp.values.astype(float)
    n = np.isfinite(v).sum(axis=1)
    lo = np.nanmin(v, axis=1); hi = np.nanmax(v, axis=1)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    base = (n >= 2) & ((hi - lo) <= 15)
    near = base & (dist >= 1) & (dist <= 10)
    far = base & (dist > 25)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
