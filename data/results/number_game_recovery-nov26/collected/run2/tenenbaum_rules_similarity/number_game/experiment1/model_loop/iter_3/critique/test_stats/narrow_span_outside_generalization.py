# name: narrow_span_outside_generalization
# description: Among multi-example sets and targets outside the example range, yes-rate when the examples span <= 48 minus when they span > 48; observed below null means people generalise beyond tightly clustered examples less than the model's Erlang-width intervals imply (relative to spread sets), above null means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    span = hi - lo
    outside = ((t < lo) | (t > hi)) & (n >= 2)
    a = outside & (span <= 48); b = outside & (span > 48)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
