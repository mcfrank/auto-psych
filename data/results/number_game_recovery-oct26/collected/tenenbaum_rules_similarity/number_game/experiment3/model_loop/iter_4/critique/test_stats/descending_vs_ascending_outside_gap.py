# name: descending_vs_ascending_outside_gap
# description: For sets of >=3 examples shown in strictly monotone order, yes-rate on non-example targets 1-15 outside the span for descending minus ascending presentation; observed differing from null means presentation direction changes outward generalization in a way the model's first-min/last-max edge marking misses (above null: ascending sets restrict more than modelled, or descending less).
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    d = np.diff(X, axis=1)
    dv = ~np.isnan(d)
    asc = (n >= 3) & ((d > 0) | ~dv).all(axis=1)
    desc = (n >= 3) & ((d < 0) | ~dv).all(axis=1)
    near = (dist >= 1) & (dist <= 15) & ~is_ex
    a = near & desc
    b = near & asc
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
