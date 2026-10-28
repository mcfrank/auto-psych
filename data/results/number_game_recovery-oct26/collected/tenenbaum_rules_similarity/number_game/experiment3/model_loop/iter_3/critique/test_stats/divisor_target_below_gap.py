# name: divisor_target_below_gap
# description: For non-example targets 2 or more below the smallest example, yes-rate for targets that divide some example (factors, e.g. 5 or 19 after 95) minus non-divisor targets in the same below-span region; observed above null means people generalize to factors of the examples, a concept absent from the model's hypothesis space.
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
    Xf = np.nan_to_num(X, nan=0.0).astype(np.int64)
    valid = ~np.isnan(X)
    tt = np.maximum(t, 1)[:, None]
    divs = (valid & (np.mod(Xf, tt) == 0)).any(axis=1) & (t >= 2)
    below = (mn - t >= 2) & ~is_ex & (t >= 2)
    a = below & divs
    b = below & ~divs
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
