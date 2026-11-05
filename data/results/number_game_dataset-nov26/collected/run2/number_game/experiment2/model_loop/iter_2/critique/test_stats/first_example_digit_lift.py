# name: first_example_digit_lift
# description: Among multi-example sets without a shared final digit, yes-rate on non-example targets sharing the FIRST example's final digit (but not the last's) minus yes-rate on targets sharing no example's final digit; the model gives only the last example a lingering digit, so observed above null means a primacy digit effect the model misses.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    D = np.mod(E, 10)
    dmin = np.nanmin(D, axis=1); dmax = np.nanmax(D, axis=1)
    ok = (n >= 2) & (dmin != dmax) & (~inset)
    td = t % 10
    first = D[:, 0]
    last = D[np.arange(len(t)), n - 1]
    anyd = np.any(D == td[:, None], axis=1)
    a = ok & (td == first) & (td != last)
    b = ok & (~anyd)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
