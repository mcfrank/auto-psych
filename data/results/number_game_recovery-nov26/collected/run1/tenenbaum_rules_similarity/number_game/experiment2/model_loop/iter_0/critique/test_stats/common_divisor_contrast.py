# name: common_divisor_contrast
# description: For multi-example sets whose GCD is >= 3, yes rate on targets divisible by that GCD minus non-divisible targets; observed below null means the model over-relies on multiples-of-k rules (people generalise along common factors less sharply), above null means it under-relies on them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    e = np.nan_to_num(ex, nan=0.0).astype(np.int64)
    g = np.gcd.reduce(e, axis=1)
    t = df["target"].astype(np.int64).to_numpy()
    sel = (g >= 3) & (n >= 2)
    div = (t % np.maximum(g, 1)) == 0
    y = df["rating"].astype(float).to_numpy()
    a = sel & div; b = sel & ~div
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
