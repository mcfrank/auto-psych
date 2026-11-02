# name: prime_vs_odd_composite_contrast
# description: For sets of >=2 examples that are all primes (>2), among non-example odd targets, yes rate on prime targets minus odd composite targets; observed > null means people pick out "primes" over "odd numbers" more sharply than the model's prior/size principle predicts, observed < null means they blur primes into odd numbers more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    P = np.array([p for p in range(3, 101) if all(p % a for a in range(2, int(p ** 0.5) + 1))])
    allp = np.all(np.isin(E, P) | ~valid, axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    base = (n >= 2) & allp & ~in_set & (np.mod(t, 2) == 1) & (t > 1)
    a, b = base & np.isin(t, P), base & ~np.isin(t, P)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
